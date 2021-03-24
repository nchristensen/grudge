from meshmode.array_context import PyOpenCLArrayContext, ArrayContext
from meshmode.dof_array import IsDOFArray
from pytools.tag import Tag
from pytools import memoize_method
import loopy as lp
import pyopencl.array as cla
import grudge.loopy_dg_kernels as dgk
from numpy import prod
import hjson
import numpy as np

#from grudge.loopy_dg_kernels.run_tests import analyzeResult

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Use backported version for python < 3.7
    import importlib_resources as pkg_resources

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

def get_transformation_id(device_id):
    hjson_file = pkg_resources.open_text(dgk, "device_mappings.hjson") 
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od[device_id]

class VecIsDOFArray(Tag):
    pass


class FaceIsDOFArray(Tag):
    pass


class VecOpIsDOFArray(Tag):
    pass


# This was use in grudge_elwise_reduction but maybe is not needed.
#class IsOpArray(Tag):
#    pass

class GrudgeArrayContext(PyOpenCLArrayContext):

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def thaw(self, array):
        thawed = super().thaw(array)
        if type(getattr(array, "tags", None)) == IsDOFArray:
            cq = thawed.queue
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
            # May or may not be needed
            #thawed.tags = "dof_array"
        return thawed

    @memoize_method
    def transform_loopy_program(self, program):
        #print(program.name)

        for arg in program.args:
            if isinstance(arg.tags, IsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            #elif isinstance(arg.tags, IsOpArray):
            #    program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, VecIsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            #elif isinstance(arg.tags, VecOpIsDOFArray):
            #    program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, FaceIsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")

        if program.name == "opt_diff":
            # TODO: Dynamically determine device id,
            # Rename this file
            hjson_file = pkg_resources.open_text(dgk, "diff.hjson")
            device_id = "NVIDIA Titan V"
            transform_id = get_transformation_id(device_id)

            pn = -1
            fp_format = None
            dofs_to_order = {10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
            # Is this a list or a dictionary?
            for arg in program.args:
                if arg.name == "diff_mat":
                    pn = dofs_to_order[arg.shape[2]]
                    fp_format = arg.dtype.numpy_dtype
                    break

            #print(pn)
            #print(fp_format)
            #print(pn<=0)
            #exit()
            #print(type(fp_format) == None)
            #print(type(None) == None)
            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            #if pn <= 0 or not isinstance(fp_format, :
                #print("Need to specify a polynomial order and data type")
                # Should throw an error
                #exit()

            # Probably need to generalize this
            fp_string = "FP64" if fp_format == np.float64 else "FP32"
            indices = [transform_id, str(3), fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)#transform_id, fp_string, pn)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)
        elif "actx_special" in program.name:
            program = lp.split_iname(program, "i0", 512, outer_tag="g.0",
                                        inner_tag="l.0", slabs=(0, 1))
            #program = lp.split_iname(program, "i0", 128, outer_tag="g.0",
            #                           slabs=(0,1))
            #program = lp.split_iname(program, "i0_inner", 32, outer_tag="ilp",
            #                           inner_tag="l.0")
            #program = lp.split_iname(program, "i1", 20, outer_tag="g.1",
            #                           inner_tag="l.1", slabs=(0,0))
            #program2 = lp.join_inames(program, ("i1", "i0"), "i")
            #from islpy import BasicMap
            #m = BasicMap("[x,y] -> {[n0,n1]->[i]:}")
            #program2 = lp.map_domain(program, m)
            #print(program2)
            #exit()

            #program = super().transform_loopy_program(program)
            #print(program)
            #print(lp.generate_code_v2(program).device_code())
        elif "grudge_assign" in program.name or \
             "flatten" in program.name or \
             "resample" in program.name or  \
             "face_mass" in program.name:
            #program = lp.set_options(program, "write_cl")
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))
        else:
            program = super().transform_loopy_program(program)

        return program

    def call_loopy(self, program, **kwargs):
        evt, result = super().call_loopy(program, **kwargs)
        evt.wait()
        dt = (evt.profile.end - evt.profile.start) / 1e9
        nbytes = 0
        # Could probably just use program.args
        for val in kwargs.values():
            if isinstance(val, lp.ArrayArg): 
              nbytes += prod(val.shape)*8
        for val in result.values():
            nbytes += prod(val.shape)*8

        bw = nbytes / dt / 1e9

        print("Kernel {}, Time {}, Bytes {}, Bandwidth {}".format(program.name, dt, nbytes, bw))
        return evt, result

    '''
    def call_loopy(self, program, **kwargs):
        if program.name == "opt_diff":
            self.queue.finish()
            start = time.time()
            evt, result = program(self.queue, **kwargs, allocator=self.allocator)
            self.queue.finish()
            dt = time.time() - start
            _, nelem, n = program.args[0].shape
            print(program.args[0].shape)
            #print(lp.generate_code_v2(program).device_code())
            analyzeResult(n, n, nelem, 6144, 540, dt, 8)
            print(dt)
            # First is warmup
            self.queue.finish()
            start = time.time()
            evt, result = program(self.queue, **kwargs, allocator=self.allocator)
            self.queue.finish()
            dt = time.time() - start
            _, nelem, n = program.args[0].shape
            print(program.args[0].shape)
            #print(lp.generate_code_v2(program).device_code())
            analyzeResult(n, n, nelem, 6144, 540, dt, 8)
            print(dt)

            #exit()
            result = kwargs["result"]
        elif "actx_special" in program.name:
            print(program.name)
            start = time.time()
            evt, result = program(self.queue, **kwargs, allocator=self.allocator)
            self.queue.finish()
            dt = time.time() - start
            print(dt)
            d1, d2 = program.args[0].shape
            print((d1, d2))
            nbytes = d1*d2*8
            bandwidth = 2*(nbytes / dt) / 1e9
            print(bandwidth)
        else:
            evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        """
        if program.name == "opt_diff":
             self.queue.finish()
             start = time.time()
             evt, result = super().call_loopy(program, **kwargs)
             #evt, result = program(self.queue, **kwargs, allocator=self.allocator)
             self.queue.finish()
             dt = time.time() - start
             _, nelem, n = program.args[0].shape
             print(program.args[0].shape)
             #print(lp.generate_code_v2(program).device_code())
             analyzeResult(n, n, nelem, 6144, 540, dt, 8)
             print(dt)

             # First was warmup
             self.queue.finish()
             start = time.time()
             evt, result = program(self.queue, **kwargs, allocator=self.allocator)
             self.queue.finish()
             dt = time.time() - start
             _, nelem, n = program.args[0].shape
             print(program.args[0].shape)
             #print(lp.generate_code_v2(program).device_code())
             analyzeResult(n, n, nelem, 6144, 540, dt, 8)
             print(dt)


             #exit()
             result = kwargs["result"]
        else:
            evt, result = super().call_loopy(program, **kwargs)
             #evt, result = program(self.queue, **kwargs, allocator=self.allocator)
        """
        # """
        #start = time.time()
        evt, result = super().call_loopy(program, **kwargs)
        """
        if False:#program.name == "opt_diff":
             self.queue.finish()
             dt = time.time() - start
             _, nelem, n = program.args[0].shape
             print(program.args[0].shape)
             print(lp.generate_code_v2(program).device_code())
             analyzeResult(n, n, nelem, 6144, 540, dt, 8)
             exit()
        """
        # """

        return evt, result
        '''

class BaseNumpyArrayContext(ArrayContext):

    #def __init__(self):
    #    super().__init__()

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def from_numpy(self, np_array: np.ndarray):
        return np_array

    def to_numpy(self, np_array: np.ndarray):
        return np_array

    def freeze(self, np_array: np.ndarray):
        return np_array

    def thaw(self, np_array: np.ndarray):
        return np_array

    #def call_loopy(self, program, **kwargs):
    #    program = self.transform_loopy_program(program)
    #    evt, result = program(self.queue, **

    # TODO
    #def transform_loopy_program(self, program):
    #    pass

class MultipleDispatchArrayContext(BaseNumpyArrayContext):
    
    def __init__(self, queues, allocator=None, wait_event_queue_length=None):

        super().__init__()
        self.queues = queues
        self.contexts = (queue.context for queue in queues)
        self.allocator = allocator if allocator else None

        # TODO add queue length stuff as needed
        #if wait_event_queue_length is None:
        #    wait_event_queue_length = 10

    def call_loopy(self, program, **kwargs):
        print(program.name)
        #print(kwargs)
        #for arg in kwargs.values():
        #    #if isinstance(arg, np.ndarray):
        #    #    setattr(arg, "offset", 0)
        #    print(type(arg))
        #print(program)
        #print(program.options)
        program = lp.set_options(program, no_numpy=False)
        #exit()

        #for arg in program.args:
            #try:
            #    print(arg.shape)
            #except AttributeError:
            #    pass
            #print(arg.tags)
        for key, val in kwargs.items():
            try:
                #pass
                print(key)
                #print(val.shape)
            except AttributeError:
                pass
        

        # If it has a DOF Array, then split the dof array into two or more viewed and execute with each view
        # What about any loop bounds. Do they need to be reassigned?
        # Loopy limits, maybe use fix_parameters to adjust?
        # Experiment with sqrt function first

        """
        has_dof_array = False
        has_vec_dof_array = False
        has_face_dof_array = False
        for arg in program.args:
            if isinstance(arg.tags, IsDOFArray):
                has_dof_array = True
            elif isinstance(arg.tags, VecIsDOFArray):
                has_vec_dof_array = True
            elif isinstance(arg.tags, FaceIsDOFArray):
                has_face_dof_array = True
            #pass
            #print(arg.tags)
        """

        excluded = ["nodes", "resample_by_picking", "grudge_assign_0", "grudge_assign_2", "grudge_assign_1", "resample_by_mat","face_mass","opt_diff","diff"]

        if program.name not in excluded:
            print(program.name)
            n = 4
            
            dof_array_names = []
            for arg in program.args:
                if isinstance(arg.tags, IsDOFArray):
                    nelem = kwargs[arg.name].shape[0]
                    dof_array_names.append(arg.name)

            split_points = []
            step = nelem // n
            for i in range(0, nelem, step):
                split_points.append(i)
            split_points.append(nelem)
            print(split_points)

            # make separate kwargs for each n
            kwargs_list = []
            program_list = []
            for i in range(n):
                kwargs_list.append(kwargs.copy())

            for i in range(n):
                # FIXME: What if it is called something else
                if "nelements" in program.args:
                    nelem = split_points[i+1] - split_points[i]
                    p = lp.set_parameters(program, nelements=nelem)
                    program_list.append(p)
                else:
                    print("nelements is not the name")
                    program_list.append(program.copy())

            # Create separate views of input and output arrays
            start = 0
            for i in range(n-1):
                start = split_points[i]
                end = split_points[i+1]
                for name in dof_array_names:
                    kwargs_list[i-1][name] = kwargs[name][start:end,:]

            # Dispatch the tasks to each queue
            result_list = []
            evt_list = []
            queue_count = len(self.queues)
            for i in range(n):
                evt, result = program_list[i](self.queues[i % queue_count], **kwargs_list[i], allocator=self.allocator)
                evt_list.append(evt)
                result_list.append(result)

            

            #for i in range(n):
            #    print(result_list[i])
            # kwargs["out"] should be completely filled by both operations
            #print(kwargs["out"])
            #print(np.sum(result["out"] - result2["out"))
            #exit()
            return evt, result        

        else:
            evt, result = program(self.queues[0], **kwargs, allocator=self.allocator)
            return evt, result        

    @memoize_method
    def transform_loopy_program(self, program):
        pass 
    
# vim: foldmethod=marker
