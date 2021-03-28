from meshmode.array_context import PyOpenCLArrayContext, ArrayContext
from pytools import memoize_method
from pytools.obj_array import make_obj_array
import loopy as lp
import pyopencl.array as cla
import grudge.loopy_dg_kernels as dgk
from grudge.grudge_tags import IsDOFArray, IsVecDOFArray, IsFaceDOFArray, IsOpArray, IsVecOpArray, ParameterValue
from numpy import prod
import hjson
import numpy as np
import time

#from grudge.loopy_dg_kernels.run_tests import analyzeResult
import pyopencl as cl

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

def get_fp_string(dtype):
    return "FP64" if dtype == np.float64 else "FP32"

def get_order_from_dofs(dofs):
    dofs_to_order = {10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
    return dofs_to_order[dofs]

def calc_bandwidth_usage(dt, program, result, kwargs):
    #dt = dt / 1e9
    nbytes = 0
    if program.name == "resample_by_mat":
        n_to_nodes, n_from_nodes = kwargs["resample_mat"].shape
        nbytes = (kwargs["to_element_indices"].shape[0]*n_to_nodes +
                    n_to_nodes*n_from_nodes +
                    kwargs["from_element_indices"].shape[0]*n_from_nodes) * 8
    elif program.name == "resample_by_picking":
        # Double check this
        nbytes = kwargs["pick_list"].shape[0] * (kwargs["from_element_indices"].shape[0]
                    + kwargs["to_element_indices"].shape[0])*8
    else:
        #print(kwargs.keys())
        for key, val in kwargs.items():
            # output may be a list of pyopenclarrays or it could be a 
            # pyopenclarray. This prevents double counting (allowing
            # other for-loop to count the bytes in the former case)
            if key not in result.keys(): 
                try: 
                    nbytes += prod(val.shape)*8
                    #print(val.shape)
                except AttributeError:
                    nbytes += 0 # Or maybe 1*8 if this is a scalar
            #print(nbytes)
        #print("Output")
        #print(result.keys())
        for val in result.values():
            try:
                nbytes += prod(val.shape)*8
                #print(val.shape)
            except AttributeError:
                nbytes += 0 # Or maybe this is a scalar?

    bw = nbytes / dt / 1e9


    print("Kernel {}, Time {}, Bytes {}, Bandwidth {}".format(program.name, dt, nbytes, bw))


class GrudgeArrayContext(PyOpenCLArrayContext):

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def thaw(self, array):
        thawed = super().thaw(array)
        # Is the below necessary?
        if type(getattr(array, "tags", None)) == IsDOFArray:
            cq = thawed.queue
            # Should this be run through the array context
            #evt, out = self.call_loopy(ctof_knl, **{input: thawed})
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
        return thawed

    @memoize_method
    def transform_loopy_program(self, program):

        # This assumes arguments have only one tag
        for arg in program.args:
            if isinstance(arg.tags, IsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, IsVecDOFArray):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            elif isinstance(arg.tags, IsVecOpArray):
                program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, IsFaceDOFArray):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")
            elif isinstance(arg.tags, ParameterValue):
                program = lp.fix_parameters(program, **{arg.name: arg.tags.value})

        # Set no_numpy and return_dict options here?

        device_id = "NVIDIA Titan V"
        # This read could be slow
        transform_id = get_transformation_id(device_id)

        if "diff" in program.name:

            #program = lp.set_options(program, "write_cl")
            # TODO: Dynamically determine device id,
            # Rename this file
            pn = -1
            fp_format = None
            dim = -1
            for arg in program.args:
                if arg.name == "diff_mat":
                    dim = arg.shape[0]
                    pn = get_order_from_dofs(arg.shape[2])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            hjson_file = pkg_resources.open_text(dgk, "diff_{}d_transform.hjson".format(dim))

            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            #if pn <= 0 or not isinstance(fp_format, :
                #print("Need to specify a polynomial order and data type")
                # Should throw an error
                #exit()

            # Probably need to generalize this
            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)#transform_id, fp_string, pn)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)


            # Print the Code
            """
            platform = cl.get_platforms()
            my_gpu_devices = platform[1].get_devices(device_type=cl.device_type.GPU)
            #ctx = cl.create_some_context(interactive=True)
            ctx = cl.Context(devices=my_gpu_devices)
            kern = program.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
            code = lp.generate_code_v2(kern).device_code()
            prog = cl.Program(ctx, code)
            prog = prog.build()
            ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
            #errors="ignore") #Breaks pocl
            from bs4 import UnicodeDammit
            dammit = UnicodeDammit(ptx)
            print(dammit.unicode_markup)
            print(program.options)
            exit()
            """

        elif "elwise_linear" in program.name:
            hjson_file = pkg_resources.open_text(dgk, "elwise_linear_transform.hjson")
            pn = -1
            fp_format = None
            for arg in program.args:
                if arg.name == "mat":
                    pn = get_order_from_dofs(arg.shape[1])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)
        
        elif program.name == "nodes":
            # Only works for pn=3
            program = lp.split_iname(program, "iel", 64, outer_tag="g.0", slabs=(0,1))
            program = lp.split_iname(program, "iel_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1", slabs=(0,0))
            program = lp.split_iname(program, "idof_inner", 10, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))
                      
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
        elif program.name == "resample_by_mat":
            hjson_file = pkg_resources.open_text(dgk, "resample_by_mat.hjson")

            pn = 3 # This needs to  be not fixed
            fp_string = "FP64"
            
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            print(transformations)
            program = dgk.apply_transformation_list(program, transformations)

        elif "grudge_assign" in program.name or \
             "flatten" in program.name or \
             "resample_by_picking" in program.name or  \
             "face_mass" in program.name:
            # This is hardcoded. Need to move this to separate transformation file
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

        if False:#"opt_diff" in program.name:
            program = self.transform_loopy_program(program)

            dt = 0
            nruns = 10

            for i in range(2):
                evt, result = program(self.queue, **kwargs, allocator=self.allocator)
                #evt, result = super().call_loopy(program, **kwargs)
                evt.wait()
            for i in range(nruns):
                evt, result = program(self.queue, **kwargs, allocator=self.allocator)
                #evt, result = super().call_loopy(program, **kwargs)
                evt.wait()
                dt += evt.profile.end - evt.profile.start
            dt = dt / nruns
        else:
            evt, result = super().call_loopy(program, **kwargs)
            evt.wait()
            dt = evt.profile.end - evt.profile.start
        dt = dt / 1e9

        # Could probably just use program.args but maybe all
        # parameters are not set

        #print("Input")

        calc_bandwidth_usage(dt, program, kwargs)

        '''
        nbytes = 0
        if program.name == "resample_by_mat":
            n_to_nodes, n_from_nodes = kwargs["resample_mat"].shape
            nbytes = (kwargs["to_element_indices"].shape[0]*n_to_nodes +
                        n_to_nodes*n_from_nodes +
                        kwargs["from_element_indices"].shape[0]*n_from_nodes) * 8
        elif program.name == "resample_by_picking":
            # Double check this
            nbytes = kwargs["pick_list"].shape[0] * (kwargs["from_element_indices"].shape[0]
                        + kwargs["to_element_indices"].shape[0])*8
        else:
            #print(kwargs.keys())
            for key, val in kwargs.items():
                # output may be a list of pyopenclarrays or it could be a 
                # pyopenclarray. This prevents double counting (allowing
                # other for-loop to count the bytes in the former case)
                if key not in result.keys(): 
                    try: 
                        nbytes += prod(val.shape)*8
                        #print(val.shape)
                    except AttributeError:
                        nbytes += 0 # Or maybe 1*8 if this is a scalar
                #print(nbytes)
            #print("Output")
            #print(result.keys())
            for val in result.values():
                try:
                    nbytes += prod(val.shape)*8
                    #print(val.shape)
                except AttributeError:
                    nbytes += 0 # Or maybe this is a scalar?

        bw = nbytes / dt / 1e9


        print("Kernel {}, Time {}, Bytes {}, Bandwidth {}".format(program.name, dt, nbytes, bw))
        '''
        #if "opt_diff" in program.name: 
        #    exit()
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

    def __init__(self, order="C"):
        self.order = order
        super().__init__()

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype, order=self.order)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype, order=self.order)

    # Is this supposed to return a desired ordering?
    # If so, the input arrays need flags to distinguish
    # operators (c,c) from DOF arrays (f,f).
    def from_numpy(self, np_array: np.ndarray):
        return np_array

    def to_numpy(self, np_array: np.ndarray):
        return np_array
        #return np.array(np_array, order=self.order)

    def freeze(self, np_array: np.ndarray):
        return np_array
        #return np.array(np_array, order=self.order)

    def thaw(self, np_array: np.ndarray):
        #print(self.order)
        return np_array
        #return np.asfortranarray(np_array)
        #return np.array(np_array, order=self.order)
        #print(np_array.tags)
        #if isinstance(np_array.tags, IsDOFArray):
        #    return np.array(np_array, order=self.order)
        #else:
        #    return np.array(np_array)

    #def call_loopy(self, program, **kwargs):
    #    program = self.transform_loopy_program(program)
    #    evt, result = program(self.queue, **

    # TODO
    #def transform_loopy_program(self, program):
    #    pass

class MultipleDispatchArrayContext(BaseNumpyArrayContext):
    
    def __init__(self, queues, order="C", allocator=None, wait_event_queue_length=None):

        super().__init__(order=order)
        self.queues = queues
        self.contexts = (queue.context for queue in queues)
        self.allocator = allocator if allocator else None

        # TODO add queue length stuff as needed
        #if wait_event_queue_length is None:
        #    wait_event_queue_length = 10

    def call_loopy(self, program, **kwargs):
        print(program.name)
        # No code transformations enabled at this point
        #program = self.transform_loopy_program(program)
        
        #print(program.name)
        #print(kwargs)
        #for arg in kwargs.values():
        #    #if isinstance(arg, np.ndarray):
        #    #    setattr(arg, "offset", 0)
        #    print(type(arg))
        #print(program)
        #print(program.options)

        # Move this into transform code
        #program = lp.set_options(program, no_numpy=False)
        #for arg in program.args:
        #    if isinstance(arg.tags, ParameterValue):
        #        program = lp.fix_parameters(program, **{arg.name: arg.tags.value})


        #exit()

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

        excluded = ["nodes", "resample_by_picking", 
                    "grudge_assign_0", "grudge_assign_2", 
                    "grudge_assign_1", "resample_by_mat",
                    "face_mass"]

        if program.name not in excluded:
            #print(program.name)
            n = 4 # Total number of tasks to dole out round-robin to queues

            dof_array_names = {}
            for arg in program.args:
                if isinstance(arg.tags, IsDOFArray):
                    nelem = kwargs[arg.name].shape[0]
                    dof_array_names[arg.name] = arg.tags
                elif isinstance(arg.tags, IsVecDOFArray):
                    nelem = kwargs[arg.name][0].shape[0]
                    dof_array_names[arg.name] = arg.tags
            
            split_points = []
            step = nelem // n
            for i in range(n):
                split_points.append(step*i)
            split_points.append(nelem)

            program_list = []
            for i in range(n):
                p = program.copy()
                #if program.name == "elwise_linear":   
                for arg in p.args:
                    if arg.name == "nelements" and arg.tags is not None:
                        nelem = split_points[i+1] - split_points[i]
                        arg.tags.value = nelem

                p = self.transform_loopy_program(p)
                program_list.append(p)

            # Create separate views of input and output arrays
            # This assumes the output is in kwargs but that is not necessarily the case
            kwargs_list = []
            start = 0
            for i in range(n):
                # make separate kwargs for each n
                kwargs_list.append(kwargs.copy())  
                start = split_points[i]
                end = split_points[i+1]
                for (name, tag) in dof_array_names.items():
                    if isinstance(tag, IsDOFArray):
                        kwargs_list[i][name] = kwargs[name][start:end,:]
                    # This is more complicated because the there are arrays inside arrays
                    # and we need a new outer array filled with references to views of the original
                    #inner array
                    elif isinstance(tag, IsVecDOFArray):
                        # Create new outer array
                        naxes = kwargs[name].shape[0]
                        kwargs_list[i][name] = make_obj_array([kwargs[name][axis][start:end,:] for axis in range(naxes)])

            # Dispatch the tasks to each queue
            result_list = []
            evt_list = []
            times = []
            queue_count = len(self.queues)
            for i in range(n):
                start_time = time.time()
                '''
                for key, val in kwargs_list[i].items():
                    print(key)
                    print(val.shape)
                    print(val.flags["F"])
                '''

                evt, result = program_list[i](self.queues[i % queue_count], **kwargs_list[i], allocator=self.allocator)
                # evt times apparently do not cover transfers to and from device
                self.queues[i % queue_count].finish() # Comment during end-to-end timing
                dt = time.time() - start_time

                #print("Kwargs")
                #for key, val in kwargs.items():
                #    print(key)
                #    print(val.shape)
                #    #print(val.flags["F"])

                #print("Result")
                #for key, val in result.items():
                #    print(key)
                #    print(val.shape)
                #    #print(val.flags["F"])
                

                evt_list.append(evt)
                result_list.append(result)

                # If the output / result is in kwargs then can return that
                # If it is not then either need to add it to kwargs or 
                # Combine the results
  
                calc_bandwidth_usage(dt, program_list[i], result_list[i], kwargs_list[i])
    
           


            #cl.wait_for_events(evt_list)
            #for evt in evt_list:
            #    dt = evt.profile.end - evt.profile.start # Add the division back into calc_bandwidth_usage
            #    calc_bandwidth_usage(dt/1e9, program_list[i], result_list[i], kwargs_list[i])


            #for i in range(n):
            #    print(result_list[i])
            # kwargs["out"] should be completely filled by both operations
            #print(kwargs["out"])
            #print(np.sum(result["out"] - result2["out"))
            #exit()
            # This should return a combined result, not a single result

            # This will fail for the differentiation kernel
            # There are three cases
            # 1: The output array is in kwargs with views in kwargs_list. If so, return that. (Done below).
            # 2: The output is a IsVecDOF. Then create a new dictionary with kwargs["result"][:]
            # with keys result0, result1, result2.
            # 3: The output is not in kwargs. If so, then either
            #       - Add it to kwargs (before creating kwargs_list
            #       - Merge the results after execution

            result_name = list(result.keys())[0]
            if result_name in kwargs:
                result = kwargs[result_name]
                return evt, result
            elif "diff" in program.name:
                # Doesn't really matter, the argument rather than the return value is used 
                # in the calling function
                d = {}
                for i, array in enumerate(kwargs["result"]):
                    d["result_s{}".format(i)] = array
                return evt, d
            else: 
                print("ERROR: Result view is not in kwarg. See above for what to do. (probably)")
                exit()

        else:
            program = self.transform_loopy_program(program)

            #for key, val in kwargs.items():
            #    print(key)
            #    print(val.shape)
            #    print(val.flags["F"])

            evt, result = program(self.queues[0], **kwargs, allocator=self.allocator)
            #for key, val in kwargs.items():
            #    print(key)
            #    print(val.shape)
            #    print(val.flags["F"])
            #for key, val in result.items():
            #    print(key)
            #    print(val.shape)
            #    print(val.flags["F"])





            evt.wait()
            # Does not include host->device, device->host transfer time
            dt = evt.profile.end - evt.profile.start
            # This measures the bandwidth after the data reaches the GPU, not
            # The host <-> device transfers
            calc_bandwidth_usage(dt/1e9, program, result, kwargs)

            return evt, result        

    # Somehow memoization cannot detect changes in tags
    #@memoize_method
    def transform_loopy_program(self, program):
        # Move this into transform code
        program = lp.set_options(program, no_numpy=False)

        for arg in program.args:
            if isinstance(arg.tags, ParameterValue):
                program = lp.fix_parameters(program, **{arg.name: arg.tags.value})
            """
            elif isinstance(arg.tags, IsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, IsVecDOFArray):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            elif isinstance(arg.tags, IsVecOpArray):
                program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, IsOpArray):
                program = lp.tag_array_axes(program, arg.name, "c,c")
            elif isinstance(arg.tags, IsFaceDOFArray):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")
            """

        # Copied from GrudgeArrayContext
        # Should move this to separate file probably
        device_id = "NVIDIA Titan V"
        # This read could be slow
        transform_id = get_transformation_id(device_id)

        if "diff" in program.name:

            #program = lp.set_options(program, "write_cl")
            # TODO: Dynamically determine device id,
            # Rename this file
            pn = -1
            fp_format = None
            dim = -1
            for arg in program.args:
                if arg.name == "diff_mat":
                    dim = arg.shape[0]
                    pn = get_order_from_dofs(arg.shape[2])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            hjson_file = pkg_resources.open_text(dgk, "diff_{}d_transform.hjson".format(dim))

            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            #if pn <= 0 or not isinstance(fp_format, :
                #print("Need to specify a polynomial order and data type")
                # Should throw an error
                #exit()

            # Probably need to generalize this
            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)#transform_id, fp_string, pn)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)


            # Print the Code
            """
            platform = cl.get_platforms()
            my_gpu_devices = platform[1].get_devices(device_type=cl.device_type.GPU)
            #ctx = cl.create_some_context(interactive=True)
            ctx = cl.Context(devices=my_gpu_devices)
            kern = program.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
            code = lp.generate_code_v2(kern).device_code()
            prog = cl.Program(ctx, code)
            prog = prog.build()
            ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
            #errors="ignore") #Breaks pocl
            from bs4 import UnicodeDammit
            dammit = UnicodeDammit(ptx)
            print(dammit.unicode_markup)
            print(program.options)
            exit()
            """
        elif "elwise_linear" in program.name:
            hjson_file = pkg_resources.open_text(dgk, "elwise_linear_transform.hjson")
            pn = -1
            fp_format = None
            for arg in program.args:
                if arg.name == "mat":
                    pn = get_order_from_dofs(arg.shape[1])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)
        elif "actx_special" in program.name:
            program = lp.split_iname(program, "i0", 512, outer_tag="g.0",
                                        inner_tag="l.0", slabs=(0, 1))

        return program

# vim: foldmethod=marker
