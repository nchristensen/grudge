from meshmode.array_context import PyOpenCLArrayContext, ArrayContext
from pytools import memoize_method
from pytools.obj_array import make_obj_array
import loopy as lp
import pyopencl.array as cla
import grudge.loopy_dg_kernels as dgk
from grudge.grudge_tags import IsDOFArray, IsVecDOFArray, IsFaceDOFArray, IsVecOpArray, ParameterValue
from numpy import prod
import hjson
import numpy as np
import time
from os.path import expanduser
from copy import deepcopy

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

            # Use the same set of transformations for all of the differentiation kernels
            hjson_file = pkg_resources.open_text(dgk, "diff_1d_transform.hjson")
            #hjson_file = pkg_resources.open_text(dgk, "diff_{}d_transform.hjson".format(dim))

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
            # print(transformations)
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
        # Ease debugging, changing this temporarily
        return np.zeros(shape, dtype=dtype, order=self.order)
        #return np.empty(shape, dtype=dtype, order=self.order)

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


def async_transfer_args_to_device(program_args, kwargs, queue, allocator):
    cl_dict = {} # Transfer data to device asynchronously
    for arg in program_args:
        # Should probably do this even if they aren't, it will pre-allocate the output arrays then
        # But size is not necessarily known? Will fixing the parameters make the size known?

        #print(arg.name)
        #print(type(arg))
        if arg.name in kwargs:           
            key = arg.name
            if arg.is_output_only: # Output arrays only need to be allocated
                if isinstance(arg.tags, IsVecDOFArray):
                    obj_array_np = kwargs[key]
                    obj_array_cl = []
                    for np_a in obj_array_np:
                        cl_a = cla.empty(queue, np_a.shape, np_a.dtype, allocator=allocator)
                        obj_array_cl.append(cl_a)
                    cl_dict[key] = make_obj_array(obj_array_cl)
                else:
                    cl_dict[key] = cla.empty(queue, kwargs[key].shape, kwargs[key].dtype, allocator=allocator)
            else:
                if isinstance(arg.tags, IsVecDOFArray):
                    obj_array_np = kwargs[key]
                    obj_array_cl = []
                    for np_a in obj_array_np:
                        obj_array_cl.append(cla.to_device(queue, np_a, allocator=allocator, async_=True))
                    cl_dict[key] = make_obj_array(obj_array_cl)
                else:                
                    cl_dict[key] = cla.to_device(queue, kwargs[key], allocator=allocator, async_=True)
    return cl_dict

# Assume output args are in kwargs
def async_transfer_args_to_host(program_args, cl_dict, kwargs, queue):

    cnt = 0
    for arg in program_args:
        if arg.is_output_only:
            cnt += 1
    if cnt == 0:
        print("Cannot determine output argument")
        exit()

    for arg in program_args:
        # This won't work if the input is also the output, but I don't think any kernels re-use arrays
        if arg.is_output_only:
            if arg.name in kwargs:
                key = arg.name
                if isinstance(arg.tags, IsVecDOFArray):
                    for i, entry in enumerate(cl_dict[key]):
                        cl_dict[key][i].get_async(queue=queue, ary=kwargs[key][i])
                else:
                    cl_dict[key].get_async(queue=queue, ary=kwargs[key])
            # else:
            #    print("The output arg is not in kwargs. This case is not currently handled.")
            #    exit()

    return kwargs

var_dict = {}
def init_worker(queues, allocators, program_list, kwargs_list):
    var_dict["queues"] = queues
    var_dict["allocators"] = allocators
    var_dict["program_list"] = program_list
    var_dict["kwargs_list"] = kwargs_list
    #var_dict["devices"] = [queue.device for queue in queues]


class MultipleDispatchArrayContext(BaseNumpyArrayContext):
    
    def __init__(self, queues, order="C", allocators=None, read_in_weights=False, wait_event_queue_length=None):

        super().__init__(order=order)
        self.queues = queues
        self.contexts = (queue.context for queue in queues)
        if allocators is None:
            self.allocators = [None for i in range(len(queues))]
        else:
            self.allocators = allocators
        self.read_in_weights = read_in_weights
        self.weight_dict = {}
        if read_in_weights:
            self.populate_weights_dict()
        self.weight_dict = {'actx_special_exp': [0.73,0.27],
                            'diff_3_axis': [0.8,0.2],
                            'actx_special_sqrt': [0.71, 0.29],
                            'grudge_assign_0': [0.78, 0.22],
                            'diff_1_axis': [0.8, 0.2],
                            'diff_3_axis': [0.8, 0.2],
                            'grudge_assign_2': [0.82, 0.18],
                            'grudge_assign_1': [0.8, 0.2],
                            'diff_2_axis': [0.9, 0.1],
                            'elwise_linear': [0.07, 0.93]
                            }
        # TODO add queue length stuff as needed
        #if wait_event_queue_length is None:
        #    wait_event_queue_length = 10

    def populate_weights_dict(self):
        if self.read_in_weights:
            f = open(expanduser("~/.cache/grudge/weights"), "r")
            for l in f.readlines():
                tokens = l.split()
                assert len(tokens) == len(self.queues), "Invalid weight file given"
                id = tokens[0]
                weights = []
                for t in tokens[1:]:
                    weights.append(float(t))
                self.weight_dict[id] = weights
            f.close()
        else:
            print("Unable to read weights because the flag is set to False")

    def write_weights_out(self):
        f = open( expanduser("~/.cache/grudge/weights"), "w")
        for key in self.weight_dict:
            f.write(key)
            f.write(" ")
            weight_list = self.weight_dict[key]
            for w in weight_list:
                f.write(w)
                f.write(" ")
            f.write("\n")
        f.close()



    def call_loopy(self, program, **kwargs):
        
        #print(program.name)

        # flatten is only relevant to output
        # face_mass has strange shape
        # resample kernels have scattered reads and writes
        # nodes breaks with fixed parameters


        excluded = ["nodes",
                    "resample_by_picking", 
                    "resample_by_mat",
                    "face_mass", "flatten"]


        if "grudge_assign" in program.name:
            for arg in program.args:
                if isinstance(arg, lp.ValueArg):
                    arg.tags = ParameterValue(kwargs[arg.name])
                    del kwargs[arg.name]

        if program.name not in excluded:
            print(program.name)
            n = len(self.queues) # Total number of tasks to dole out round-robin to queues

            if program.args[0].name not in kwargs:
                print("Output not in kwargs. Adding it.")
                # assumes the second argument is a dof array input and the output is the same
                # shape and type as it
                kwargs[program.args[0].name] = np.empty_like(kwargs[program.args[1].name]) 

            dof_array_names = {}
            for arg in program.args:
                if arg.name == "nelements" and isinstance(arg.tags, ParameterValue):
                    nelem = arg.tags.value
                if arg.name in kwargs:
                    if isinstance(arg.tags, IsDOFArray):
                        nelem = kwargs[arg.name].shape[0]
                        dof_array_names[arg.name] = arg.tags
                    elif isinstance(arg.tags, IsVecDOFArray):
                        nelem = kwargs[arg.name][0].shape[0]
                        dof_array_names[arg.name] = arg.tags

            #either read weights from class dict or just generate regular weights and add to class dict
            weights = []
            if program.name not in self.weight_dict:
                frac = 1 / n
                for i in range(n):
                    weights.append(frac)
                self.weight_dict[program.name] = weights
            weights = self.weight_dict[program.name]

            split_points = [0]
            point = 0
            print(weights)
            for w in weights:
                point += round(w * nelem)
                split_points.append(point)
            split_points.append(nelem)
            split_points[-1] = nelem
            print(split_points)
            if any([x==0.0 for x in weights]):
                n = 1

            program_list = []
            for i in range(n):
                p = deepcopy(program)
                #p = program.copy()
                for arg in p.args:
                    if arg.name == "nelements" and arg.tags is not None:
                        nelem = split_points[i+1] - split_points[i]
                        arg.tags.value = nelem

                p = lp.set_options(p, no_numpy=False)
                p = self.transform_loopy_program(p)
                program_list.append(p)


            split_points = list(map(int, split_points))
            # Create separate views of input and output arrays
            # This assumes the output is in kwargs but that is not necessarily the case
            kwargs_list = []
            start = 0
            for i in range(n):
                # make separate kwargs for each n
                #from copy import deepcopy
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
            times = [0] * n
            queue_count = len(self.queues)


            #Execute and transfer back to host              
            # Break into three separate for loops
            # Transfer to device, kernel launch, transfer to host
            start_times = []
            cl_dicts = []

            # Set to false for performance testing.
            # Also disables dynamic load balancing.
            report_performance = False

            loop_start = time.process_time()            
            # Transfer data to devices asynchronously 
            for i in range(n):
 
                queue = self.queues[i % queue_count]
                start_times.append(time.process_time())                
                td_start = time.process_time()

                cl_dict =  async_transfer_args_to_device(program_list[i].args, kwargs_list[i], queue, self.allocators[i])
                cl_dicts.append(cl_dict)

                if report_performance:
                    queue.finish()                
                    td_dt = time.process_time() - td_start
                    print("Transfer to device: {}".format(td_dt))
                    times[i] += td_dt

            nvals = [0, 0]
            # Execute
            for i in range(n):

                start, end = split_points[i], split_points[i+1]
                nvals[i] = end - start
                queue = self.queues[i % queue_count]
                te_start = time.process_time()

                # Hopefully this waits for the transfers to complete before launching and then runs asynchronously
                #print(program_list[i])
                evt, result = program_list[i](queue, **cl_dicts[i], allocator=self.allocators[i])
                evt_list.append(evt) 
                result_list.append(result)

                if report_performance:                
                    queue.finish()
                    te_dt = time.process_time() - te_start
                    times[i] += te_dt

                # Original execution method
                ##evt, result = program_list[i](queue, **kwargs_list[i], allocator=self.allocators[i])

            # Transfer data back asynchronously
            for i in range(n):

                queue = self.queues[i % queue_count]
                th_start = time.process_time()

                print("prog_transfer:", program.name)
                async_transfer_args_to_host(program.args, cl_dicts[i], kwargs_list[i], queue) 

                if report_performance:
                    queue.finish()
                    th_dt = time.process_time() - th_start
                    times[i] += th_dt
                    print("Transfer to host: {}".format(th_dt))
                    dt = time.process_time() - start_times[i]
                    calc_bandwidth_usage(times[i], program_list[i], result_list[i], kwargs_list[i])
                    te_dt = times[i]
                    thpt = nvals[i] / te_dt
                    print(f"Execution time kernel {program_list[i].name}, device {i}, thpt: {thpt}items/sec: {te_dt}, items: {nvals[i]}")

                # evt times apparently do not cover transfers to and from device
                #if report_performance:
                    #evt = evt_list[i]
                    #evt.wait()
                    # print("Kernel time from queued: {}".format( (evt.profile.end - evt.profile.queued)/1e9))
                    # print("Kernel time from submitted: {}".format( (evt.profile.end - evt.profile.submit)/1e9))
                    # print("Kernel time from started: {}".format( (evt.profile.end - evt.profile.start)/1e9))

                # I'm pretty certain this is not occurring asynchronously. If it was, the loop times should
                # be similar. They actually resemble the execution time so I think they are not.
                # --Since changed to separate transfer and execution loops so need to reassess this
                #print(time.process_time() - start_time)
        
            for queue in self.queues:
                queue.finish()
            loop_end = time.process_time()
            loop_time = loop_end - loop_start

            # This number is meaningless when other calls to queue.finish exist inside the loops.
            print(f"Sum time of Loops: {loop_time}")
            if n == 1:
                print("PROG:", program.name, times, 0, self.weight_dict[program.name])
            if n > 1 and report_performance: # report_performance==True is needed for dynamic load balancing
                t_dev0, t_dev1 = times
                lib = (max(times) - min(times))/max(times)
                print("PROG:", program.name, times, lib, self.weight_dict[program.name])
                if lib > 0.15:
                    if t_dev0 > t_dev1:
                        self.weight_dict[program.name][0] -= 0.1
                        self.weight_dict[program.name][1] += 0.1
                    else:
                        self.weight_dict[program.name][0] += 0.1
                        self.weight_dict[program.name][1] -= 0.1

                    self.weight_dict[program.name][0] = max(round(self.weight_dict[program.name][0], 2), 0.1)
                    self.weight_dict[program.name][1] = max(round(self.weight_dict[program.name][1], 2), 0.1)
                    self.weight_dict[program.name][0] = min(self.weight_dict[program.name][0], 0.9)
                    self.weight_dict[program.name][1] = min(self.weight_dict[program.name][1], 0.9)
                    # ratio = t_dev0 / t_dev1
                    # f = ratio
            if not report_performance:
                print("Report performance = False, dynamic load balancing disabled")

            #cl.wait_for_events(evt_list)
            #for evt in evt_list:
            #    dt = evt.profile.end - evt.profile.start # Add the division back into calc_bandwidth_usage
            #    calc_bandwidth_usage(dt/1e9, program_list[i], result_list[i], kwargs_list[i])

            # If the output / result is in kwargs then can return that
            # If it is not then either need to add it to kwargs or 
            # Combine the results

            # This will fail for the differentiation kernel
            # There are three cases
            # 1: The output array is in kwargs with views in kwargs_list. If so, return that. (Done below).
            # 2: The output is a IsVecDOF. Then create a new dictionary with kwargs["result"][:]
            # with keys result0, result1, result2.
            # 3: The output is not in kwargs. If so, then either
            #       - Add it to kwargs (before creating kwargs_list) (Done for grudge_assign)
            #       - Merge the results after execution
            # Fix the result return value
            result_name = list(result.keys())[0]
            if result_name in kwargs:  
                result = kwargs[result_name]
                return None, {result_name: result} # There are multiple events so returning one is pointless
            elif "diff" in program.name:
                # Doesn't really matter, the argument rather than the return value is used 
                # in the calling function
                d = {}
                for i, array in enumerate(kwargs["result"]):
                    #print("RETURN SHAPE")
                    #print(array.shape)
                    d[f"result_s{i}"] = array

                return None, d

            else: 
                print(program.name)
                for arg in program.args:
                    if arg.is_output_only:
                        print(arg.name) 

                print("ERROR: Result view is not in kwarg. See above for what to do. (probably)")
                exit()

        else:

            program = lp.set_options(program, no_numpy=False)
            program = self.transform_loopy_program(program)
            evt, result = program(self.queues[0], **kwargs, allocator=self.allocators[0])

            # Does not include host->device, device->host transfer time
            evt.wait()
            dt = evt.profile.end - evt.profile.start
            calc_bandwidth_usage(dt/1e9, program, result, kwargs)
            
            #write new weights to the file, if flag is set
            
            return evt, result        

    # Somehow memoization cannot detect changes in tags
    #@memoize_method
    def transform_loopy_program(self, program):
        # Move this into transform code

        program = lp.set_options(program, "return_dict")
        for arg in program.args:
            if isinstance(arg.tags, ParameterValue):
                program = lp.fix_parameters(program, **{arg.name: arg.tags.value})
            # Fortran ordering is not possible with numpy array slicing because the
            # slices become discontinuous and PyOpenCL cannot load them to the GPU
            # We either need transformations for the element-contiguous case
            # or need to divide the arrays into contiguous blocks (harder to 
            # implement and makes dynamic work division hard.
            # The GPU performance is problably limited by the PCI express bandwidth
            # in any case so maybe this is not needed.
            """
            elif isinstance(arg.tags, IsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, IsVecDOFArray):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            elif isinstance(arg.tags, IsVecOpArray):
                program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, IsFaceDOFArray):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")
            """

        # Copied from GrudgeArrayContext
        # Should move this to separate file probably
        device_id = "NVIDIA Titan V"
        # This read could be slow
        transform_id = get_transformation_id(device_id)

        # Turn off these code transformations for now. They cause nans in 
        # the two axis diff kernel
        if "diff" in program.name:
            # TODO: Dynamically determine device id,
            pn = -1
            fp_format = None
            dim = -1
            for arg in program.args:
                if arg.name == "diff_mat":
                    dim = arg.shape[0]
                    pn = get_order_from_dofs(arg.shape[2])                    
                    fp_format = arg.dtype.numpy_dtype
                    break
            
            #program = lp.set_options(program, "write_cl")
            # Use 1D file for everything for now
            hjson_file = pkg_resources.open_text(dgk, "diff_1d_transform.hjson")
            #hjson_file = pkg_resources.open_text(dgk, "diff_{}d_transform.hjson".format(dim))

            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)#transform_id, fp_string, pn)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)

            

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

        # This is terrible for the element-contiguous layout
        elif "actx_special" in program.name:
            program = lp.split_iname(program, "i0", 256, outer_tag="g.0",
                                        inner_tag="l.0", slabs=(0, 1))
        elif "grudge_assign" in program.name:
            program = lp.split_iname(program, "iel", 256, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))

        return program

# vim: foldmethod=marker
