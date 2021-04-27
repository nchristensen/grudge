__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from grudge.grudge_array_context import GrudgeArrayContext
from meshmode.array_context import PyOpenCLArrayContext  # noqa F401
from meshmode.dof_array import thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization, interior_trace_pair
from grudge.shortcuts import make_visualizer
from grudge.symbolic.primitives import TracePair


# {{{ wave equation bits

def wave_flux(discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(u.int.array_context, discr.normal(w_tpair.dd))

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal*u.avg,
            )

    # upwind
    flux_weak += flat_obj_array(
            0.5*(u.ext-u.int),
            0.5*normal*np.dot(normal, v.ext-v.int),
            )

    return discr.project(w_tpair.dd, "all_faces", c*flux_weak)

def print_allocated_arrays(id_lower_bound=0):
    # See what exists at this point    
    from pyopencl.array import Array
    print(len(Array.alloc_dict.keys()))
    """
    for key, value in Array.alloc_dict.items():
        print("{} {}".format(key, value[1]/1e9))
        if key >= id_lower_bound:
            for entry in value[0]:
                print(entry)
            print()
    """

    print('=====Condensed version==========')
    cdict = {}
    for id, (stack, size) in Array.alloc_dict.items():
        stack_frag = tuple(stack[:6])
        if stack_frag not in cdict:
            cdict[stack_frag] = np.array((size, 1))
        else:
            cdict[stack_frag] += np.array((size, 1))

    used = 0
    for stack, (sum_size, count) in cdict.items():
        gb_size = sum_size/1e9
        used += gb_size
        if gb_size > 0.00:
            print(f"SIZE {gb_size}, COUNT {count}")
            for entry in stack:
                print(entry)
            print()
    print("TOTAL USED MEMORY")
    print(used)
    print()          

def wave_operator(discr, c, w):
    from pyopencl import MemoryError
    from pyopencl.array import Array
    try:

        u = w[0]
        v = w[1:]

        """
        dir_u = discr.project("vol", BTAG_ALL, u)
        dir_v = discr.project("vol", BTAG_ALL, v)
        dir_bval = flat_obj_array(dir_u, dir_v)
        neg_dir_u = -dir_u; del dir_u
        dir_bc = flat_obj_array(neg_dir_u, dir_v)
        #print(discr._discr_scoped_subexpr_name_to_value.keys())
        div = discr.weak_div(v)

        #print(discr._discr_scoped_subexpr_name_to_value.keys())

        neg_c_div = (-c)*div; del div

        #print(discr._discr_scoped_subexpr_name_to_value.keys())
        grad = discr.weak_grad(u)

        neg_c_grad = (-c)*grad; del grad
        obj_array = flat_obj_array(neg_c_div, neg_c_grad)

        trace_pair1 = interior_trace_pair(discr, w)
        wave_flux1 = wave_flux(discr, c=c, w_tpair=trace_pair1)
        #del trace_pair1

        trace_pair2 = TracePair(BTAG_ALL, interior=dir_bval, exterior=dir_bc)
        wave_flux2 = wave_flux(discr, c=c, w_tpair=trace_pair2)
        #del trace_pair2
        #del dir_bc
        #del neg_dir_u
        #del dir_v
        #del dir_bval

        wave_flux_sum = wave_flux1 + wave_flux2;
        del wave_flux1
        del wave_flux2

        face_mass = discr.face_mass(wave_flux_sum)
        del wave_flux_sum

        inverse_arg = obj_array + face_mass
        del obj_array
        del face_mass
        del neg_c_div
        del neg_c_grad

        inverse_mass = discr.inverse_mass(inverse_arg)
        """

        dir_u = discr.project("vol", BTAG_ALL, u)
        dir_v = discr.project("vol", BTAG_ALL, v)
        dir_bval = flat_obj_array(dir_u, dir_v)
        dir_bc = flat_obj_array(-dir_u, dir_v)
     
        return (
                discr.inverse_mass(
                    flat_obj_array(
                        -c*discr.weak_div(v),
                        -c*discr.weak_grad(u)
                        )
                    +  # noqa: W504
                    discr.face_mass(
                        wave_flux(discr, c=c, w_tpair=interior_trace_pair(discr, w))
                        + wave_flux(discr, c=c, w_tpair=TracePair(
                            BTAG_ALL, interior=dir_bval, exterior=dir_bc))
                        ))
                    )

        from time import sleep
        sleep(3)
        print_allocated_arrays() 

        scoped = discr._discr_scoped_subexpr_name_to_value
        print(len(scoped.items()))
        print(scoped.keys())
        for value in scoped.values():
            #print(type(value))
            if isinstance(value, tuple):
                for entry in tuple:
                    print(entry.shape)
            else:
                print(value._data.shape)
        exit()

    except MemoryError:
        for key, value in Array.alloc_dict.items():
            print("{} {}".format(key, value[1]/1e9))
            for entry in value[0]:
                print(entry)
            print()
        exit() 


    return (result)



# }}}


def rk4_step(y, t, h, f):
    print("===================K1===================")
    k1 = f(t, y)
    print("===================K2===================")
    k2 = f(t+h/2, y + h/2*k1)
    print("===================K3===================")
    k3 = f(t+h/2, y + h/2*k2)
    print("===================K4===================")
    k4 = f(t+h, y + h*k3)
    print("===================UPDATE===================")
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(actx, discr, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, discr.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    from pyopencl.tools import ImmediateAllocator, MemoryPool
    memory_pool = MemoryPool(ImmediateAllocator(queue))
    #memory_pool.stop_holding()
    actx = GrudgeArrayContext(queue, allocator=memory_pool)

    dim = 3
    nel_1d = 47#2**6
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            coord_dtype=np.float64,
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(actx, mesh, order=order)

    fields = flat_obj_array(
            bump(actx, discr),
            [discr.zeros(actx) for i in range(discr.dim)]
            )

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    t = 0
    t_final = dt + dt
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(f"step: {istep} t: {t} L2: {discr.norm(fields[0])} "
                    f"sol max: {discr.nodal_max('vol', fields[0])}")
            vis.write_vtk_file("fld-wave-eager-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ])

        istep += 1
        t = istep*dt


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
