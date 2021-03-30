[1mdiff --git a/examples/advection/surface.py b/examples/advection/surface.py[m
[1mindex 2ab36d5..5fb7139 100644[m
[1m--- a/examples/advection/surface.py[m
[1m+++ b/examples/advection/surface.py[m
[36m@@ -25,7 +25,7 @@[m [mimport os[m
 import numpy as np[m
 import pyopencl as cl[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from meshmode.dof_array import thaw, flatten[m
 [m
 from grudge import bind, sym[m
[36m@@ -94,7 +94,7 @@[m [mclass Plotter:[m
 def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=False):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     # {{{ parameters[m
 [m
[1mdiff --git a/examples/advection/var-velocity.py b/examples/advection/var-velocity.py[m
[1mindex 9aa4d5b..877b78c 100644[m
[1m--- a/examples/advection/var-velocity.py[m
[1m+++ b/examples/advection/var-velocity.py[m
[36m@@ -25,7 +25,7 @@[m [mimport numpy as np[m
 [m
 import pyopencl as cl[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from meshmode.dof_array import thaw, flatten[m
 [m
 from grudge import bind, sym[m
[36m@@ -91,7 +91,7 @@[m [mclass Plotter:[m
 def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=False):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     # {{{ parameters[m
 [m
[1mdiff --git a/examples/advection/weak.py b/examples/advection/weak.py[m
[1mindex 7977a43..70d39a9 100644[m
[1m--- a/examples/advection/weak.py[m
[1m+++ b/examples/advection/weak.py[m
[36m@@ -26,7 +26,6 @@[m [mimport numpy.linalg as la[m
 [m
 import pyopencl as cl[m
 [m
[31m-#from meshmode.array_context import PyOpenCLArrayContext[m
 from grudge.grudge_array_context import GrudgeArrayContext[m
 from meshmode.dof_array import thaw, flatten[m
 [m
[36m@@ -93,7 +92,6 @@[m [mclass Plotter:[m
 def main(ctx_factory, dim=2, order=4, visualize=False):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    #actx = PyOpenCLArrayContext(queue)[m
     actx = GrudgeArrayContext(queue)[m
 [m
     # {{{ parameters[m
[1mdiff --git a/examples/dagrt-fusion.py b/examples/dagrt-fusion.py[m
[1mindex 51033db..93060e5 100755[m
[1m--- a/examples/dagrt-fusion.py[m
[1m+++ b/examples/dagrt-fusion.py[m
[36m@@ -60,7 +60,7 @@[m [mimport dagrt.language as lang[m
 import pymbolic.primitives as p[m
 [m
 from meshmode.dof_array import DOFArray[m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 [m
 import grudge.symbolic.mappers as gmap[m
 import grudge.symbolic.operators as sym_op[m
[36m@@ -484,7 +484,7 @@[m [mdef get_wave_component(state_component):[m
 def test_stepper_equivalence(ctx_factory, order=4):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dims = 2[m
 [m
[36m@@ -748,7 +748,7 @@[m [mclass ExecutionMapperWithMemOpCounting(ExecutionMapperWrapper):[m
 def test_assignment_memory_model(ctx_factory):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     _, discr = get_wave_op_with_discr(actx, dims=2, order=3)[m
 [m
[36m@@ -776,7 +776,7 @@[m [mdef test_assignment_memory_model(ctx_factory):[m
 def test_stepper_mem_ops(ctx_factory, use_fusion):[m
     cl_ctx = ctx_factory()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dims = 2[m
 [m
[36m@@ -947,7 +947,7 @@[m [mdef test_stepper_timing(ctx_factory, use_fusion):[m
     queue = cl.CommandQueue([m
             cl_ctx,[m
             properties=cl.command_queue_properties.PROFILING_ENABLE)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dims = 3[m
 [m
[36m@@ -1070,7 +1070,7 @@[m [melse:[m
 def problem_stats(order=3):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     with open_output_file("grudge-problem-stats.txt") as outf:[m
         _, dg_discr_2d = get_wave_op_with_discr([m
[36m@@ -1095,7 +1095,7 @@[m [mdef problem_stats(order=3):[m
 def statement_counts_table():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     fused_stepper = get_example_stepper(actx, use_fusion=True)[m
     stepper = get_example_stepper(actx, use_fusion=False)[m
[36m@@ -1186,7 +1186,7 @@[m [mdef mem_ops_results(actx, dims):[m
 def scalar_assignment_percent_of_total_mem_ops_table():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     result2d = mem_ops_results(actx, 2)[m
     result3d = mem_ops_results(actx, 3)[m
[1mdiff --git a/examples/geometry.py b/examples/geometry.py[m
[1mindex 0affdc8..3080332 100644[m
[1m--- a/examples/geometry.py[m
[1m+++ b/examples/geometry.py[m
[36m@@ -27,13 +27,13 @@[m [mimport numpy as np  # noqa[m
 import pyopencl as cl[m
 from grudge import sym, bind, DGDiscretizationWithBoundaries, shortcuts[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 [m
 [m
 def main(write_output=True):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     from meshmode.mesh.generation import generate_warped_rect_mesh[m
     mesh = generate_warped_rect_mesh(dim=2, order=4, n=6)[m
[1mdiff --git a/examples/maxwell/cavities.py b/examples/maxwell/cavities.py[m
[1mindex 7de38a7..f1e6f27 100644[m
[1m--- a/examples/maxwell/cavities.py[m
[1m+++ b/examples/maxwell/cavities.py[m
[36m@@ -27,7 +27,6 @@[m [mimport numpy as np[m
 import pyopencl as cl[m
 [m
 from grudge.grudge_array_context import GrudgeArrayContext[m
[31m-#from meshmode.array_context import PyOpenCLArrayContext[m
 [m
 from grudge.shortcuts import set_up_rk4[m
 from grudge import sym, bind, DGDiscretizationWithBoundaries[m
[36m@@ -42,7 +41,6 @@[m [mdef main(dims, write_output=True, order=2):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
     actx = GrudgeArrayContext(queue)[m
[31m-    #actx = PyOpenCLArrayContext(queue)[m
 [m
     from meshmode.mesh.generation import generate_regular_rect_mesh[m
     mesh = generate_regular_rect_mesh([m
[1mdiff --git a/examples/wave/var-propagation-speed.py b/examples/wave/var-propagation-speed.py[m
[1mindex 6a07565..f80b5e1 100644[m
[1m--- a/examples/wave/var-propagation-speed.py[m
[1m+++ b/examples/wave/var-propagation-speed.py[m
[36m@@ -28,13 +28,13 @@[m [mimport pyopencl as cl[m
 from grudge.shortcuts import set_up_rk4[m
 from grudge import sym, bind, DGDiscretizationWithBoundaries[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 [m
 [m
 def main(write_output=True, order=4):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dims = 2[m
     from meshmode.mesh.generation import generate_regular_rect_mesh[m
[1mdiff --git a/examples/wave/wave-eager-mpi.py b/examples/wave/wave-eager-mpi.py[m
[1mindex b6ca9ae..ea33706 100644[m
[1m--- a/examples/wave/wave-eager-mpi.py[m
[1m+++ b/examples/wave/wave-eager-mpi.py[m
[36m@@ -27,7 +27,7 @@[m [mimport pyopencl as cl[m
 [m
 from pytools.obj_array import flat_obj_array[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from meshmode.dof_array import thaw[m
 [m
 from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa[m
[36m@@ -121,7 +121,7 @@[m [mdef bump(actx, discr, t=0):[m
 def main():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     comm = MPI.COMM_WORLD[m
     num_parts = comm.Get_size()[m
[1mdiff --git a/examples/wave/wave-eager-var-velocity.py b/examples/wave/wave-eager-var-velocity.py[m
[1mindex 194a1d6..0c58723 100644[m
[1m--- a/examples/wave/wave-eager-var-velocity.py[m
[1m+++ b/examples/wave/wave-eager-var-velocity.py[m
[36m@@ -27,7 +27,7 @@[m [mimport pyopencl as cl[m
 [m
 from pytools.obj_array import flat_obj_array[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from meshmode.dof_array import thaw[m
 [m
 from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa[m
[36m@@ -133,7 +133,7 @@[m [mdef bump(actx, discr, t=0, width=0.05, center=None):[m
 def main():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dim = 2[m
     nel_1d = 16[m
[1mdiff --git a/examples/wave/wave-eager.py b/examples/wave/wave-eager.py[m
[1mindex 5c2f1d7..7cac91d 100644[m
[1m--- a/examples/wave/wave-eager.py[m
[1m+++ b/examples/wave/wave-eager.py[m
[36m@@ -28,7 +28,6 @@[m [mimport pyopencl as cl[m
 from pytools.obj_array import flat_obj_array[m
 [m
 from grudge.grudge_array_context import GrudgeArrayContext[m
[31m-#from meshmode.array_context import PyOpenCLArrayContext[m
 from meshmode.dof_array import thaw[m
 [m
 from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa[m
[36m@@ -115,7 +114,7 @@[m [mdef bump(actx, discr, t=0):[m
 def main():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = GrudgeArrayContext(queue)  # PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dim = 3[m
     nel_1d = 16[m
[1mdiff --git a/examples/wave/wave-min-mpi.py b/examples/wave/wave-min-mpi.py[m
[1mindex a65582b..b67ce9f 100644[m
[1m--- a/examples/wave/wave-min-mpi.py[m
[1m+++ b/examples/wave/wave-min-mpi.py[m
[36m@@ -25,7 +25,7 @@[m [mTHE SOFTWARE.[m
 [m
 import numpy as np[m
 import pyopencl as cl[m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from grudge.shortcuts import set_up_rk4[m
 from grudge import sym, bind, DGDiscretizationWithBoundaries[m
 from mpi4py import MPI[m
[36m@@ -34,7 +34,7 @@[m [mfrom mpi4py import MPI[m
 def main(write_output=True, order=4):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     comm = MPI.COMM_WORLD[m
     num_parts = comm.Get_size()[m
[1mdiff --git a/examples/wave/wave-min.py b/examples/wave/wave-min.py[m
[1mindex c5d0e60..9d1b4b6 100644[m
[1m--- a/examples/wave/wave-min.py[m
[1m+++ b/examples/wave/wave-min.py[m
[36m@@ -26,7 +26,7 @@[m [mTHE SOFTWARE.[m
 [m
 import numpy as np[m
 import pyopencl as cl[m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 from grudge.shortcuts import set_up_rk4[m
 from grudge import sym, bind, DGDiscretizationWithBoundaries[m
 [m
[36m@@ -34,7 +34,7 @@[m [mfrom grudge import sym, bind, DGDiscretizationWithBoundaries[m
 def main(write_output=True, order=4):[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     dims = 2[m
     from meshmode.mesh.generation import generate_regular_rect_mesh[m
[1mdiff --git a/grudge/loopy_dg_kernels/__init__.py b/grudge/loopy_dg_kernels/__init__.py[m
[1mindex c4fcdf8..8b0a464 100644[m
[1m--- a/grudge/loopy_dg_kernels/__init__.py[m
[1m+++ b/grudge/loopy_dg_kernels/__init__.py[m
[36m@@ -297,7 +297,6 @@[m [mdef apply_transformation_list(knl, transformations):[m
     # bounds[m
     print(knl)[m
     for t in transformations:[m
[31m-        print("HERE")[m
         print(t)[m
         func = function_mapping[t[0]][m
         args = [knl] + t[1][m
[1mdiff --git a/test/test_mpi_communication.py b/test/test_mpi_communication.py[m
[1mindex dedc35d..3357edf 100644[m
[1m--- a/test/test_mpi_communication.py[m
[1m+++ b/test/test_mpi_communication.py[m
[36m@@ -29,7 +29,7 @@[m [mimport numpy as np[m
 import pyopencl as cl[m
 import logging[m
 [m
[31m-from meshmode.array_context import PyOpenCLArrayContext[m
[32m+[m[32mfrom grudge.grudge_array_context import GrudgeArrayContext[m
 [m
 logger = logging.getLogger(__name__)[m
 logging.basicConfig()[m
[36m@@ -42,7 +42,7 @@[m [mfrom grudge.shortcuts import set_up_rk4[m
 def simple_mpi_communication_entrypoint():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis[m
 [m
[36m@@ -100,7 +100,7 @@[m [mdef simple_mpi_communication_entrypoint():[m
 def mpi_communication_entrypoint():[m
     cl_ctx = cl.create_some_context()[m
     queue = cl.CommandQueue(cl_ctx)[m
[31m-    actx = PyOpenCLArrayContext(queue)[m
[32m+[m[32m    actx = GrudgeArrayContext(queue)[m
 [m
     from mpi4py import MPI[m
     comm = MPI.COMM_WORLD[m
[1mdiff --git a/test/test_results.txt b/test/test_results.txt[m
[1mdeleted file mode 100644[m
[1mindex b880e8e..0000000[m
[1m--- a/test/test_results.txt[m
[1m+++ /dev/null[m
[36m@@ -1,10617 +0,0 @@[m
[31m-============================= test session starts ==============================[m
[31m-platform linux -- Python 3.7.1, pytest-4.2.1, py-1.7.0, pluggy-0.8.1[m
[31m-rootdir: /home/njchris2/Workspace/grudge, inifile:[m
[31m-collected 99 items[m
[31m-[m
[31m-test_grudge.py ..................ssss..s..s.FFFFFFFFFFFFFFFFFFFFFFFFFFFF [ 57%][m
[31m-FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFsssFFF                               [100%][m
[31m-[m
[31m-=================================== FAILURES ===================================[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c748>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c710>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 3, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 4, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-strong-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-strong-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-strong-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'strong'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-weak-disk-mesh_pars0] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'disk', mesh_pars = [0.1, 0.05], op_type = 'weak'[m
[31m-flux_type = 'upwind', order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-weak-rect2-mesh_pars1] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect2', mesh_pars = [4, 8], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_advec[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5-upwind-weak-rect3-mesh_pars2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1020c780>[m
[31m-mesh_name = 'rect3', mesh_pars = [4, 6], op_type = 'weak', flux_type = 'upwind'[m
[31m-order = 5, visualize = False[m
[31m-[m
[31m-    @pytest.mark.parametrize(("mesh_name", "mesh_pars"), [[m
[31m-        ("disk", [0.1, 0.05]),[m
[31m-        ("rect2", [4, 8]),[m
[31m-        ("rect3", [4, 6]),[m
[31m-        ])[m
[31m-    @pytest.mark.parametrize("op_type", ["strong", "weak"])[m
[31m-    @pytest.mark.parametrize("flux_type", ["upwind"])[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    # test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'[m
[31m-    def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,[m
[31m-            order, visualize=False):[m
[31m-        """Test whether 2D advection actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:219: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3fd0>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3fd0>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-5] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3fd0>[m
[31m-order = 5[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f98>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f98>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-5] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f98>[m
[31m-order = 5[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f60>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f60>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_convergence_maxwell[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-5] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3f60>[m
[31m-order = 5[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [3, 4, 5])[m
[31m-    def test_convergence_maxwell(ctx_factory,  order):[m
[31m-        """Test whether 3D maxwells actually converges"""[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:340: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3470>[m
[31m-order = 2[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3470>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8f3470>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e5518>[m
[31m-order = 2[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e5518>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e5518>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e58d0>[m
[31m-order = 2[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e58d0>[m
[31m-order = 3[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_improvement_quadrature[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-4] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c1e8e58d0>[m
[31m-order = 4[m
[31m-[m
[31m-    @pytest.mark.parametrize("order", [2, 3, 4])[m
[31m-    def test_improvement_quadrature(ctx_factory, order):[m
[31m-        """Test whether quadrature improves things and converges"""[m
[31m-        from meshmode.mesh.generation import generate_regular_rect_mesh[m
[31m-        from grudge.models.advection import VariableCoefficientAdvectionOperator[m
[31m-        from pytools.convergence import EOCRecorder[m
[31m-        from pytools.obj_array import join_fields[m
[31m-        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory[m
[31m-    [m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:414: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_bessel[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c10237ba8>[m
[31m-[m
[31m-    def test_bessel(ctx_factory):[m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:489: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_bessel[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c10237b70>[m
[31m-[m
[31m-    def test_bessel(ctx_factory):[m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:489: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m- test_bessel[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>] [m
[31m-[m
[31m-ctx_factory = <pyopencl.tools.pytest_generate_tests_for_pyopencl.<locals>.ContextFactory object at 0x7f4c10237be0>[m
[31m-[m
[31m-    def test_bessel(ctx_factory):[m
[31m->       cl_ctx = cl.create_some_context()[m
[31m-[m
[31m-test_grudge.py:489: [m
[31m-_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [m
[31m-[m
[31m-interactive = False, answers = None[m
[31m-[m
[31m-    def create_some_context(interactive=None, answers=None):[m
[31m-        import os[m
[31m-        if answers is None:[m
[31m-            if "PYOPENCL_CTX" in os.environ:[m
[31m-                ctx_spec = os.environ["PYOPENCL_CTX"][m
[31m-                answers = ctx_spec.split(":")[m
[31m-    [m
[31m-            if "PYOPENCL_TEST" in os.environ:[m
[31m-                from pyopencl.tools import get_test_platforms_and_devices[m
[31m-                for plat, devs in get_test_platforms_and_devices():[m
[31m-                    for dev in devs:[m
[31m-                        return Context([dev])[m
[31m-    [m
[31m-        if answers is not None:[m
[31m-            pre_provided_answers = answers[m
[31m-            answers = answers[:][m
[31m-        else:[m
[31m-            pre_provided_answers = None[m
[31m-    [m
[31m-        user_inputs = [][m
[31m-    [m
[31m-        if interactive is None:[m
[31m-            interactive = True[m
[31m-            try:[m
[31m-                import sys[m
[31m-                if not sys.stdin.isatty():[m
[31m-                    interactive = False[m
[31m-            except Exception:[m
[31m-                interactive = False[m
[31m-    [m
[31m-        def cc_print(s):[m
[31m-            if interactive:[m
[31m-                print(s)[m
[31m-    [m
[31m-        def get_input(prompt):[m
[31m-            if answers:[m
[31m-                return str(answers.pop(0))[m
[31m-            elif not interactive:[m
[31m-                return ''[m
[31m-            else:[m
[31m-                user_input = input(prompt)[m
[31m-                user_inputs.append(user_input)[m
[31m-                return user_input[m
[31m-    [m
[31m-        # {{{ pick a platform[m
[31m-    [m
[31m-        platforms = get_platforms()[m
[31m-    [m
[31m-        if not platforms:[m
[31m-            raise Error("no platforms found")[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose platform:")[m
[31m-                for i, pf in enumerate(platforms):[m
[31m-                    cc_print("[%d] %s" % (i, pf))[m
[31m-    [m
[31m-            answer = get_input("Choice [0]:")[m
[31m-            if not answer:[m
[31m-                platform = platforms[0][m
[31m-            else:[m
[31m-                platform = None[m
[31m-                try:[m
[31m-                    int_choice = int(answer)[m
[31m-                except ValueError:[m
[31m-                    pass[m
[31m-                else:[m
[31m-                    if 0 <= int_choice < len(platforms):[m
[31m-                        platform = platforms[int_choice][m
[31m-    [m
[31m-                if platform is None:[m
[31m-                    answer = answer.lower()[m
[31m-                    for i, pf in enumerate(platforms):[m
[31m-                        if answer in pf.name.lower():[m
[31m-                            platform = pf[m
[31m-                    if platform is None:[m
[31m-                        raise RuntimeError("input did not match any platform")[m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        # {{{ pick a device[m
[31m-    [m
[31m-        devices = platform.get_devices()[m
[31m-    [m
[31m-        def parse_device(choice):[m
[31m-            try:[m
[31m-                int_choice = int(choice)[m
[31m-            except ValueError:[m
[31m-                pass[m
[31m-            else:[m
[31m-                if 0 <= int_choice < len(devices):[m
[31m-                    return devices[int_choice][m
[31m-    [m
[31m-            choice = choice.lower()[m
[31m-            for i, dev in enumerate(devices):[m
[31m-                if choice in dev.name.lower():[m
[31m-                    return dev[m
[31m-            raise RuntimeError("input did not match any device")[m
[31m-    [m
[31m-        if not devices:[m
[31m-            raise Error("no devices found")[m
[31m-        elif len(devices) == 1:[m
[31m-            pass[m
[31m-        else:[m
[31m-            if not answers:[m
[31m-                cc_print("Choose device(s):")[m
[31m-                for i, dev in enumerate(devices):[m
[31m-                    cc_print("[%d] %s" % (i, dev))[m
[31m-    [m
[31m-            answer = get_input("Choice, comma-separated [0]:")[m
[31m-            if not answer:[m
[31m-                devices = [devices[0]][m
[31m-            else:[m
[31m-                devices = [parse_device(i) for i in answer.split(",")][m
[31m-    [m
[31m-        # }}}[m
[31m-    [m
[31m-        if user_inputs:[m
[31m-            if pre_provided_answers is not None:[m
[31m-                user_inputs = pre_provided_answers + user_inputs[m
[31m-            cc_print("Set the environment variable PYOPENCL_CTX='%s' to "[m
[31m-                    "avoid being asked again." % ":".join(user_inputs))[m
[31m-    [m
[31m-        if answers:[m
[31m-            raise RuntimeError("not all provided choices were used by "[m
[31m-                    "create_some_context. (left over: '%s')" % ":".join(answers))[m
[31m-    [m
[31m->       return Context(devices)[m
[31m-E       pyopencl._cl.RuntimeError: Context failed: OUT_OF_HOST_MEMORY[m
[31m-[m
[31m-../../../miniconda3/envs/dgfem/lib/python3.7/site-packages/pyopencl/__init__.py:1462: RuntimeError[m
[31m-=============================== warnings summary ===============================[m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>][m
[31m-test/test_grudge.py::test_1d_mass_mat_trig[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'pthread-Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Portable Computing Language'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-1][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-2][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_tri_diff_mat[<context factory for <pyopencl.Device 'Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz' on 'Intel(R) OpenCL'>-3][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-  /home/njchris2/miniconda3/envs/dgfem/lib/python3.7/site-packages/grudge/symbolic/compiler.py:998: DeprecationWarning: temp_var_type should be Optional() if no temporary, not None. This usage will be disallowed soon.[m
[31m-    ("*", "any"),[m
[31m-[m
[31m-test/test_grudge.py::test_inverse_metric[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-2][m
[31m-  /home/njchris2/miniconda3/envs/dgfem/lib/python3.7/site-packages/loopy/transform/iname.py:678: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working[m
[31m-    from collections import Iterable[m
[31m-[m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-strong-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-3-upwind-weak-rect3-mesh_pars2][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-test/test_grudge.py::test_convergence_advec[<context factory for <pyopencl.Device 'TITAN V' on 'NVIDIA CUDA'>-4-upwind-strong-rect2-mesh_pars1][m
[31m-  /home/njchris2/miniconda3/envs/dgfem/lib/python3.7/site-packages/grudge/symbolic/compiler.py:998: DeprecationWarning: temp_var_type should be Optional(None) if unspecified, not auto. This usage will be disallowed soon.[m
[31m-    ("*", "any"),[m
[31m-[m
[31m--- Docs: https://docs.pytest.org/en/latest/warnings.html[m
[31m-======= 67 failed, 23 passed, 9 skipped, 321 warnings in 132.64 seconds ========[m
