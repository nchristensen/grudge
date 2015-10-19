# -*- coding: utf8 -*-
"""Wave equation operators."""

from __future__ import division
from __future__ import absolute_import

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
from grudge.models import HyperbolicOperator
from grudge.models.second_order import CentralSecondDerivative
from meshmode.mesh import BTAG_ALL, BTAG_NONE
from grudge import sym, sym_flux
from pytools.obj_array import join_fields


# {{{ constant-velocity

class StrongWaveOperator(HyperbolicOperator):
    """This operator discretizes the wave equation
    :math:`\\partial_t^2 u = c^2 \\Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \\nabla \\cdot v = 0

        \partial_t v - c \\nabla u = 0

    The sign of :math:`v` determines whether we discretize the forward or the
    backward wave equation.

    :math:`c` is assumed to be constant across all space.
    """

    def __init__(self, c, dimensions, source_f=0,
            flux_type="upwind",
            dirichlet_tag=BTAG_ALL,
            dirichlet_bc_f=0,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_NONE):
        assert isinstance(dimensions, int)

        self.c = c
        self.dimensions = dimensions
        self.source_f = source_f

        if self.c > 0:
            self.sign = 1
        else:
            self.sign = -1

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.dirichlet_bc_f = dirichlet_bc_f

        self.flux_type = flux_type

    def flux(self):
        dim = self.dimensions
        w = sym_flux.FluxVectorPlaceholder(1+dim)
        u = w[0]
        v = w[1:]
        normal = sym_flux.normal(dim)

        flux_weak = join_fields(
                np.dot(v.avg, normal),
                u.avg * normal)

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            # see doc/notes/grudge-notes.tm
            flux_weak -= self.sign*join_fields(
                    0.5*(u.int-u.ext),
                    0.5*(normal * np.dot(normal, v.int-v.ext)))
        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

        flux_strong = join_fields(
                np.dot(v.int, normal),
                u.int * normal) - flux_weak

        return -self.c*flux_strong

    def sym_operator(self):
        d = self.dimensions

        w = sym.make_sym_vector("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------

        # dirichlet BCs -------------------------------------------------------
        dir_u = sym.BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = sym.BoundarizeOperator(self.dirichlet_tag) * v
        if self.dirichlet_bc_f:
            # FIXME
            from warnings import warn
            warn("Inhomogeneous Dirichlet conditions on the wave equation "
                    "are still having issues.")

            dir_g = sym.Field("dir_bc_u")
            dir_bc = join_fields(2*dir_g - dir_u, dir_v)
        else:
            dir_bc = join_fields(-dir_u, dir_v)

        # neumann BCs ---------------------------------------------------------
        neu_u = sym.BoundarizeOperator(self.neumann_tag) * u
        neu_v = sym.BoundarizeOperator(self.neumann_tag) * v
        neu_bc = join_fields(neu_u, -neu_v)

        # radiation BCs -------------------------------------------------------
        rad_normal = sym.normal(self.radiation_tag, d)

        rad_u = sym.BoundarizeOperator(self.radiation_tag) * u
        rad_v = sym.BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                0.5*(rad_u - self.sign*np.dot(rad_normal, rad_v)),
                0.5*rad_normal*(np.dot(rad_normal, rad_v) - self.sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = sym.nabla(d)
        flux_op = sym.get_flux_operator(self.flux())

        result = (
                - join_fields(
                    -self.c*np.dot(nabla, v),
                    -self.c*(nabla*u)
                    )
                +
                sym.InverseMassOperator() * (
                    flux_op(w)
                    + flux_op(sym.BoundaryPair(w, dir_bc, self.dirichlet_tag))
                    + flux_op(sym.BoundaryPair(w, neu_bc, self.neumann_tag))
                    + flux_op(sym.BoundaryPair(w, rad_bc, self.radiation_tag))
                    ))

        result[0] += self.source_f

        return result

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

    def max_eigenvalue(self, t, fields=None, discr=None):
        return abs(self.c)

# }}}


# {{{ variable-velocity

class VariableVelocityStrongWaveOperator(HyperbolicOperator):
    r"""This operator discretizes the wave equation
    :math:`\partial_t^2 u = c^2 \Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \nabla \cdot v = 0

        \partial_t v - c \nabla u = 0
    """

    def __init__(
            self, c, dimensions, source=0,
            flux_type="upwind",
            dirichlet_tag=BTAG_ALL,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_NONE,
            time_sign=1,
            diffusion_coeff=None,
            diffusion_scheme=CentralSecondDerivative()
            ):
        """*c* and *source* are optemplate expressions.
        """
        assert isinstance(dimensions, int)

        self.c = c
        self.time_sign = time_sign
        self.dimensions = dimensions
        self.source = source

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

        self.diffusion_coeff = diffusion_coeff
        self.diffusion_scheme = diffusion_scheme

    # {{{ flux ----------------------------------------------------------------
    def flux(self):
        dim = self.dimensions
        w = sym_flux.FluxVectorPlaceholder(2+dim)
        c = w[0]
        u = w[1]
        v = w[2:]
        normal = sym_flux.normal(dim)

        flux = self.time_sign*1/2*join_fields(
                c.ext * np.dot(v.ext, normal)
                - c.int * np.dot(v.int, normal),
                normal*(c.ext*u.ext - c.int*u.int))

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            flux += join_fields(
                    c.ext*u.ext - c.int*u.int,
                    c.ext*normal*np.dot(normal, v.ext)
                    - c.int*normal*np.dot(normal, v.int)
                    )
        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

        return flux

    # }}}

    def bind_characteristic_velocity(self, discr):
        from grudge.symbolic.operators import ElementwiseMaxOperator

        compiled = discr.compile(ElementwiseMaxOperator()(self.c))

        def do(t, w, **extra_context):
            return compiled(t=t, w=w, **extra_context)

        return do

    def sym_operator(self, with_sensor=False):
        d = self.dimensions

        w = sym.make_sym_vector("w", d+1)
        u = w[0]
        v = w[1:]

        flux_w = join_fields(self.c, w)

        # {{{ boundary conditions

        # Dirichlet
        dir_c = sym.BoundarizeOperator(self.dirichlet_tag) * self.c
        dir_u = sym.BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = sym.BoundarizeOperator(self.dirichlet_tag) * v

        dir_bc = join_fields(dir_c, -dir_u, dir_v)

        # Neumann
        neu_c = sym.BoundarizeOperator(self.neumann_tag) * self.c
        neu_u = sym.BoundarizeOperator(self.neumann_tag) * u
        neu_v = sym.BoundarizeOperator(self.neumann_tag) * v

        neu_bc = join_fields(neu_c, neu_u, -neu_v)

        # Radiation
        rad_normal = sym.make_normal(self.radiation_tag, d)

        rad_c = sym.BoundarizeOperator(self.radiation_tag) * self.c
        rad_u = sym.BoundarizeOperator(self.radiation_tag) * u
        rad_v = sym.BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                rad_c,
                0.5*(rad_u - self.time_sign*np.dot(rad_normal, rad_v)),
                0.5*rad_normal*(np.dot(rad_normal, rad_v) - self.time_sign*rad_u)
                )

        # }}}

        # {{{ diffusion -------------------------------------------------------
        from pytools.obj_array import with_object_array_or_scalar

        def make_diffusion(arg):
            if with_sensor or (
                    self.diffusion_coeff is not None and self.diffusion_coeff != 0):
                if self.diffusion_coeff is None:
                    diffusion_coeff = 0
                else:
                    diffusion_coeff = self.diffusion_coeff

                if with_sensor:
                    diffusion_coeff += sym.Field("sensor")

                from grudge.second_order import SecondDerivativeTarget

                # strong_form here allows the reuse the value of grad u.
                grad_tgt = SecondDerivativeTarget(
                        self.dimensions, strong_form=True,
                        operand=arg)

                self.diffusion_scheme.grad(grad_tgt, bc_getter=None,
                        dirichlet_tags=[], neumann_tags=[])

                div_tgt = SecondDerivativeTarget(
                        self.dimensions, strong_form=False,
                        operand=diffusion_coeff*grad_tgt.minv_all)

                self.diffusion_scheme.div(div_tgt,
                        bc_getter=None,
                        dirichlet_tags=[], neumann_tags=[])

                return div_tgt.minv_all
            else:
                return 0

        # }}}

        # entire operator -----------------------------------------------------
        nabla = sym.nabla(d)
        flux_op = sym.get_flux_operator(self.flux())

        return (
                - join_fields(
                    - self.time_sign*self.c*np.dot(nabla, v) - make_diffusion(u)
                    + self.source,

                    -self.time_sign*self.c*(nabla*u) - with_object_array_or_scalar(
                        make_diffusion, v)
                    )
                +
                sym.InverseMassOperator() * (
                    flux_op(flux_w)
                    + flux_op(sym.BoundaryPair(flux_w, dir_bc, self.dirichlet_tag))
                    + flux_op(sym.BoundaryPair(flux_w, neu_bc, self.neumann_tag))
                    + flux_op(sym.BoundaryPair(flux_w, rad_bc, self.radiation_tag))
                    ))

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

    def max_eigenvalue_expr(self):
        import grudge.symbolic as sym
        return sym.NodalMax()(sym.CFunction("fabs")(self.c))

# }}}


# vim: foldmethod=marker