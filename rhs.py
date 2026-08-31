
""" SWE rhs. evaluations for various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _dx import tend_hadv, tend_upgf


def rhs_slw_h(ops, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return hh_tend


def rhs_fst_h(ops, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    # thickness advection
    hh_tend = tend_hadv(ops, hh_cell, uu_edge, hh_tend)

    return hh_tend


def rhs_all_h(ops, hh_cell, uu_edge, hh_tend):

#-- evaluate full tendencies dH/dt = RHS(t,U,H)

    hh_tend = rhs_fst_h(ops, hh_cell, uu_edge, hh_tend)

    hh_tend = rhs_slw_h(ops, hh_cell, uu_edge, hh_tend)

    return hh_tend


def rhs_slw_u(ops, hh_cell, uu_edge, uu_tend):

#-- evaluate slow tendencies dU/dt = RHS(t,U,H)

    return uu_tend


def rhs_fst_u(ops, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    return uu_tend


def rhs_pgf_u(ops, hh_cell, zb_cell, gravity, uu_tend):

#-- evaluate hPGF tendencies dU/dt = RHS(t,U,H)

    # pressure gradient
    uu_tend = tend_upgf(ops, hh_cell, zb_cell, gravity, uu_tend)

    return uu_tend


def rhs_all_u(ops, hh_cell, uu_edge, zb_cell, gravity, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    uu_tend = rhs_slw_u(ops, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_fst_u(ops, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_pgf_u(ops, hh_cell, zb_cell, gravity, uu_tend)

    return uu_tend
