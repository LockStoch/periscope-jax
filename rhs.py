
""" SWE rhs. evaluations for various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _dx import tend_hadv, tend_upgf
from _dx import calc_hmap, calc_perp, calc_u_ke, calc_u_pv, tend_uadv

#-- Every function below takes (ops, phys, ...), mirroring main's
#-- uniform (mesh, mats, flow, cnfg, ...) convention -- even functions
#-- that don't currently touch phys still take it, so a future stage
#-- adding e.g. dissipation coefficients doesn't need a fresh round of
#-- signature changes. Each dispatcher unpacks whichever phys.* fields
#-- it needs into individually-named locals before calling the leaf
#-- _dx.py tendency functions, the same way main's own rhs_slw_u pulls
#-- zb_cell/gravity/ff_* out of the variables pool/flow before calling
#-- tend_uadv/calc_u_pv -- so those leaf functions keep the exact
#-- signatures already matched to main, unchanged by this bundling.


def rhs_slw_h(ops, phys, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    return hh_tend


def rhs_fst_h(ops, phys, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    # thickness advection
    hh_tend = tend_hadv(ops, hh_cell, uu_edge, hh_tend)

    return hh_tend


def rhs_all_h(ops, phys, hh_cell, uu_edge, hh_tend):

#-- evaluate full tendencies dH/dt = RHS(t,U,H)

    hh_tend = rhs_fst_h(ops, phys, hh_cell, uu_edge, hh_tend)

    hh_tend = rhs_slw_h(ops, phys, hh_cell, uu_edge, hh_tend)

    return hh_tend


def rhs_slw_u(ops, phys, hh_cell, uu_edge, pv_scheme, uu_tend):

#-- evaluate slow tendencies dU/dt = RHS(t,U,H)

    # momentum advection + Coriolis (Stage 1)
    ff_dual = phys.ff_dual
    ff_edge = phys.ff_edge
    ff_cell = phys.ff_cell
    uu_tiny = phys.uu_tiny
    pv_tiny = phys.pv_tiny
    pv_upwind = phys.pv_upwind

    # main: +1./2.*cnfg.time_step, computed at this same call site
    delta_t = 0.5 * phys.dt

    hh_dual, hh_edge, hh_quad = calc_hmap(ops, hh_cell)

    vv_edge = calc_perp(ops, uu_edge)

    ke_cell = calc_u_ke(ops, hh_cell, hh_quad, uu_edge, vv_edge)

    pv_edge = calc_u_pv(
        ops, uu_edge, vv_edge, ff_dual, ff_edge, ff_cell,
        delta_t, pv_tiny, uu_tiny, pv_upwind, pv_scheme)

    uu_tend = tend_uadv(
        ops, hh_edge, hh_quad, uu_edge, pv_edge, ke_cell,
        ff_edge, uu_tend)

    return uu_tend


def rhs_fst_u(ops, phys, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    return uu_tend


def rhs_pgf_u(ops, phys, hh_cell, uu_tend):

#-- evaluate hPGF tendencies dU/dt = RHS(t,U,H)

    zb_cell = phys.zb_cell
    gravity = phys.gravity

    # pressure gradient
    uu_tend = tend_upgf(ops, hh_cell, zb_cell, gravity, uu_tend)

    return uu_tend


def rhs_all_u(ops, phys, hh_cell, uu_edge, pv_scheme, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    uu_tend = rhs_slw_u(ops, phys, hh_cell, uu_edge, pv_scheme, uu_tend)

    uu_tend = rhs_fst_u(ops, phys, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_pgf_u(ops, phys, hh_cell, uu_tend)

    return uu_tend
