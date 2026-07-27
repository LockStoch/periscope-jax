
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t
from _fp import utend_t, htend_t, qtend_t

from log import tcpu

from _dx import calc_obcs, calc_udry, \
                calc_hmap, tend_hadv, \
                calc_qmap, tend_qadv, \
                calc_u_ke, calc_u_pv, calc_perp, \
                tend_uadv, tend_upgf, tend_uflt, \
                calc_umix, calc_uwav, calc_hmix, \
                tend_umix, tend_hmix, \
                tend_ugeo, tend_utau, \
                tend_utde, calc_tide, calc_self
                
from mem import variables

def rhs_all_q(mesh, mats, flow, cnfg, hh_cell, uu_edge, qq_cell, 
                                               qq_tend):

    return qq_tend


def rhs_tde_d(mesh, mats, flow, cnfg, hh_cell, uu_edge):
    
#-- evaluate tide tendency diagnostics
    
    if cnfg.no_u_tend or \
            not cnfg.calc_tide or cnfg.rhs_stage != 1: 
        return

    zb_cell = variables.zb_cell 

    gravity = flow.gravity

    Xi_tide = variables.Xi_tide
    Xi_self = variables.Xi_self

    # tidal forcing
    Xi_tide = calc_tide(mesh, mats, cnfg, gravity, Xi_tide)
    
    Xi_self = calc_self(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gravity, Xi_self)

    return


def rhs_all_d(mesh, mats, flow, cnfg, hh_cell, uu_edge):

#-- evaluate full tendency diagnostics

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    hE_prev = flow.prev.hE_edge
    uE_prev = flow.prev.uE_edge
    
    hE_next = flow.next.hE_edge
    uE_next = flow.next.uE_edge

    ff_cell = variables.ff_cell
    ff_edge = variables.ff_edge
    ff_dual = variables.ff_vert

    ke_diss = variables.ke_diss

    ke_diss = set_x_vec(cnfg, ke_diss, 0.0)

    # construct vel^\perp
    vv_edge = calc_perp(
        mesh, mats, cnfg, uu_edge)
    
    # construct thickness
    hh_dual, hh_edge, hh_quad, hh_bias = \
              calc_hmap(mesh, mats, cnfg, gravity, hh_cell, 
                                          uu_edge, vv_edge)

    # do extrap. for OBCs
    hh_edge, uu_edge = \
              calc_obcs(mesh, mats, cnfg, hh_edge, uu_edge,
                                          gravity, 
                                          hE_prev, uE_prev, 
                                          hE_next, uE_next)

    # apply wet-dry limit
    # here, so shall feedback on nonlinear terms
    uu_edge, vv_edge, nu_thin = \
              calc_udry(mesh, mats, cnfg, hh_edge, 
                                          uu_edge, vv_edge)

    # nonlinear variables
    ke_cell, ke_bias = calc_u_ke(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, 
        uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, rv_wide, pv_wide, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    # shock sub-grid
    nu_shoc = calc_hmix(mesh, mats, cnfg, hh_cell, zb_cell,
                                          gravity,
                                          hh_edge,
                                          uu_edge, vv_edge)

    # waves sub-grid
    nu_wave = calc_uwav(mesh, mats, cnfg, hh_cell, zb_cell,
                                          gravity,
                                          hh_edge,
                                          uu_edge, vv_edge)

    # leith sub-grid
    nu_turb = calc_umix(mesh, mats, cnfg, rv_wide, rv_cell)

    return


def rhs_slw_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

    if cnfg.no_h_tend or not cnfg.calc_slow:return hh_tend

    return hh_tend


def rhs_fst_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate fast tendencies dH/dt = RHS(t,U,H)

    if cnfg.no_h_tend or not cnfg.calc_fast:return hh_tend

    zb_cell = variables.zb_cell

    gravity = flow.gravity
    
    hh_edge = variables.hh_edge

    # thickness advection
    hh_tend = tend_hadv(mesh, mats, cnfg, hh_edge, hh_cell,
                                          uu_edge,
                                          gravity, 
                                          hh_tend)

    nu_shoc = variables.nu_shoc

    # del^k dissipation
    hh_tend = tend_hmix(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gravity,
                                          nu_shoc, 
                                          hh_tend)

    return hh_tend


def rhs_all_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):
    
#-- evaluate full tendencies dH/dt = RHS(t,U,H)
    
    hh_tend = rhs_fst_h(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    hh_tend = rhs_slw_h(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend)
        
    return hh_tend


def rhs_slw_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):
    
#-- evaluate slow tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend or not cnfg.calc_slow:return uu_tend

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    Xi_prev = flow.prev.Xi_cell
    Xi_next = flow.next.Xi_cell

    Xi_tide = variables.Xi_tide

    Tu_prev = flow.prev.Tu_edge
    Tu_next = flow.next.Tu_edge
 
    ff_cell = variables.ff_cell
    ff_edge = variables.ff_edge
    ff_dual = variables.ff_vert

    hh_dual = variables.hh_dual
    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad

    ke_cell = variables.ke_cell
    pv_edge = variables.pv_edge
    
    # nonlinear advection
    uu_tend = tend_uadv(mesh, mats, cnfg, hh_edge, hh_quad,
                                          uu_edge,
                                          pv_edge, ke_cell,
                                 ff_dual, ff_edge, ff_cell, 
                                          uu_tend)

    # btr-bcl dissipation
    uu_tend = tend_uflt(mesh, mats, cnfg, uu_edge, hh_edge, 
                                          uu_tend)

    # external tend's here re: flux split
    # external geo-pot.
    uu_tend = tend_ugeo(mesh, mats, cnfg, Xi_prev, Xi_next,
                                          hh_cell,
                                          gravity,
                                          uu_tend)

    # tide+SAL geo-pot.
    uu_tend = tend_utde(mesh, mats, cnfg, Xi_tide,
                                          hh_cell, zb_cell,
                                          gravity,
                                          uu_tend)

    # external stresses
    uu_tend = tend_utau(mesh, mats, cnfg, Tu_prev, Tu_next,
                                          hh_edge,
                                          uu_tend)
    
    return uu_tend


def rhs_fst_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend or not cnfg.calc_fast:return uu_tend

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    hh_dual = variables.hh_dual
    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad
    
    nu_turb = variables.nu_turb
    nu_wave = variables.nu_wave
    nu_thin = variables.nu_thin

    # del^k dissipation
    uu_tend = tend_umix(mesh, mats, cnfg, hh_cell, hh_edge, 
                                          hh_quad, hh_dual, 
                                          uu_edge,
                                          nu_turb, nu_wave,
                                          nu_thin,
                                          uu_tend)

    return uu_tend


def rhs_pgf_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate hPGF tendencies dU/dt = RHS(t,U,H)

    if cnfg.no_u_tend or not cnfg.calc_fast:return uu_tend

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    Xi_self = variables.Xi_self

    # pressure gradient
    uu_tend = tend_upgf(mesh, mats, cnfg, hh_cell, zb_cell, 
                                          gravity, Xi_self,
                                          uu_tend)

    return uu_tend


def rhs_all_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate full tendencies dU/dt = RHS(t,U,H)

    uu_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend)

    uu_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend)

    return uu_tend


try:
    # load cython kernels, if compiled
    from _kt import _set_x_vec as set_x_vec
    from _kt import _cpy_x_vec as cpy_x_vec
    
except ImportError:
    raise RuntimeError("Cython back-end not found")


