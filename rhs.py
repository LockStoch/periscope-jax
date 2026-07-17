
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

from _dx import calc_hmap, tend_hadv, \
                calc_perp, \
                tend_upgf
                
from mem import variables


def rhs_tde_d(mesh, mats, flow, cnfg, hh_cell, uu_edge):
    
#-- evaluate tide tendency diagnostics
    
    return


def rhs_all_d(mesh, mats, flow, cnfg, hh_cell, uu_edge):

#-- evaluate full tendency diagnostics

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    # construct vel^\perp
    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)
    
    # construct thickness
    hh_dual, hh_edge, hh_quad, hh_bias = \
              calc_hmap(mesh, mats, cnfg, gravity, hh_cell, 
                                          uu_edge, 
                                          vv_edge)

    return


def rhs_slw_h(mesh, mats, flow, cnfg, hh_cell, uu_edge, hh_tend):

#-- evaluate slow tendencies dH/dt = RHS(t,U,H)

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

    return uu_tend


def rhs_fst_u(mesh, mats, flow, cnfg, hh_cell, uu_edge, uu_tend):

#-- evaluate fast tendencies dU/dt = RHS(t,U,H)

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


