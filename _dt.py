
import time
import numpy as np

""" SWE time integration via various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda, Jeremy Lilly
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

from _dx import calc_drag

from rhs import rhs_tde_d, rhs_all_d, \
                rhs_all_u, rhs_slw_u, rhs_fst_u, \
                rhs_pgf_u, \
                rhs_all_h, rhs_slw_h, rhs_fst_h

def mark_time(cnfg, flow, time):

#-- Update simulation time and interp. on forc. tendencies

    if (flow.xx_time is not None):
    
    #-- linear interp. between prev and next
        prev = flow.xx_time [flow.prev.step]
        next = flow.xx_time [flow.next.step]
        cnfg.timeisnow = time
        cnfg.frc_blend = min(1.0, (time - prev) / 
                                  (next - prev) )

    else:
    
    #-- piecewise const. data, so do nothing
        cnfg.timeisnow = time
        cnfg.frc_blend = reals_t(0.0)
    
    if (cnfg.forc_ramp > 0.0):
        cnfg.frc_start = min(1.0, time / cnfg.forc_ramp)
    else:
        cnfg.frc_start = reals_t(1.0)

    return cnfg


def init_step(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,
              qq_cell):

#-- Set initial estimate of time step size, via CFL bounds

    if (cnfg.time_step <= 0.0):
        cnfg.time_step = cfl_adapt(
            mesh, mats, cnfg, 
            flow.gravity, hh_thin=cnfg.wetdry_h0 * 100.0,
            rk_scal=1.0, rk_adv_=4.0/3.0,
            cfl_num=cnfg.cfl_limit, dt_prev=0,
            dt_jump=cnfg.dt_margin,
            hh_prev=hh_cell,hh_cell=hh_cell,
            uu_prev=uu_edge,uu_edge=uu_edge)

    return cnfg
    

def init_RKFB(cnfg):

#-- Initialise coefficients for user time-stepping schemes

    if   ("RK33" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.301666666666667, 0.316666666666667,
            0.366666666666667
            ] )

    else:#"RK43" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.000000000000000, 0.500000000000000,
            0.500000000000000, 0.000000000000000
            ] )

    return cnfg


def step_eqns(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell):

#-- A single time-step - via user-defined method of choice

    for ssub in range(cnfg.dt_cycles):

        hh_cell, uu_edge, qq_cell = step_try_(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, qq_cell)

        if (cnfg.next_step < 0.0):
            cnfg.time_step = -cnfg.next_step
            print (f">REDO: {-cnfg.next_step:+.2E}")
        else: 
            break

    return hh_cell, uu_edge, qq_cell


def step_try_(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell):

#-- A single time-step - via user-defined method of choice

    if   ("RK33" in cnfg.integrate):

        hh_cell, uu_edge, hb_cell = step_RK33(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, None, None)

    elif ("RK43" in cnfg.integrate):

        hh_cell, uu_edge, hb_cell = step_RK43(
            mesh, mats, 
            flow, cnfg, hh_cell, uu_edge, None, None)

    return hh_cell, uu_edge, qq_cell
    
    
def step_bnds(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              qq_cell,
              hh_min_, hh_max_,     # up/lo bounds
              uu_min_, uu_max_,
              qq_min_, qq_max_,
              zt_rms_,
     ke_ave_, ke_rms_, ke_max_,
     dk_ave_, dk_rms_, dk_max_):
              
#-- Expand the min./max. status for each degree of freedom
    
    if ("hh_cell" in cnfg.stat_vars): \
    hh_min_, hh_max_ = \
        bnd_x_vec(cnfg, hh_cell, hh_min_, hh_max_)

    if ("uu_edge" in cnfg.stat_vars): \
    uu_min_, uu_max_ = \
        bnd_x_vec(cnfg, uu_edge, uu_min_, uu_max_)

    if ("qq_cell" in cnfg.stat_vars): \
    qq_min_, qq_max_ = \
        bnd_x_vec(cnfg, qq_cell, qq_min_, qq_max_)

    zb_cell = variables.zb_cell

    if ("zt_cell" in cnfg.stat_vars): \
    zt_rms_ = \
        nrm_z_vec(cnfg, zb_cell, hh_cell, zt_rms_)

    ke_cell = variables.ke_cell  # from previous

    if ("ke_cell" in cnfg.stat_vars): \
    ke_ave_, ke_rms_, ke_max_ = \
        nrm_x_vec(cnfg, ke_cell, ke_ave_, ke_rms_, 
                                 ke_max_)

    ke_diss = variables.ke_diss

    if ("cd_diss" in cnfg.stat_vars or 
        "nu_diss" in cnfg.stat_vars): \
    dk_ave_, dk_rms_, dk_max_ = \
        nrm_x_vec(cnfg, ke_diss, dk_ave_, dk_rms_, 
                                 dk_max_)
   
    return hh_min_, hh_max_, uu_min_, uu_max_, \
           qq_min_, qq_max_, zt_rms_, \
           ke_ave_, ke_rms_, ke_max_, \
           dk_ave_, dk_rms_, dk_max_


def step_RK33(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              Rh_cell, Ru_edge):    # slow tend.

#-- A 3-stage 3rd/2nd-order RK scheme:
#-- D. Engwirda (2025): 3- and 4-stage forward-backward
#-- Runge-Kutta methods for geophysical flows

#-- drag included via a 2nd-order IMEX scheme

    start_t = cnfg.timeisnow

    k1_step = 1.0 / 3.0 * cnfg.time_step
    k2_step = 2.0 / 3.0 * cnfg.time_step
    k3_step = 1.0 / 1.0 * cnfg.time_step
    dt_step = 1.0 / 1.0 * cnfg.time_step

    isFB = 1.0 * ("FB" in cnfg.integrate)

    gravity = flow.gravity
    
    dz_drag = variables.dz_drag
    c1_edge = variables.c1_edge
    c2_edge = variables.c2_edge
    z0_edge = variables.z0_edge
    n0_edge = variables.n0_edge

    cd_edge = variables.cd_save

    hb_cell = variables.hb_cell
    h1_cell = variables.h1_cell  # kth storage in h
    h2_cell = variables.h2_cell
    h3_cell = variables.h3_cell

    uk_edge = variables.uk_edge  # low storage in u
    
    hk_tend = variables.hh_tend  # zero by construction
    uk_tend = variables.uu_tend
    h0_tend = variables.h0_tend
    u0_tend = variables.u0_tend

#-- 1st RK + FB stage

    ttic = time.time()

    uk_edge = cpy_x_vec(  cnfg, uu_edge, uk_edge)

    cnfg.rhs_stage = 1
    cnfg.time_step = k1_step
    cnfg = mark_time(
        cnfg, flow, start_t + 0. / 1.0 * dt_step)

    BETA = cnfg.fb_weight[0] * isFB
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)

    ttic = time.time()
    
    if (Rh_cell is not None): \
    h0_tend = cpy_x_vec(  cnfg, Rh_cell, h0_tend)

    h0_tend = rhs_all_h(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, h0_tend)

    hk_tend = cpy_x_vec(  cnfg, h0_tend, hk_tend)

    h1_cell = inc_x_rhs(
        cnfg, hh_cell, k1_step, hk_tend, h1_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    u0_tend = cpy_x_vec(  cnfg, Ru_edge, u0_tend)

    u0_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)
    u0_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)

    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h1_cell,
                       1.0 - 1.0 * BETA, hh_cell)

    uk_tend = cpy_x_vec(  cnfg, u0_tend, uk_tend)
    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k1_step, uk_tend, uk_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- euler scheme implicit solve
        uk_edge = inv_x_1st(
            cnfg, uk_edge, 1. * k1_step, ck_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    cnfg.rhs_stage = 2
    cnfg.time_step = k2_step
    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[1] * isFB
    
#-- skipping the new tide eval. is still 2nd-order accurate
#   rhs_tde_d(  # eval. tides state 
#       mesh, mats, flow, cnfg, h1_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h1_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)

    ttic = time.time()

    if (Rh_cell is not None): \
    hk_tend = cpy_x_vec(  cnfg, Rh_cell, hk_tend)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, hk_tend)

    h2_cell = inc_x_rhs(
        cnfg, hh_cell, k2_step, hk_tend, h2_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    uk_tend = cpy_x_vec(  cnfg, Ru_edge, uk_tend)

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)    
    
    if (BETA > +0.0): \
    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h2_cell,
                       1.0 - 1.0 * BETA, hh_cell)
    else: \
    hb_cell = cpy_x_vec(  cnfg, h1_cell, hb_cell)
    
    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k2_step, uk_tend, uk_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- store for final theta solve
        cd_edge = \
            cpy_x_vec(  cnfg, ck_edge, cd_edge)

    #-- theta scheme implicit solve
        uk_edge = inv_x_2nd(
            cnfg, uk_edge, .5 * k2_step, ck_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    cnfg.rhs_stage = 3
    cnfg.time_step = k3_step
    cnfg = mark_time(
        cnfg, flow, start_t + 2. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[2] * isFB
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)

    ttic = time.time()

    if (Rh_cell is not None): \
    hk_tend = cpy_x_vec(  cnfg, Rh_cell, hk_tend)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, hk_tend)

   #hh_tend = +1./4. * h0_tend + 3./4. * hk_tend
    hk_tend = sum_2_way(
        cnfg, hk_tend, 1. / 4., h0_tend, 3. / 4., hk_tend)

    h3_cell = inc_x_rhs(
        cnfg, hh_cell, k3_step, hk_tend, h3_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    uk_tend = cpy_x_vec(  cnfg, Ru_edge, uk_tend)

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)

   #uu_tend = +1./4. * u0_tend + 3./4. * uk_tend
    uk_tend = sum_2_way(
        cnfg, uk_tend, 1. / 4., u0_tend, 3. / 4., uk_tend)

   #hm_cell = (1.-2.*BETA)* h2_cell + 
   #           2./3.*BETA * hh_cell + 4./3.*BETA* h3_cell
   #hm_cell = +1./4. * hh_cell + 3./4. * hm_cell
    hb_cell = sum_3_way(
        cnfg, hb_cell, 0. + 1.0 * BETA , h3_cell,
        +3.0 / 4.0 * (1.0 - 2.0 * BETA), h2_cell,
        +1.0 / 2.0 * (1.0 / 2.0 + BETA), hh_cell)

    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k3_step, uk_tend, uk_edge)
 
    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
    #-- KE evaluated at t+2/3 above
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- interp. drag to t+1/2
        cd_edge = sum_2_way(
            cnfg, cd_edge, 1./2., cd_edge, 1./2., ck_edge)

    #-- theta scheme implicit solve
        uk_edge = inv_x_2nd(
            cnfg, uk_edge, .5 * k3_step, cd_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- adapt next time-step & finalise

    ttic = time.time()

    cnfg.time_step = dt_step
    cnfg.next_step = dt_step

    if (cnfg.cfl_limit > 0.E+0): \
    cnfg.next_step = cfl_adapt(
        mesh, mats, cnfg, 
        gravity=gravity, hh_thin=cnfg.wetdry_h0 * 100.,
        rk_scal=15./8.0, rk_adv_=4.0/3.0,
        cfl_num=cnfg.cfl_limit, dt_prev=cnfg.time_step,
        dt_jump=cnfg.dt_margin,
        hh_prev=hh_cell, hh_cell=h3_cell,
        uu_prev=uu_edge, uu_edge=uk_edge)

    if (cnfg.verbosity >= 1 and 
        cnfg.next_step != 
        cnfg.time_step ): 
        print (f"*NEXT: {cnfg.next_step:+.2E}")

    h0_tend = set_x_vec(cnfg, h0_tend, 0.0E+00)
    u0_tend = set_x_vec(cnfg, u0_tend, 0.0E+00)

    if (cnfg.next_step > 0.0): \
    hh_cell = cpy_x_vec(cnfg, h3_cell, hh_cell)

    if (cnfg.next_step > 0.0): \
    uu_edge = cpy_x_vec(cnfg, uk_edge, uu_edge)

    ttoc = time.time()
    tcpu.finalise_+= (ttoc - ttic)
    
    return  hh_cell, uu_edge, hb_cell


def step_RK43(mesh, mats, flow, cnfg, 
              hh_cell, uu_edge,     # state
              Rh_cell, Ru_edge):    # slow tend.

#-- A 4-stage 4th/3rd-order RK scheme:
#-- D. Engwirda (2025): 3- and 4-stage forward-backward
#-- Runge-Kutta methods for geophysical flows

#-- drag included via a 2nd-order IMEX scheme

    start_t = cnfg.timeisnow

    k0_step = 1.0 / 4.0 * cnfg.time_step
    k1_step = 1.0 / 3.0 * cnfg.time_step
    k2_step = 2.0 / 3.0 * cnfg.time_step
    k3_step = 1.0 / 1.0 * cnfg.time_step
    dt_step = 1.0 / 1.0 * cnfg.time_step

    isFB = 1.0 * ("FB" in cnfg.integrate)

    gravity = flow.gravity

    dz_drag = variables.dz_drag
    c1_edge = variables.c1_edge
    c2_edge = variables.c2_edge
    z0_edge = variables.z0_edge
    n0_edge = variables.n0_edge

    cd_edge = variables.cd_save

    hb_cell = variables.hb_cell
    h0_cell = variables.h1_cell  # NB: aliased buffer!
    h1_cell = variables.h1_cell  # kth storage in h
    h2_cell = variables.h2_cell
    h3_cell = variables.h3_cell

    uk_edge = variables.uk_edge  # low storage in u
    
    hk_tend = variables.hh_tend  # zero by construction
    uk_tend = variables.uu_tend
    h0_tend = variables.h0_tend
    u0_tend = variables.u0_tend

#-- 1st RK + FB stage

    ttic = time.time()

    uk_edge = cpy_x_vec(  cnfg, uu_edge, uk_edge)

    cnfg.rhs_stage = 1
    cnfg.time_step = k0_step
    cnfg = mark_time(
        cnfg, flow, start_t + 0. / 1.0 * dt_step)

    BETA = cnfg.fb_weight[0] * isFB
    
    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, hh_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)
    
    ttic = time.time()
    
    if (Rh_cell is not None): \
    h0_tend = cpy_x_vec(  cnfg, Rh_cell, h0_tend)

    h0_tend = rhs_all_h(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, h0_tend)

    hk_tend = cpy_x_vec(  cnfg, h0_tend, hk_tend)

    h0_cell = inc_x_rhs(
        cnfg, hh_cell, k0_step, hk_tend, h0_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    u0_tend = cpy_x_vec(  cnfg, Ru_edge, u0_tend)

    u0_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)
    u0_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, hh_cell, uk_edge, u0_tend)

    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h1_cell,
                       1.0 - 1.0 * BETA, hh_cell)

    uk_tend = cpy_x_vec(  cnfg, u0_tend, uk_tend)
    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k0_step, uk_tend, uk_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- euler scheme implicit solve
        uk_edge = inv_x_1st(
            cnfg, uk_edge, 1. * k0_step, ck_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    cnfg.rhs_stage = 2
    cnfg.time_step = k1_step
    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 4.0 * dt_step)

    BETA = cnfg.fb_weight[1] * isFB

#-- skipping the new tide eval. is still 2nd-order accurate
#   rhs_tde_d(  # eval. tides state 
#       mesh, mats, flow, cnfg, h0_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h0_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)
    
    ttic = time.time()
    
    if (Rh_cell is not None): \
    hk_tend = cpy_x_vec(  cnfg, Rh_cell, hk_tend)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h0_cell, uk_edge, hk_tend)

    h1_cell = inc_x_rhs(
        cnfg, hh_cell, k1_step, hk_tend, h1_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    uk_tend = cpy_x_vec(  cnfg, Ru_edge, uk_tend)

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h0_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h0_cell, uk_edge, uk_tend)

    if (BETA > +0.0): \
    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h1_cell,
                       1.0 - 1.0 * BETA, hh_cell)
    else: \
    hb_cell = cpy_x_vec(  cnfg, h0_cell, hb_cell)

    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k1_step, uk_tend, uk_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- euler scheme implicit solve
        uk_edge = inv_x_1st(
            cnfg, uk_edge, 1. * k1_step, ck_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    cnfg.rhs_stage = 3
    cnfg.time_step = k2_step
    cnfg = mark_time(
        cnfg, flow, start_t + 1. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[2] * isFB
   
#-- skipping the new tide eval. is still 2nd-order accurate
#   rhs_tde_d(  # eval. tides state 
#       mesh, mats, flow, cnfg, h1_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h1_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)

    ttic = time.time()

    if (Rh_cell is not None): \
    hk_tend = cpy_x_vec(  cnfg, Rh_cell, hk_tend)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, hk_tend)

    h2_cell = inc_x_rhs(
        cnfg, hh_cell, k2_step, hk_tend, h2_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    uk_tend = cpy_x_vec(  cnfg, Ru_edge, uk_tend)

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h1_cell, uk_edge, uk_tend)

    if (BETA > +0.0): \
    hb_cell = sum_2_way(
        cnfg, hb_cell, 0.0 + 1.0 * BETA, h2_cell,
                       1.0 - 1.0 * BETA, hh_cell)
    else: \
    hb_cell = cpy_x_vec(  cnfg, h1_cell, hb_cell)
    
    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k2_step, uk_tend, uk_edge)

    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- store for final theta solve
        cd_edge = \
            cpy_x_vec(  cnfg, ck_edge, cd_edge)

    #-- theta scheme implicit solve
        uk_edge = inv_x_2nd(
            cnfg, uk_edge, .5 * k2_step, ck_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

#-- 4th RK + FB stage

    ttic = time.time()

    cnfg.rhs_stage = 4
    cnfg.time_step = k3_step
    cnfg = mark_time(
        cnfg, flow, start_t + 2. / 3.0 * dt_step)

    BETA = cnfg.fb_weight[3] * isFB

    rhs_tde_d(  # eval. tides state 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)
    rhs_all_d(  # eval. diagnostics 
        mesh, mats, flow, cnfg, h2_cell, uk_edge)

    ttoc = time.time()
    tcpu.evaluate_+= (ttoc - ttic)
    
    ttic = time.time()

    if (Rh_cell is not None): \
    hk_tend = cpy_x_vec(  cnfg, Rh_cell, hk_tend)

    hk_tend = rhs_all_h(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, hk_tend)

   #hh_tend = +1./4. * h0_tend + 3./4. * hk_tend
    hk_tend = sum_2_way(
        cnfg, hk_tend, 1. / 4., h0_tend, 3. / 4., hk_tend)

    h3_cell = inc_x_rhs(
        cnfg, hh_cell, k3_step, hk_tend, h3_cell)

    ttoc = time.time()
    tcpu.thickness+= (ttoc - ttic)

    ttic = time.time()

    if (Ru_edge is not None): \
    uk_tend = cpy_x_vec(  cnfg, Ru_edge, uk_tend)

    uk_tend = rhs_slw_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(
        mesh, mats, flow, cnfg, h2_cell, uk_edge, uk_tend)

   #uu_tend = +1./4. * u0_tend + 3./4. * uk_tend
    uk_tend = sum_2_way(
        cnfg, uk_tend, 1. / 4., u0_tend, 3. / 4., uk_tend)

   #hm_cell = (1.-2.*BETA)* h2_cell + 
   #           2./3.*BETA * hh_cell + 4./3.*BETA* h3_cell
   #hm_cell = +1./4. * hh_cell + 3./4. * hm_cell
    hb_cell = sum_3_way(
        cnfg, hb_cell, 0. + 1.0 * BETA , h3_cell,
        +3.0 / 4.0 * (1.0 - 2.0 * BETA), h2_cell,
        +1.0 / 2.0 * (1.0 / 2.0 + BETA), hh_cell)

    uk_tend = rhs_pgf_u(
        mesh, mats, flow, cnfg, hb_cell, uk_edge, uk_tend)

    uk_edge = inc_x_rhs(
        cnfg, uu_edge, k3_step, uk_tend, uk_edge)
 
    if (cnfg.calc_drag and cnfg.anylaw_cd > 0.):
    #-- KE evaluated at t+2/3 above
        ck_edge = calc_drag(
            mesh, mats, cnfg, gravity, dz_drag,
            c1_edge, c2_edge, z0_edge, n0_edge)

    #-- interp. drag to t+1/2
        cd_edge = sum_2_way(
            cnfg, cd_edge, 1./2., cd_edge, 1./2., ck_edge)

    #-- theta scheme implicit solve
        uk_edge = inv_x_2nd(
            cnfg, uk_edge, .5 * k3_step, cd_edge, uu_edge)

    ttoc = time.time()
    tcpu.momentum_+= (ttoc - ttic)

    ttic = time.time()

    cnfg.time_step = dt_step
    cnfg.next_step = dt_step

    if (cnfg.cfl_limit > 0.E+0): \
    cnfg.next_step = cfl_adapt(
        mesh, mats, cnfg, 
        gravity=gravity, hh_thin=cnfg.wetdry_h0 * 100.,
        rk_scal=3.0/2.0, rk_adv_=4.0/3.0,
        cfl_num=cnfg.cfl_limit, dt_prev=cnfg.time_step,
        dt_jump=cnfg.dt_margin,
        hh_prev=hh_cell, hh_cell=h3_cell,
        uu_prev=uu_edge, uu_edge=uk_edge)

    if (cnfg.verbosity >= 1 and 
        cnfg.next_step != 
        cnfg.time_step ): 
        print (f"*NEXT: {cnfg.next_step:+.2E}")

    h0_tend = set_x_vec(cnfg, h0_tend, 0.0E+00)
    u0_tend = set_x_vec(cnfg, u0_tend, 0.0E+00)

    hh_cell = cpy_x_vec(cnfg, h3_cell, hh_cell)
    uu_edge = cpy_x_vec(cnfg, uk_edge, uu_edge)

    ttoc = time.time()
    tcpu.finalise_+= (ttoc - ttic)
    
    return  hh_cell, uu_edge, hb_cell


try:
    # load cython kernels, if compiled
    from _kt import _cfl_adapt as cfl_adapt
    from _kt import _bnd_x_vec as bnd_x_vec
    from _kt import _nrm_x_vec as nrm_x_vec
    from _kt import _nrm_z_vec as nrm_z_vec
    from _kt import _set_x_vec as set_x_vec
    from _kt import _cpy_x_vec as cpy_x_vec
    from _kt import _inc_x_rhs as inc_x_rhs
    from _kt import _inv_x_1st as inv_x_1st
    from _kt import _inv_x_2nd as inv_x_2nd
    from _kt import _sum_2_way as sum_2_way
    from _kt import _sum_3_way as sum_3_way
    from _kt import _sym_3_way as sym_3_way
    
except ImportError:
    raise RuntimeError("Cython back-end not found")

