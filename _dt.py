
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

""" SWE time integration via various Runge-Kutta methods
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda, Jeremy Lilly
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import reals_t

from rhs import rhs_slw_u, rhs_fst_u, rhs_pgf_u, \
                rhs_all_h

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
        raise Exception("Must set fixed DT")

    return cnfg
    

def init_RKFB(cnfg):

#-- Initialise coefficients for user time-stepping schemes

    if   ("RK33" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.301666666666667, 0.316666666666667,
            0.366666666666667
            ] )

    """
    else:#"RK43" in cnfg.integrate):
        cnfg.fb_weight = np.array([
            0.000000000000000, 0.500000000000000,
            0.500000000000000, 0.000000000000000
            ] )
    """

    return cnfg


@jax.jit
def step_RK33(ops, gravity, zb_cell, fb_weight, dt,
              hh_cell, uu_edge):

#-- A 3-stage 3rd/2nd-order RK scheme:
#-- D. Engwirda (2025): 3- and 4-stage forward-backward
#-- Runge-Kutta methods for geophysical flows

#-- pure jax: OPS is an ops.JaxOps bundle of gather-operator
#-- arrays (fixed for the whole run), everything else is a jnp
#-- array or scalar. jit-compiled; called many times back-to-back
#-- by run_scan below.

    k1_step = 1.0 / 3.0 * dt
    k2_step = 2.0 / 3.0 * dt
    k3_step = 1.0 / 1.0 * dt

#-- 1st RK + FB stage

    h0_tend = jnp.zeros_like(hh_cell)
    u0_tend = jnp.zeros_like(uu_edge)

    uk_edge = uu_edge

    BETA = fb_weight[0]

    h0_tend = rhs_all_h(ops, hh_cell, uk_edge, h0_tend)

    h1_cell = hh_cell - k1_step * h0_tend

    u0_tend = rhs_slw_u(ops, hh_cell, uk_edge, u0_tend)
    u0_tend = rhs_fst_u(ops, hh_cell, uk_edge, u0_tend)

    hb_cell = (0.0 + 1.0 * BETA) * h1_cell + \
              (1.0 - 1.0 * BETA) * hh_cell

    uk_tend = u0_tend
    uk_tend = rhs_pgf_u(ops, hb_cell, zb_cell, gravity, uk_tend)

    uk_edge = uu_edge - k1_step * uk_tend


#-- 2nd RK + FB stage

    hk_tend = jnp.zeros_like(hh_cell)
    uk_tend = jnp.zeros_like(uu_edge)

    BETA = fb_weight[1]

    hk_tend = rhs_all_h(ops, h1_cell, uk_edge, hk_tend)

    h2_cell = hh_cell - k2_step * hk_tend

    uk_tend = rhs_slw_u(ops, h1_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(ops, h1_cell, uk_edge, uk_tend)

    hb_cell = (0.0 + 1.0 * BETA) * h2_cell + \
              (1.0 - 1.0 * BETA) * hh_cell

    uk_tend = rhs_pgf_u(ops, hb_cell, zb_cell, gravity, uk_tend)

    uk_edge = uu_edge - k2_step * uk_tend


#-- 3rd RK + FB stage

    hk_tend = jnp.zeros_like(hh_cell)
    uk_tend = jnp.zeros_like(uu_edge)

    BETA = fb_weight[2]

    hk_tend = rhs_all_h(ops, h2_cell, uk_edge, hk_tend)

    hk_tend = +1./4. * h0_tend + 3./4. * hk_tend

    h3_cell = hh_cell - k3_step * hk_tend

    uk_tend = rhs_slw_u(ops, h2_cell, uk_edge, uk_tend)
    uk_tend = rhs_fst_u(ops, h2_cell, uk_edge, uk_tend)

    uk_tend = +1./4. * u0_tend + 3./4. * uk_tend

    hb_cell = (0.0 + 1.0 * BETA) * h3_cell + \
        +3.0 / 4.0 * (1.0 - 2.0 * BETA) * h2_cell + \
        +1.0 / 2.0 * (1.0 / 2.0 + BETA) * hh_cell

    uk_tend = rhs_pgf_u(ops, hb_cell, zb_cell, gravity, uk_tend)

    uk_edge = uu_edge - k3_step * uk_tend

    return h3_cell, uk_edge


@partial(jax.jit, static_argnums=(6,))
def run_scan(ops, gravity, zb_cell, fb_weight, dt,
             state, nstep):

#-- advance NSTEP fixed-dt RK33-FB steps back-to-back on-device,
#-- via lax.scan -- this is the perf-critical replacement for the
#-- old per-step Python driver loop in slv.py.

    def body(carry, _):
        hh_cell, uu_edge = carry
        hh_cell, uu_edge = step_RK33(
            ops, gravity, zb_cell, fb_weight, dt,
            hh_cell, uu_edge)
        return (hh_cell, uu_edge), None

    state, _ = jax.lax.scan(body, state, None, length=nstep)

    return state


