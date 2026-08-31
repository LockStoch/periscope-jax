
import time
import math
import numpy as np
import jax.numpy as jnp

""" SLV: solve the nonlinear SWE on generalised MPAS meshes.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t

from log import tcpu

from msh import load_mesh, sort_mesh, \
                load_flow, sort_flow, \
                load_forc, sort_forc, \
                init_wall, init_obcs
from ops import operators

from mem import init_pool
from mem import variables as _var

from io_ import init_file, save_step, save_last

from _dt import init_RKFB, init_step, run_scan
from _dx import invariant

def swe(cnfg):

    print(
    "#"+"=========================================="*2+"\n" +
    "#              o                     \n" +
    "#   ,_   _  _  `  .   _, __  ,_   _  \n" +
    "# _/|_)_(/_/ (_(_/_)_(__(_)_/|_)_(/_ \n" +
    "#  /|                       /|       \n" +
    "# (/                       (/        \n" +
    "#"+"=========================================="*2+"\n"
         )

    cnfg.timeisnow = cnfg.timestart

    cnfg.completed = False

    if not cnfg.fb_weight: init_RKFB (cnfg)

    # mesh, forcing & solution i/o
    name = cnfg.mesh_file
    forc = cnfg.forc_file
    save = cnfg.soln_file

    print("Loading input assets...")

    ttic = time.time()

    # load mesh + init. conditions
    mesh = load_mesh(name)
    flow = load_flow(name, mesh, lean=True)
    flow = load_forc(forc, flow, lean=True)

    # offset, if ICs are a restart
    cnfg.timestart+= flow.elapsed
    cnfg.timeisnow+= flow.elapsed

    ttoc = time.time()
    print("*READ done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Creating output file...")

    ttic = time.time()

    init_file(name, cnfg, save, mesh, flow)

    ttoc = time.time()
    print("*SAVE done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Reordering mesh data...")

    ttic = time.time()

    mesh = sort_mesh(mesh, True)
    flow = sort_flow(flow, mesh, lean=True)
    flow = sort_forc(flow, mesh, lean=True)

    flow.hh_cell = \
        np.maximum(cnfg.wetdry_h0 / 2., flow.hh_cell)

    ttoc = time.time()
    print("*SORT done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Forming coefficients...")

    ttic = time.time()

    # set basic wall masks + lists
    mesh = init_wall(mesh, flow)

    # set sparse spatial operators -- also builds mats.jx, the
    # jax gather-form of the operators used by the RK stepper
    mats = operators(mesh)

    # set domain boundary stencils
    mesh = init_obcs(mesh, flow, mats)

    ttoc = time.time()
    print("*FORM done (sec):", round(ttoc - ttic, 2))

    print("")
    print("Integrating the flow...")

    # alloc. host-side diagnostic pool -- kept only for the
    # (currently zero/unused-by-this-physics) fields the NetCDF
    # writer in io_.py may be asked to save via --save-vars
    init_pool(cnfg, mesh)

    qq_cell = _var.qq_cell   # inert for this physics; carried
                              # through only for save_step/save_last
                              # API compatibility

    hh_min_, hh_max_ = _var.hh_min_, _var.hh_max_
    uu_min_, uu_max_ = _var.uu_min_, _var.uu_max_
    qq_min_, qq_max_ = _var.qq_min_, _var.qq_max_
    zt_rms_ = _var.zt_rms_
    ke_ave_, ke_rms_, ke_max_ = \
        _var.ke_ave_, _var.ke_rms_, _var.ke_max_
    dk_ave_, dk_rms_, dk_max_ = \
        _var.dk_ave_, _var.dk_rms_, _var.dk_max_

    # start forward integrations
    flow, cnfg = pre (mesh, mats, flow, cnfg)

    # initial state -- apply wall/open BCs host-side, then move
    # to device once
    hh_cell = flow.hh_cell.copy()
    uu_edge = flow.uu_edge.copy()

    uu_edge[mesh.edge.mask] = 0.  # ensure BC
    uu_edge[mesh.edge.open] =flow.uu_edge[mesh.edge.open]

    cnfg = init_step (mesh, mats, flow, cnfg,
                      hh_cell, uu_edge,
                      qq_cell)

    state = (
        jnp.asarray(hh_cell, dtype=reals_t),
        jnp.asarray(uu_edge, dtype=reals_t),
    )

    dt = float(cnfg.time_step)
    gravity = float(flow.gravity)
    zb_cell = jnp.asarray(flow.zb_cell, dtype=reals_t)
    fb_weight = jnp.asarray(cnfg.fb_weight, dtype=reals_t)

    nsteps = int(cnfg.iteration)
    save_freq = int(cnfg.save_freq)
    stat_freq = int(cnfg.stat_freq)

    # advance in chunks of GCD(save_freq, stat_freq) steps at a
    # time via a single fused lax.scan per chunk, so both output
    # cadences are always hit exactly on a chunk boundary. Freqs
    # left unset default to a huge sentinel (np.iinfo(index_t).max,
    # a Mersenne prime) -- GCD-ing against that directly would
    # collapse the chunk size to 1, so exclude "unset" freqs first
    never = np.iinfo(index_t).max
    freqs = [f for f in (save_freq, stat_freq) if f < never]

    if   (len(freqs) == 2): chunk = math.gcd(*freqs)
    elif (len(freqs) == 1): chunk = freqs[0]
    else:                   chunk = max(1, nsteps)

    chunk = max(1, min(chunk, max(1, nsteps)))

    kp_sum_ = []; hr_sum_ = []

    def do_stat(step):
        hh_now = np.asarray(state[0])
        uu_now = np.asarray(state[1])

        kp_val_, hr_val_ = invariant(mesh, hh_now, uu_now)
        kp_sum_.append(kp_val_)
        hr_sum_.append(hr_val_)

        done = step / max(1, nsteps)
        print (
         f"*STEP: {step:>7} [{done * 100.:>5.1f}%] ",
        f"d(Vol): {rdf(kp_val_, kp_sum_[+0]):+.6E} ",
        f"d(Hrms): {rdf(hr_val_, hr_sum_[+0]):+.6E} ",
        )

    def do_save(step, freq):
        hh_now = np.asarray(state[0])
        uu_now = np.asarray(state[1])

        save_step(save, mesh, mats,
                  flow, cnfg, freq, hh_now, uu_now,
                                     qq_cell
        )

    ttic = time.time(); step = 0; freq = 0

    # step-0: write ICs
    do_stat(step)
    do_save(step, freq); freq+= 1

    while (step < nsteps):

        take = min(chunk, nsteps - step)

        state = run_scan(
            mats.jx, gravity, zb_cell, fb_weight, dt,
                                        state, take)

        step+= take
        cnfg.timeisnow = cnfg.timestart + step * dt

        if (step % stat_freq == 0 or step >= nsteps):
            do_stat(step)

        if (step % save_freq == 0 or step >= nsteps):
            do_save(step, freq); freq+= 1

    cnfg.completed = True

    ttoc = time.time()

    hh_cell = np.asarray(state[0])
    uu_edge = np.asarray(state[1])

    save_last(save, mesh, mats, flow, cnfg, step,
              kp_sum_, hr_sum_,
              hh_min_, hh_max_,
              uu_min_, uu_max_,
              qq_min_, qq_max_,
                       zt_rms_,
              ke_ave_, ke_rms_, ke_max_,
              dk_ave_, dk_rms_, dk_max_)

    print("")
    print("Run complete; runtime:")
    print("*wall-time (sec):", round(ttoc - ttic, 2))
    print("*file-i/o. (sec):", round(tcpu.filewrite, 2))


def rdf(xval, yval):
#-- return relative change -- floor'd to zero near eps
    eps_ = np.finfo(reals_t).eps
    rdel = (xval - yval) / (yval + eps_)
    return  rdel * (abs (rdel) >= +1 * eps_)


def pre(mesh, mats, flow, cnfg):
#-- do various init. ops for flow + config. at pre-run

    # remap coriolis onto msh DoFs -- unused by the current
    # reduced (PGF + continuity) physics, kept as groundwork for
    # re-introducing rotation later
    flow.ff_edge = mats.edge_tail_sums*flow.ff_vert
    flow.ff_edge/= mesh.edge.area

    flow.ff_cell = mats.cell_kite_sums*flow.ff_vert
    flow.ff_cell/= mesh.cell.area

    _var.ff_vert[:] = np.asarray(
           flow.ff_vert, dtype=flt32_t)
    _var.ff_edge[:] = np.asarray(
           flow.ff_edge, dtype=flt32_t)
    _var.ff_cell[:] = np.asarray(
           flow.ff_cell, dtype=flt32_t)

    _var.ff_cell*= (not cnfg.no_rotate)
    _var.ff_edge*= (not cnfg.no_rotate)
    _var.ff_vert*= (not cnfg.no_rotate)

    cnfg.ff_max_ = np.max(np.abs(_var.ff_edge))

    # NB: read off FLOW directly (not the _var.* pool -- the jax
    # hot path doesn't stage the working state through it)
    flow.h0_rms_ = \
        np.sqrt(np.mean(flow.hh_cell ** 2))
    flow.u0_rms_ = \
        np.sqrt(np.mean(flow.uu_edge ** 2))
    flow.p0_rms_ = \
        np.sqrt(np.mean(_var.ff_cell ** 2))

    flow.c0_rms_ = flow.u0_rms_ + \
        np.sqrt (flow.gravity * flow.h0_rms_)

    cnfg.hh_tiny = 100. * \
        np.finfo(hdata_t).eps * flow.h0_rms_
    cnfg.uu_tiny = 1. * \
        np.finfo(flt64_t).eps * flow.c0_rms_
    cnfg.pv_tiny = 1. * \
        np.finfo(reals_t).eps * flow.p0_rms_
    cnfg.pv_tiny+= cnfg.uu_tiny

    cnfg.ke_tiny = np.sqrt(cnfg.uu_tiny)

    # const. scaling on drag param. -- diagnostic-only, unused
    # by the current reduced physics
    cnfg.anylaw_cd = \
        max([cnfg.linlaw_cd, cnfg.sqrlaw_cd,
             cnfg.loglaw_z0, cnfg.manlaw_n0
           ] )

    _var.c1_edge[:] = flow.c1_edge * cnfg.linlaw_cd
    _var.c2_edge[:] = flow.c2_edge * cnfg.sqrlaw_cd
    _var.z0_edge[:] = flow.z0_edge * cnfg.loglaw_z0
    _var.n0_edge[:] = flow.n0_edge * cnfg.manlaw_n0

    # subgrid drag thickness scale
    flow.dz_drag = np.asarray (
        mats.edge_wing_sums * (
        np.maximum(0.0, flow.zb_drag - flow.zb_cell
        ) ), dtype=flt32_t)
    flow.dz_drag /= mesh.edge.area

    _var.zb_cell[:] = flow.zb_cell
    _var.dz_drag[:] = flow.dz_drag

    # mesh scaling for dissipation -- diagnostic-only, unused
    # by the current reduced physics
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_2, cnfg.uu_visc_4)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.leith_chi)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.waves_chi)
    cnfg.uu_visc_k = \
        max (cnfg.uu_visc_k, cnfg.wetdry_h0)

    cnfg.hh_diff_k = \
        max (cnfg.hh_diff_2, cnfg.hh_diff_4)
    cnfg.hh_diff_k = \
        max (cnfg.hh_diff_k, cnfg.shock_chi)

    s2_edge = 1.0
    s4_edge = 1.0
    msh_fix = 1.0

    _var.msh_fix[:] = msh_fix
    _var.msh_nu2[:] = s2_edge
    _var.msh_nu4[:] = s4_edge

    _var.visc_u2[:] = np.asarray(
        (cnfg.uu_visc_2 * s2_edge), dtype=reals_t)
    _var.visc_u4[:] = np.asarray(
        (cnfg.uu_visc_4 * s4_edge), dtype=reals_t)

    _var.diff_h2[:] = np.asarray(
        (cnfg.hh_diff_2 * s2_edge), dtype=reals_t)
    _var.diff_h4[:] = np.asarray(
        (cnfg.hh_diff_4 * s4_edge), dtype=reals_t)

    _var.diff_h4[:] = np.sqrt(_var.diff_h4)

    return flow, cnfg
