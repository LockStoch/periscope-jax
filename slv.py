
import time
import copy
import numpy as np

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

from _dt import step_eqns, step_bnds, mark_time
from _dt import init_RKFB, init_step
from _dx import invariant, scale_mix

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

    cnfg.calc_tide = True
    cnfg.calc_slow = True
    cnfg.calc_fast = True
    cnfg.calc_drag = True
    
    cnfg.timeisnow = cnfg.timestart
    cnfg.stat_step = +0.0
    cnfg.stat_prev = +0
    cnfg.save_step = +0.0
    cnfg.save_prev = +0

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

    # set sparse spatial operators
    mats = operators(mesh)

    # set domain boundary stencils
    mesh = init_obcs(mesh, flow, mats)

    ttoc = time.time()
    print("*FORM done (sec):", round(ttoc - ttic, 2))
   
    print("")
    print("Integrating the flow...")

    kp_sum_ = []; en_sum_ = [];
    
    init_pool(cnfg, mesh)  # alloc. internal arrays

    hh_cell = _var.hh_cell
    hh_cell[:] = flow.hh_cell
    hh_min_ = _var.hh_min_
    hh_max_ = _var.hh_max_
    hh_min_[:] = flow.hh_cell; hh_max_[:] = flow.hh_cell
    
    uu_edge = _var.uu_edge
    uu_edge[:] = flow.uu_edge
    uu_filt = _var.uu_filt
    uu_filt[:] = flow.uu_filt
    uu_min_ = _var.uu_min_
    uu_max_ = _var.uu_max_
    uu_min_[:] = flow.uu_edge; uu_max_[:] = flow.uu_edge

    qq_cell = _var.qq_cell
    qq_min_ = _var.qq_min_
    qq_max_ = _var.qq_max_

    zt_rms_ = _var.zt_rms_
    ke_ave_ = _var.ke_ave_
    ke_rms_ = _var.ke_rms_
    ke_max_ = _var.ke_max_
    dk_ave_ = _var.dk_ave_
    dk_rms_ = _var.dk_rms_
    dk_max_ = _var.dk_max_

    uu_edge[mesh.edge.mask] = 0.  # ensure BC
    uu_edge[mesh.edge.open] =flow.uu_edge[mesh.edge.open]
   
    # start forward integrations
    flow, cnfg = pre (mesh, mats, flow, cnfg)

    ttic = time.time(); next = +0; freq = +0

    flow.prev = flow.next  # if forc. time-invariant...

    cnfg = init_step (mesh, mats, flow, cnfg, 
                      hh_cell, uu_edge, 
                      qq_cell)

    cnfg.time_stop+= cnfg.timeisnow * (cnfg.time_stop>0.)
    cnfg.stat_next = cnfg.timeisnow
    cnfg.save_next = cnfg.timeisnow

    for step in range(+0, cnfg.iteration + 1):

        tnow = cnfg.timeisnow
        
        #-- syncronise time-step to hit output epoch
        sync = np.inf
        if (step > 0 and cnfg.stat_freq > 0 and ( 
               (isinstance(cnfg.stat_freq, flt64_t))
                ) ):
            sync = min(sync, cnfg.stat_next)

        if (step > 0 and cnfg.save_freq > 0 and ( 
               (isinstance(cnfg.save_freq, flt64_t))
                ) ):
            sync = min(sync, cnfg.save_next)

        if (step > 0 and cnfg.time_stop > 0 and ( 
               (isinstance(cnfg.time_stop, flt64_t))
                ) ):
            sync = min(sync, cnfg.time_stop)
        
        if (cnfg.timeisnow + 1./1. * cnfg.time_step > sync):
            cnfg.time_step = (sync - tnow) / 1.
           #print (f">SYNC: {+cnfg.time_step:+.2E}")

        elif (
            cnfg.timeisnow + 3./2. * cnfg.time_step > sync):
            cnfg.time_step = (sync - tnow) / 2.
           #print (f">SYNC: {+cnfg.time_step:+.2E}")
        
        if (step > 0):
        #-- 0-th step is just to write ICs to output...
            if (flow.xx_time is not None):
                # find needed forcing step to interp.
                need = np.searchsorted(
                    flow.xx_time, 
                        cnfg.timeisnow + cnfg.time_step)
                      
                if (need>= flow.xx_time.size):
                    print(f">FORC:", 
                          f"{need} >= {flow.xx_time.size}")
                    need = flow.xx_time.size - 1 

                if (need > flow.next.step):
                # a piecewise linear interp. for now...
                    flow.prev = copy.deepcopy(flow.next)
                    flow = load_forc(forc, flow, 
                                     step= need)
                    flow = sort_forc(flow, mesh)
                    
            hh_cell, uu_edge, qq_cell = step_eqns(
                mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                        qq_cell
            )

            hh_min_, hh_max_, \
            uu_min_, uu_max_, \
            qq_min_, qq_max_, \
                     zt_rms_, \
            ke_ave_, ke_rms_, ke_max_, \
            dk_ave_, dk_rms_, dk_max_ = step_bnds(
                mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                        qq_cell,
                                        hh_min_, hh_max_,
                                        uu_min_, uu_max_,
                                        qq_min_, qq_max_,
                                        zt_rms_,
                               ke_ave_, ke_rms_, ke_max_,
                               dk_ave_, dk_rms_, dk_max_)
                
            cnfg = mark_time(
            cnfg, flow, tnow+cnfg.time_step)

            tnow = cnfg.timeisnow

            cnfg.stat_step+= cnfg.time_step
            cnfg.save_step+= cnfg.time_step
    
            cnfg.time_step = cnfg.next_step

        if (step >= cnfg.iteration or (
                cnfg.time_stop > 0 and
                    cnfg.timeisnow-cnfg.time_stop >= 0)):
            cnfg.completed =  True            

        if ( out(cnfg.completed, 
            cnfg.stat_freq, cnfg.stat_next, step, tnow)):
        #-- eval. statistics at every stat steps
            kp_val_, en_val_ = invariant(
                mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                        qq_cell
            )
            kp_sum_.append (kp_val_)
            en_sum_.append (en_val_)

            cnfg.stat_step/= max(+1, step-cnfg.stat_prev)

            done = max(step/ cnfg.iteration, 
                       tnow/ cnfg.time_stop)

            print (
             f"*STEP: {step:>7} [{done * 100.:>5.1f}%] ", 
                f"dt: {cnfg.stat_step:+.2E} ",
            f"d(K+P): {rdf(kp_val_, kp_sum_[+0]):+.6E} ",
            f"d(Q^2): {rdf(en_val_, en_sum_[+0]):+.6E} ",
            )

            cnfg.stat_next+= cnfg.stat_freq
            cnfg.stat_prev = step
            cnfg.stat_step = 0; next = next + 1

        if ( out(cnfg.completed, 
            cnfg.save_freq, cnfg.save_next, step, tnow)):
        #-- & save all state at every save steps
            save_step(save, mesh, mats,
                      flow, cnfg, freq, hh_cell, uu_edge,
                                        qq_cell
            )

            cnfg.save_step/= max(+1, step-cnfg.save_prev)

            done = max(step/ cnfg.iteration, 
                       tnow/ cnfg.time_stop)

            cnfg.save_next+= cnfg.save_freq
            cnfg.save_prev = step
            cnfg.save_step = 0; freq = freq + 1

        if (cnfg.completed): break
        
    ttoc = time.time()

    save_last(save, mesh, mats, flow, cnfg, step, 
              kp_sum_, en_sum_,
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
    print("*evaluate_ (sec):", round(tcpu.evaluate_, 2))    
    print("*thickness (sec):", round(tcpu.thickness, 2))
    print("*momentum_ (sec):", round(tcpu.momentum_, 2))
    print("*finalise_ (sec):", round(tcpu.finalise_, 2))
    print("*calc-obcs (sec):", round(tcpu.calc_obcs, 2))
    print("*calc-udry (sec):", round(tcpu.calc_udry, 2))
    print("*upwinding (sec):", round(tcpu.upwinding, 2))
    print("*calc-hmap (sec):", round(tcpu.calc_hmap, 2))
    print("*tend-hadv (sec):", round(tcpu.tend_hadv, 2))
    print("*calc-perp (sec):", round(tcpu.calc_perp, 2))
    print("*calc-u-ke (sec):", round(tcpu.calc_u_ke, 2))
    print("*calc-u-pv (sec):", round(tcpu.calc_u_pv, 2))
    print("*tend-uadv (sec):", round(tcpu.tend_uadv, 2))
    print("*tend-upgf (sec):", round(tcpu.tend_upgf, 2))
    print("*calc-umix (sec):", round(tcpu.calc_umix, 2))
    print("*calc-uwav (sec):", round(tcpu.calc_uwav, 2))
    print("*tend-umix (sec):", round(tcpu.tend_umix, 2))
    print("*calc-hmix (sec):", round(tcpu.calc_hmix, 2))    
    print("*tend-hmix (sec):", round(tcpu.tend_hmix, 2))
    print("*calc-tide (sec):", round(tcpu.calc_tide, 2))
    print("*calc-sal_ (sec):", round(tcpu.calc_self, 2))
    print("*tend-ugeo (sec):", round(tcpu.tend_ugeo, 2))
    print("*tend-utau (sec):", round(tcpu.tend_utau, 2))
    print("*calc-drag (sec):", round(tcpu.calc_drag, 2))


def out(done, freq, mark, step, time):
#-- return TRUE if an output step has been reached
    if (freq > 0 and (isinstance(freq, index_t) 
        and step % freq == 0)): return True

    if (freq > 0 and (isinstance(freq, flt64_t) 
        and time - mark >= 0)): return True

    return  done


def rdf(xval, yval):
#-- return relative change -- floor'd to zero near eps
    eps_ = np.finfo(reals_t).eps
    rdel = (xval - yval) / (yval + eps_)
    return  rdel * (abs (rdel) >= +1 * eps_)


def pre(mesh, mats, flow, cnfg):
#-- do various init. ops for flow + config. at pre-run
   
    # determine "optimal" chunking
    cnfg.chunkcell = (
        mesh.cell.size // cnfg.numthread // 
                          cnfg.numchunks
        )
    cnfg.chunkcell =(cnfg.chunkcell // 8) * 8
    cnfg.chunkcell = max(1, cnfg.chunkcell)

    cnfg.chunkedge = (
        mesh.edge.size // cnfg.numthread // 
                          cnfg.numchunks
        )
    cnfg.chunkedge =(cnfg.chunkedge // 8) * 8
    cnfg.chunkedge = max(1, cnfg.chunkedge)

    cnfg.chunkvert = (
        mesh.vert.size // cnfg.numthread // 
                          cnfg.numchunks
        )
    cnfg.chunkvert =(cnfg.chunkvert // 8) * 8
    cnfg.chunkvert = max(1, cnfg.chunkvert)

    cnfg.chunksize = \
        min(cnfg.chunkcell, cnfg.chunkedge)
    cnfg.chunksize = \
        min(cnfg.chunksize, cnfg.chunkvert)

    # remap coriolis onto msh DoFs
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

    flow.h0_rms_ = \
        np.sqrt(np.mean(_var.hh_cell ** 2))
    flow.u0_rms_ = \
        np.sqrt(np.mean(_var.uu_edge ** 2))
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

    # const. scaling on drag param.
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

    # mesh scaling for dissipation
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

    s2_edge, s4_edge, msh_fix = \
        scale_mix(mesh, mats, cnfg)

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

