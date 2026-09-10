
import time
import math
import numpy as np
import jax.numpy as jnp

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

from ops import gather_apply

def calc_vars(mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                      qq_cell):

#-- compute diagnostic variables from the current state

    ff_dual = variables.ff_vert
    ff_edge = variables.ff_edge
    ff_cell = variables.ff_cell
    
    Xi_tide = variables.Xi_tide  # lagged values
    Xi_self = variables.Xi_self

    uu_filt = variables.uu_filt

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)

    hh_dual, hh_edge, hh_quad, hh_bias = calc_hmap(
        mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge)

    """
    ke_cell, ke_bias = calc_u_ke(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        +1. / 2. * cnfg.time_step)

    rv_dual, pv_dual, r2_dual, p2_dual, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)
    """    

    ke_cell = variables.ke_cell
    ke_bias = variables.ke_bias

    rv_dual = variables.rv_dual
    pv_dual = variables.pv_dual
    rv_cell = variables.rv_cell
    pv_cell = variables.pv_cell
    pv_edge = variables.pv_edge
    pv_bias = variables.pv_bias

    nu_turb = variables.nu_turb  # lagged values

    nu_thin = variables.nu_thin

    nu_wave = variables.nu_wave
    os_wave = variables.os_wave

    nu_shoc = variables.nu_shoc
    os_shoc = variables.os_shoc

    return hh_edge, hh_dual, hh_bias, \
           ke_cell, ke_bias, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, \
           pv_edge, pv_bias, \
           vv_edge, nu_turb, \
           nu_wave, os_wave, \
           nu_shoc, os_shoc, \
           nu_thin, uu_filt, \
           Xi_tide, Xi_self


def invariant(mesh, hh_cell, uu_edge):

#-- compute basic scalar diagnostics for the reduced (PGF +
#-- continuity) physics: total volume (should be conserved by
#-- the flux-form continuity scheme up to boundary effects) and
#-- domain rms thickness (a bounded-ness sanity check).

    kp_sums = np.sum(hh_cell * mesh.cell.area)
    hr_sums = np.sqrt(np.mean(hh_cell ** 2))

    return kp_sums, hr_sums


def calc_hmap(mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge):

#-- compute discrete thickness

    ttic = time.time()
    
    hh_dual = variables.hh_dual
    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad
    hh_bias = variables.hh_bias

    hh_dual[:] = mats.dual_kite_sums * hh_cell
    hh_dual[:]/= mesh.vert.area

    hh_edge[:] = mats.edge_wing_sums * hh_cell
    hh_edge[:]/= mesh.edge.area

    # don't worry about hh_quad or hh_bias for now

    ttoc = time.time()
    tcpu.calc_hmap = tcpu.calc_hmap + (ttoc - ttic)

    return hh_dual, hh_edge, hh_quad, hh_bias
              
              
def calc_perp(mesh, mats, cnfg, uu_edge):

#-- get tangential velocity

    ttic = time.time()

    vv_edge = variables.vv_edge

    vv_edge[:] = mats.edge_lsqr_perp * uu_edge

    ttoc = time.time()
    tcpu.calc_perp = tcpu.calc_perp + (ttoc - ttic)

    return vv_edge
              
              
def calc_hh_edge(ops, hh_cell):

#-- cell-to-edge thickness remap -- JAX, GPU-resident. Reused by both
#-- tend_hadv (continuity) and the Stage 1 advection terms below.
#-- This is the CENTRE-scheme formula (main's calc_hmap, hh_scheme ==
#-- "CENTRE" branch); the reduced physics here doesn't implement the
#-- UPWIND thickness-blend branch main defaults to, matching the
#-- simplification tend_hadv already made for continuity.

    return gather_apply(ops.wing, hh_cell) / ops.edge_area


def tend_hadv(ops, hh_cell, uu_edge, hh_tend):

#-- div. for thickness flux -- JAX, GPU-resident, called from the
#-- jit-compiled RK step in _dt.py. OPS is an ops.JaxOps bundle
#-- (see ops.to_jax); replaces calc_hmap's edge-remap + the old
#-- csr-matrix divergence with padded-gather equivalents.

    hh_edge = calc_hh_edge(ops, hh_cell)

    uh_flux = uu_edge * hh_edge

    hh_tend = hh_tend + gather_apply(ops.div, uh_flux) / ops.cell_area

    return hh_tend


def tend_upgf(ops, hh_cell, zb_cell, gravity, uu_tend):

#-- get z pressure gradient -- JAX, GPU-resident, see tend_hadv above.

    zt_cell = zb_cell + hh_cell

    uu_tend = uu_tend + gravity * gather_apply(ops.grad, zt_cell)

    return uu_tend


#-- Stage 1: momentum advection + Coriolis (vorticity-flux formulation,
#-- Coriolis fused into pv_* per main's _build_pv/calc_u_pv). All
#-- functions below are pure JAX, GPU-resident, called from rhs_slw_u.
#--
#-- Assumptions specific to the Galewsky jet test case (no land
#-- boundaries on this mesh): main's wall/partial-cell correction
#-- factors mesh.edge.perp, mesh.vert.slip, mesh.edge.part/vert.part
#-- all reduce to 1 (no-op) when mesh.edge.mask is all-False, verified
#-- against msh.py's init_obcs -- so they're omitted here rather than
#-- threaded through as extra JaxOps fields. Revisit if this port is
#-- ever pointed at a regional/walled mesh (Stage 5 territory).
#--
#-- Config knobs baked in at their swe.py defaults (not threaded
#-- through as CLI flags, consistent with the rest of this reduced
#-- port): hh_scheme=CENTRE (see calc_hh_edge/calc_hh_quad above),
#-- ke_weight=1.0 + ke_method=1.0 (pure cell-wing KE remap, no
#-- dual-blend term), wetdry_h0=0.0 (its swe.py default; the KE
#-- wet-dry factor below still applies since it doesn't vanish at
#-- wetdry_h0=0, it's a genuine hh_cell/hh_quad ratio correction).

def calc_hh_dual(ops, hh_cell):

#-- cell-to-dual thickness remap

    return gather_apply(ops.dual_kite, hh_cell) / ops.dual_area


def calc_hh_quad(ops, hh_edge, hh_dual):

#-- thickness at the PV "quad" point -- Simpson's-rule blend of the
#-- edge value and its two neighbouring duals (main's calc_hmap,
#-- CENTRE-scheme branch).

    return (4.0 * hh_edge + gather_apply(ops.edge_vert, hh_dual)) / 6.0


def calc_vv_edge(ops, uu_edge):

#-- tangential (perpendicular) velocity reconstruction -- main's
#-- calc_perp, LSQR form. (mesh.edge.perp wall-factor omitted, see
#-- note above -- it's 1 everywhere for the jet mesh.)

    return gather_apply(ops.edge_perp, uu_edge)


def calc_u_ke(ops, hh_cell, hh_quad, uu_edge, vv_edge):

#-- reconstruct kinetic energy 1/2 |u|^2 on cells -- main's
#-- calc_u_ke/_calc_u_ke, at ke_weight=1.0/ke_method=1.0 (see note
#-- above): pure edge-to-cell remap via cell_wing_sums, with the
#-- per-edge wet-dry thickness-ratio correction hFAC still applied
#-- (it's active even at wetdry_h0=0, since it's just (hh_cell /
#-- hh_quad)^2, not a thickness floor).

    ke_edge = 0.5 * (uu_edge * uu_edge + vv_edge * vv_edge)

    hq_gather = hh_quad[ops.cell_wing.idx]
    ke_gather = ke_edge[ops.cell_wing.idx]

    h_fac = (hh_cell[:, None] / hq_gather) ** 2

    ke_cell = jnp.sum(
        ops.cell_wing.wgt * h_fac * ke_gather, axis=-1) / ops.cell_area

    return ke_cell


def calc_u_pv(ops, uu_edge, ff_dual, ff_edge, ff_cell):

#-- relative + absolute (pv = rv + f) vorticity, remapped to every
#-- staggering the upwind blend below needs -- main's
#-- calc_u_pv/_build_pv, pre-upwinding portion. Returns pv_dual,
#-- pv_wide, pv_cell (dual/dual-Gassmann-widened/cell centred) plus
#-- pv_edge_ctr, the centred (non-upwinded) edge estimate that main's
#-- upwinding() blends against pv_wide/pv_dual/pv_cell to get the
#-- final, upwind-biased pv_edge.

    rv_dual = gather_apply(ops.dual_curl, uu_edge) / ops.dual_area
    pv_dual = rv_dual + ff_dual

    dual_area_gather = ops.dual_area[ops.edge_vert.idx]
    rv_edge = jnp.sum(
        ops.edge_vert.wgt * dual_area_gather * rv_dual[ops.edge_vert.idx],
        axis=-1) / ops.quad_area
    pv_edge_ctr = rv_edge + ff_edge

    rv_wide = jnp.sum(
        ops.dual_edge.wgt * rv_edge[ops.dual_edge.idx],
        axis=-1) / jnp.sum(ops.dual_edge.wgt, axis=-1)
    pv_wide = rv_wide + ff_dual

    rv_cell = gather_apply(ops.cell_kite, rv_dual) / ops.cell_area
    pv_cell = rv_cell + ff_cell

    return pv_dual, pv_wide, pv_cell, pv_edge_ctr


PV_UPWIND = 1.0000   # cnfg default (--pv-upwind); AUST-adapt bias scale
UP_TINY_  = 1.0E-02  # hardcoded floor inside main's _upwinding (kx.pyx)


def calc_pv_edge(ops, pv_dual, pv_wide, pv_cell, pv_edge_ctr,
                  uu_edge, vv_edge, pv_tiny, uu_tiny):

#-- upwind-biased edge PV -- main's upwinding(), AUST-adapt branch
#-- (cnfg.pv_scheme default). Blends the centred estimate pv_edge_ctr
#-- against an upwind correction sized by how much pv disagrees across
#-- the two duals either side of the edge (up_sum_edge), scaled by the
#-- local PV gradient (dN_edge/dP_edge) and a smooth 0-1 limiter.

    dN_edge = gather_apply(ops.grad, pv_cell)
    dP_edge = gather_apply(ops.grad_perp, pv_dual)

    diff_vert = jnp.abs(pv_wide - pv_dual)
    up_sum_edge = gather_apply(ops.edge_vert, diff_vert)

    # mesh.edge.slen = 0.5 * sqrt(edge.area * 2), doubled on wall
    # edges in main -- omitted here, no walls on the jet mesh.
    slen = jnp.sqrt(ops.edge_area / 2.0)

    pv_rms = jnp.sqrt(jnp.mean(pv_wide * pv_wide))
    pv_tiny = jnp.maximum(
        pv_tiny, 2.0 * jnp.finfo(reals_t).eps * pv_rms)

    ds_edge = pv_tiny + slen * 0.5 * (
        jnp.abs(dN_edge) + jnp.abs(dP_edge))

    bias = PV_UPWIND * up_sum_edge / ds_edge
    bias = bias * bias / (bias * bias + 1.0)
    bias = bias + UP_TINY_

    um_edge = uu_tiny + jnp.sqrt(uu_edge * uu_edge + vv_edge * vv_edge)

    pv_edge = pv_edge_ctr - bias / um_edge * (
        uu_edge * dN_edge + vv_edge * dP_edge) * slen

    return pv_edge


PV_WEIGHT = 0.1000  # cnfg default (--pv-weight): linear/nonlinear PV split


def tend_uadv(ops, hh_edge, hh_quad, uu_edge, pv_edge, ke_cell,
              ff_edge, uu_tend):

#-- energy-neutral momentum advection: KE-gradient + PV-flux, Coriolis
#-- already fused into pv_edge upstream (calc_pv_edge above) -- main's
#-- tend_uadv/_tend_uadv. ff_dual/ff_cell aren't used by this term in
#-- main either, only ff_edge (re-splitting out the linear pv_weight
#-- share for the energy-neutral perp-flux average). Main also gates
#-- ke_grad by (not cnfg.no_advect) and the whole sum by
#-- mesh.edge.fmsk (=1-edge.mask) -- both omitted here since they're
#-- identity at this port's defaults (advection on, no walls).

    pv_split = (pv_edge - ff_edge * PV_WEIGHT) / hh_quad
    uh_flux = uu_edge * hh_edge
    fh_flux = 2.0 * PV_WEIGHT * ff_edge / hh_quad

    uh_gather = uh_flux[ops.flux_perp.idx]
    fh_gather = fh_flux[ops.flux_perp.idx]
    pv_gather = pv_split[ops.flux_perp.idx]

    uv_flux = -jnp.sum(
        ops.flux_perp.wgt * uh_gather *
        (fh_gather + pv_split[:, None] + pv_gather), axis=-1)

    ke_grad = gather_apply(ops.grad, ke_cell)

    uu_tend = uu_tend + ke_grad + 0.5 * uv_flux

    return uu_tend

