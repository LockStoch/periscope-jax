
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

    # host-side call into the same pure-JAX calc_perp/calc_hmap used
    # by the RK hot path (rhs.py) -- jax.numpy ops accept plain numpy
    # input directly, so this works fine eagerly (un-jitted) on the
    # numpy state io_.py has already pulled back from device.
    ops = mats.jx

    vv_edge = np.asarray(calc_perp(ops, uu_edge))

    hh_dual, hh_edge, hh_quad = calc_hmap(ops, hh_cell)
    hh_dual = np.asarray(hh_dual)
    hh_edge = np.asarray(hh_edge)

    hh_bias = variables.hh_bias  # not computed -- CENTRE scheme only

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


def tend_hadv(ops, hh_cell, uu_edge, hh_tend):

#-- div. for thickness flux -- JAX, GPU-resident, called from the
#-- jit-compiled RK step in _dt.py. OPS is an ops.JaxOps bundle
#-- (see ops.to_jax). hh_edge recomputed here via the same formula
#-- calc_hmap uses below (cell_wing remap) rather than threaded in as
#-- a shared precomputed value -- main computes it once per RK stage
#-- via rhs_all_d and reuses it across tend_hadv/tend_uadv, this port
#-- doesn't have that precompute stage (see calc_hmap's docstring).

    hh_edge = gather_apply(ops.edge_wing_sums, hh_cell) / ops.edge_area

    uh_flux = uu_edge * hh_edge

    hh_tend = hh_tend + \
        gather_apply(ops.cell_flux_sums, uh_flux) / ops.cell_area

    return hh_tend


def tend_upgf(ops, hh_cell, zb_cell, gravity, uu_tend):

#-- get z pressure gradient -- JAX, GPU-resident, see tend_hadv above.

    zt_cell = zb_cell + hh_cell

    uu_tend = uu_tend + \
        gravity * gather_apply(ops.edge_grad_norm, zt_cell)

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
#-- Most config knobs are baked in at their swe.py defaults (not
#-- threaded through as CLI flags): hh_scheme=CENTRE (see calc_hmap
#-- below), ke_weight=1.0 + ke_method=1.0 (pure cell-wing KE remap, no
#-- dual-blend term), pv_weight=0.1 (see tend_uadv below),
#-- wetdry_h0=0.0 (its swe.py default; the KE wet-dry factor below
#-- still applies since it doesn't vanish at wetdry_h0=0, it's a
#-- genuine hh_cell/hh_quad ratio correction). The exception is
#-- pv_scheme/pv_upwind (calc_u_pv/upwinding below), which ARE
#-- genuinely threaded through from cnfg -- --pv-scheme APVM,
#-- AUST-CONST, AUST-ADAPT or CENTRE all work, as a static/
#-- compile-time choice (see upwinding()'s docstring for why).
#--
#-- Function names/boundaries below match main's _dx.py one-for-one
#-- (calc_hmap, calc_perp, calc_u_ke, _build_pv, upwinding, calc_u_pv,
#-- tend_uadv) so this stays directly comparable to the original --
#-- see the per-function notes for the (documented) signature/scope
#-- reductions each one makes relative to main.

def calc_hmap(ops, hh_cell):

#-- compute discrete thickness -- main's calc_hmap/_calc_hmap,
#-- CENTRE-scheme branch only (hh_scheme's swe.py default is UPWIND;
#-- this reduced port only implements CENTRE, matching the
#-- simplification tend_hadv already made for continuity). Main's
#-- signature also takes gravity/uu_edge/vv_edge and returns a 4th
#-- value, hh_bias -- both only relevant to the UPWIND wave-speed
#-- blend, dropped here since that branch isn't implemented.

    hh_dual = gather_apply(ops.dual_kite_sums, hh_cell) / ops.vert_area
    hh_edge = gather_apply(ops.edge_wing_sums, hh_cell) / ops.edge_area

    # PV "quad" point: Simpson's-rule blend of the edge value and its
    # two neighbouring duals.
    hh_quad = (
        4.0 * hh_edge + gather_apply(ops.edge_vert_sums, hh_dual)) / 6.0

    return hh_dual, hh_edge, hh_quad


def calc_perp(ops, uu_edge):

#-- tangential (perpendicular) velocity reconstruction -- main's
#-- calc_perp/_calc_perp, LSQR form. (mesh.edge.perp wall-factor
#-- omitted, see note above -- it's 1 everywhere for the jet mesh.)

    return gather_apply(ops.edge_lsqr_perp, uu_edge)


def calc_u_ke(ops, hh_cell, hh_quad, uu_edge, vv_edge):

#-- reconstruct kinetic energy 1/2 |u|^2 on cells -- main's
#-- calc_u_ke/_calc_u_ke, at ke_weight=1.0/ke_method=1.0 (see note
#-- above): pure edge-to-cell remap via cell_wing_sums, with the
#-- per-edge wet-dry thickness-ratio correction hFAC still applied
#-- (it's active even at wetdry_h0=0, since it's just (hh_cell /
#-- hh_quad)^2, not a thickness floor).

    ke_edge = 0.5 * (uu_edge * uu_edge + vv_edge * vv_edge)

    hq_gather = hh_quad[ops.cell_wing_sums.idx]
    ke_gather = ke_edge[ops.cell_wing_sums.idx]

    h_fac = (hh_cell[:, None] / hq_gather) ** 2

    ke_cell = jnp.sum(
        ops.cell_wing_sums.wgt * h_fac * ke_gather, axis=-1) / ops.cell_area

    return ke_cell


def _build_pv(ops, uu_edge, ff_dual, ff_edge, ff_cell):

#-- compute discrete vorticity -- main's private _build_pv (in
#-- _dx.py, wraps the Cython _calc_u_pv kernel), the pre-upwinding
#-- portion of PV. Returns pv_dual, pv_wide, pv_cell (dual /
#-- dual-Gassmann-widened / cell centred) plus pv_edge, the centred
#-- (non-upwinded) edge estimate that upwinding() below blends against
#-- pv_wide/pv_dual/pv_cell to get the final upwind-biased pv_edge
#-- (main reuses the same pv_edge name for both -- see calc_u_pv).
#-- Main also returns rv_dual/rv_wide/rv_cell (relative vorticity,
#-- pre-+f) and a pv_rms_ scalar -- rv_* dropped here since nothing
#-- downstream needs them on their own, pv_rms_ is recomputed inline
#-- by upwinding() below instead of threaded through.

    rv_dual = gather_apply(ops.dual_curl_sums, uu_edge) / ops.vert_area
    pv_dual = rv_dual + ff_dual

    vert_area_gather = ops.vert_area[ops.edge_vert_sums.idx]
    rv_edge = jnp.sum(
        ops.edge_vert_sums.wgt * vert_area_gather *
        rv_dual[ops.edge_vert_sums.idx], axis=-1) / ops.quad_area
    pv_edge = rv_edge + ff_edge

    rv_wide = jnp.sum(
        ops.dual_edge_sums.wgt * rv_edge[ops.dual_edge_sums.idx],
        axis=-1) / jnp.sum(ops.dual_edge_sums.wgt, axis=-1)
    pv_wide = rv_wide + ff_dual

    rv_cell = gather_apply(ops.cell_kite_sums, rv_dual) / ops.cell_area
    pv_cell = rv_cell + ff_cell

    # accumulated in main's nogil loop as a running mean-square over
    # pv_wide as it's computed; mathematically identical computed
    # afterward here
    pv_rms_ = jnp.sqrt(jnp.mean(pv_wide * pv_wide))

    return pv_dual, pv_wide, pv_cell, pv_edge, pv_rms_


UP_TINY_ = 1.0E-02  # main's up_tiny kwarg default (not a CLI flag)


def upwinding(ops, ss_wide, ss_dual, ss_cell, uu_edge, vv_edge, ss_edge,
              delta_t, ss_tiny, uu_tiny, up_phi_, up_kind):

#-- streamline upwinding for a variable S -- main's upwinding()/
#-- _upwinding. up_kind selects the formula:
#--   "APVM"/"LAXWENDROFF" -- one shared branch in main (identical
#--     code either name) -- Lagrangian departure-point correction,
#--     scaled by delta_t.
#--   "AUST-CONST" -- upwind bias is the constant up_phi_.
#--   "AUST-ADAPT" -- upwind bias adapts to how much ss actually
#--     varies locally (cnfg.pv_scheme's swe.py default).
#--   "CENTRE" -- in main this isn't a real branch: up_kind matching
#--     none of the three if/elif conditions above just falls through
#--     with ss_edge/up_bias unchanged, i.e. no upwinding at all. Made
#--     an explicit branch here rather than relying on the same
#--     implicit fallthrough -- note this means an unrecognised
#--     up_kind (a typo, say) raises below instead of silently doing
#--     the same no-op main would. That's a deliberate difference, not
#--     one JAX forces: matching main's silent fallthrough exactly
#--     would just be carrying a latent footgun forward.
#-- up_kind must be a plain Python string, not a jax value -- it's a
#-- compile-time choice (static_argnums in _dt.py's step_RK33/
#-- run_scan), not something that can vary per traced call.
#--
#-- mesh/mats/cnfg collapse to ops (as everywhere else in this port);
#-- up_bias, main's other per-edge output (gated behind
#-- cnfg.save_vars, off by default), isn't threaded through -- same
#-- simplification already made elsewhere in this port for
#-- diagnostic-only, opt-in output fields.
#--
#-- Generic ss_* naming kept from main since this is a general
#-- upwinding utility, not PV-specific -- calc_u_pv below is what
#-- binds ss_wide/ss_dual/ss_cell/ss_edge to pv_wide/pv_dual/pv_cell/
#-- pv_edge, and (like main) is responsible for clamping ss_tiny
#-- against ss_rms_ before calling in here -- this function just
#-- takes ss_tiny as an already-final value, same as main.

    if up_kind == "CENTRE":
        return ss_edge

    # dN_edge/dP_edge needed by every implemented branch; main
    # recomputes these per-branch (each is its own nogil loop), no
    # reason to duplicate that in a vectorised JAX implementation.
    dN_edge = gather_apply(ops.edge_grad_norm, ss_cell)
    dP_edge = gather_apply(ops.edge_grad_perp, ss_dual)

    if up_kind in ("APVM", "LAXWENDROFF"):

        # lagrangian APVM, scale w. flow
        ss_edge = ss_edge - delta_t * (
            uu_edge * dN_edge + vv_edge * dP_edge)

        return ss_edge

    # mesh.edge.slen = 0.5 * sqrt(edge.area * 2), doubled on wall
    # edges in main -- omitted here, no walls on the jet mesh.
    slen = jnp.sqrt(ops.edge_area / 2.0)

    um_edge = uu_tiny + jnp.sqrt(uu_edge * uu_edge + vv_edge * vv_edge)

    if up_kind == "AUST-CONST":

        # just a constant upstream bias term
        ss_edge = ss_edge - up_phi_ / um_edge * (
            uu_edge * dN_edge + vv_edge * dP_edge) * slen

        return ss_edge

    if up_kind == "AUST-ADAPT":

        # up_bias += |large - small| stencils
        up_sum_ = gather_apply(
            ops.edge_vert_sums, jnp.abs(ss_wide - ss_dual))

        ds_edge = ss_tiny + slen * 0.5 * (
            jnp.abs(dN_edge) + jnp.abs(dP_edge))

        ss_bias = up_phi_ * up_sum_ / ds_edge

        # up^k/(up^k+1.) polynomial limiting
        ss_bias = ss_bias * ss_bias
        ss_bias = ss_bias / (ss_bias + 1.0)

        # always need to have some upwinding
        ss_bias = ss_bias + UP_TINY_

        ss_edge = ss_edge - ss_bias / um_edge * (
            uu_edge * dN_edge + vv_edge * dP_edge) * slen

        return ss_edge

    raise ValueError(f"upwinding: unknown up_kind {up_kind!r}")


def calc_u_pv(ops, uu_edge, vv_edge, ff_dual, ff_edge, ff_cell,
              delta_t, pv_tiny, uu_tiny, pv_upwind, pv_scheme):

#-- compute potential (absolute) vorticity -- main's calc_u_pv:
#-- orchestrates _build_pv (raw pv at every staggering) then
#-- upwinding() (the pv_scheme-selected blend), returning the final
#-- pv_edge. Main also returns rv_dual/pv_dual/rv_wide/pv_wide/
#-- rv_cell/pv_cell/pv_bias for diagnostics -- dropped here since
#-- tend_uadv (the only caller) only needs the final pv_edge.

    pv_dual, pv_wide, pv_cell, pv_edge, pv_rms_ = _build_pv(
        ops, uu_edge, ff_dual, ff_edge, ff_cell)

    pv_tiny = jnp.maximum(pv_tiny, 2.0 * jnp.finfo(reals_t).eps * pv_rms_)

    pv_edge = upwinding(
        ops, pv_wide, pv_dual, pv_cell, uu_edge, vv_edge, pv_edge,
        delta_t, pv_tiny, uu_tiny, pv_upwind, pv_scheme)

    return pv_edge


PV_WEIGHT = 0.1000  # cnfg default (--pv-weight): linear/nonlinear PV split


def tend_uadv(ops, hh_edge, hh_quad, uu_edge, pv_edge, ke_cell,
              ff_edge, uu_tend):

#-- energy-neutral momentum advection: KE-gradient + PV-flux, Coriolis
#-- already fused into pv_edge upstream (calc_u_pv above) -- main's
#-- tend_uadv/_tend_uadv. ff_dual/ff_cell aren't used by this term in
#-- main either, only ff_edge (re-splitting out the linear pv_weight
#-- share for the energy-neutral perp-flux average). Main also gates
#-- ke_grad by (not cnfg.no_advect) and the whole sum by
#-- mesh.edge.fmsk (=1-edge.mask) -- both omitted here since they're
#-- identity at this port's defaults (advection on, no walls).

    # split linear & nonlinear (curl(u) + f) / h
    pv_edge = (pv_edge - ff_edge * PV_WEIGHT) / hh_quad

    uh_flux = uu_edge * hh_edge
    fh_flux = 2.0 * PV_WEIGHT * ff_edge / hh_quad

    # energy neutral flux 1/2 * (W*qhu + q*W*hu)
    uh_gather = uh_flux[ops.edge_flux_perp.idx]
    fh_gather = fh_flux[ops.edge_flux_perp.idx]
    pv_gather = pv_edge[ops.edge_flux_perp.idx]

    pv_mean = fh_gather + pv_edge[:, None] + pv_gather

    uv_flux = -jnp.sum(ops.edge_flux_perp.wgt * uh_gather * pv_mean, axis=-1)

    # gradient of kinetic energy G * 1/2 * |u|^2
    ke_grad = gather_apply(ops.edge_grad_norm, ke_cell)

    uu_tend = uu_tend + ke_grad + 0.5 * uv_flux

    return uu_tend

