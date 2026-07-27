
import time
import math
import numpy as np

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

def soft_step(xx, lo, up):

#-- soft step function, using a Hermite-like polynomial
#-- https://en.wikipedia.org/wiki/smoothstep

    yy = np.minimum(1., 
         np.maximum(0., (xx - lo) / (up - lo)))

    return (
    yy * yy * yy * (yy * (6. * yy - 15.) + 10.)
           )

def hrmn_mean(xone, xtwo):

#-- harmonic mean of vectors (ie. biased toward lesser)

    return 2. * xone * xtwo / (xone + xtwo)


def scale_mix(mesh, mats, cnfg):

#-- local gridsize scaling on div^k and del^k operators

    # diam. of equiv. circle
    dx_cell = 2. * np.sqrt(mesh.cell.area / np.pi)

    # cell topo. dissipation
    sf_cell = 1. * cnfg.msh_fixes * \
        np.sqrt(np.abs(mesh.cell.topo - 6))
    sf_cell = np.asarray(sf_cell, dtype=reals_t)

    # smooth near grid-scale
    dx_edge = mats.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = mats.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    sf_edge = mats.edge_wing_sums * sf_cell
    sf_edge/= mesh.edge.area
    sf_cell = mats.cell_wing_sums * sf_edge
    sf_cell/= mesh.cell.area

    dx_edge = mats.edge_wing_sums * dx_cell
    dx_edge/= mesh.edge.area
    dx_cell = mats.cell_wing_sums * dx_edge
    dx_cell/= mesh.cell.area
    
    sf_edge = mats.edge_wing_sums * sf_cell
    sf_edge/= mesh.edge.area
    sf_cell = mats.cell_wing_sums * sf_edge
    sf_cell/= mesh.cell.area

    if (cnfg.ref_scale > 0.0):
        s2_edge = (dx_edge / cnfg.ref_scale) ** 1
        s4_edge = (dx_edge / cnfg.ref_scale) ** 3
    else:
        s2_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)
        s4_edge = np.ones(
            (mesh.edge.size), dtype=reals_t)

    s2_edge*= (1. + sf_edge)
    s4_edge*= (1. + sf_edge)

    return s2_edge, s4_edge, sf_edge

 
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
        
    nu_turb = variables.nu_turb  # lagged values

    nu_thin = variables.nu_thin

    nu_wave = variables.nu_wave
    os_wave = variables.os_wave

    nu_shoc = variables.nu_shoc
    os_shoc = variables.os_shoc

    return hh_edge, hh_dual, hh_bias, \
           ke_cell, ke_bias, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, pv_edge, pv_bias, \
           vv_edge, nu_turb, \
           nu_wave, os_wave, nu_shoc, os_shoc, \
           nu_thin, uu_filt, Xi_tide, Xi_self


def invariant(mesh, mats, flow, cnfg, hh_cell, uu_edge,
                                      qq_cell):

#-- compute the discrete energy and enstrophy invariants

    ff_dual = variables.ff_vert
    ff_edge = variables.ff_edge
    ff_cell = variables.ff_cell

    zb_cell = variables.zb_cell

    gravity = flow.gravity

    vv_edge = calc_perp(mesh, mats, cnfg, uu_edge)

    hh_dual, hh_edge, hh_quad, hh_bias = calc_hmap(
        mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge)

    ke_edge = uu_edge ** 2
    ke_edge*= hh_edge * mesh.edge.area
    
    pe_cell = flow.gravity * (
        hh_cell * 0.5 + zb_cell - np.min(zb_cell))

    pe_cell*= hh_cell * mesh.cell.area

    kp_sums = math.fsum(ke_edge) \
            + math.fsum(pe_cell)

    rv_dual, pv_dual, rv_wide, pv_wide, \
    rv_cell, pv_cell, \
    pv_edge, pv_bias = calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +1. / 2. * cnfg.time_step)

    # include wet-dry ramp in pv budget
    hh_dtol = cnfg.wetdry_h0 + cnfg.hh_tiny
    hh_ramp = hh_dual / hh_dtol / 10. - .01
    hh_ramp = np.maximum(+0.0, 
              np.minimum(+1.0, hh_ramp))

    # pv is curl(u)+f here, so factor hh dependence
    pv_sums = 0.5 * math.fsum(
        mesh.vert.area * (hh_ramp ** 2)
                       * (pv_dual ** 2 / hh_dual))

    return kp_sums, pv_sums


def calc_obcs(mesh, mats, cnfg, 
        hh_edge, uu_edge, 
        gravity, hE_prev, uE_prev, hE_next, uE_next):
        
#-- setup open bnd. conditions
   
    if (hE_prev is None): return hh_edge, uu_edge
    if (uE_prev is None): return hh_edge, uu_edge
   
    ttic = time.time()
        
    hh_edge, uu_edge = _calc_obcs(
        mesh, mats, cnfg, 
        hh_edge, uu_edge, gravity, 
        hE_prev, uE_prev, hE_next, uE_next)
        
    ttoc = time.time()
    tcpu.calc_obcs = tcpu.calc_obcs + (ttoc - ttic)
        
    return hh_edge, uu_edge
    
    
def calc_udry(
        mesh, mats, cnfg, hh_edge, uu_edge, vv_edge):
        
#-- apply wet-dry velocity lim.
   
    nu_thin = variables.nu_thin

    if (cnfg.wetdry_h0 <= 0.): 
        return uu_edge, vv_edge, nu_thin
   
    ttic = time.time()
        
    hh_tiny = cnfg.wetdry_h0 * 10.0
    
    uu_edge, vv_edge, nu_thin = _calc_udry(
        mesh, mats, cnfg, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
        
    ttoc = time.time()
    tcpu.calc_udry = tcpu.calc_udry + (ttoc - ttic)
        
    return uu_edge, vv_edge, nu_thin


def upwinding(mesh, mats, cnfg, 
        ss_wide, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias,
        delta_t, sv_tiny, uu_tiny,
        up_kind, up_phi_):

#-- streamline upwind eval.'s

    ttic = time.time()

    ss_edge, up_bias = _upwinding(
        mesh, mats, cnfg, 
        ss_wide, ss_dual, ss_cell, uu_edge, vv_edge, 
        ss_edge, up_bias, 
        delta_t, sv_tiny, uu_tiny, 
        up_kind, up_phi_)

    ttoc = time.time()
    tcpu.upwinding = tcpu.upwinding + (ttoc - ttic)

    return ss_edge, up_bias


def calc_hmap(mesh, mats, cnfg, 
        gravity, hh_cell, uu_edge, vv_edge):

#-- compute discrete thickness

    ttic = time.time()
    
    hh_dual, hh_edge, hh_quad, hh_bias = \
        _calc_hmap(mesh, mats, cnfg, 
            gravity, hh_cell, uu_edge, vv_edge)
    
    ttoc = time.time()
    tcpu.calc_hmap = tcpu.calc_hmap + (ttoc - ttic)

    return hh_dual, hh_edge, hh_quad, hh_bias
    

def calc_u_ke(mesh, mats, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        delta_t):

#-- reconstruct kinetic energy

    ttic = time.time()

    up_edge = variables.ke_bias
 
    ke_cell = _calc_u_ke(
        mesh, mats, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge)

    ttoc = time.time()
    tcpu.calc_u_ke = tcpu.calc_u_ke + (ttoc - ttic)

    return ke_cell, up_edge


def _build_pv(mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell,
        delta_t):
           
#-- compute discrete vorticity
              
    ttic = time.time()
              
    rv_dual, pv_dual, rv_wide, pv_wide, pv_rms_, \
    rv_cell, pv_cell, \
    rv_edge, pv_edge = _calc_u_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell)
    
    ttoc = time.time()
    tcpu.calc_u_pv = tcpu.calc_u_pv + (ttoc - ttic)

    return rv_dual, pv_dual, \
           rv_wide, pv_wide, pv_rms_, \
           rv_cell, pv_cell, \
           rv_edge, pv_edge
              
              
def calc_u_pv(mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell,
        delta_t):
  
#-- compute discrete vorticity
  
    rv_dual, pv_dual, rv_wide, pv_wide, pv_rms_, \
    rv_cell, pv_cell, \
    rv_edge, pv_edge =  _build_pv(
        mesh, mats, cnfg, 
        hh_cell, hh_quad, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        delta_t)
    
    up_edge = variables.pv_bias
            
    uu_tiny = cnfg.uu_tiny * 1.
    pv_tiny = cnfg.pv_tiny * 1.
    pv_tiny = max (pv_tiny, 
        2.0 * np.finfo(reals_t).eps * pv_rms_)

    pv_edge, up_edge =  upwinding(
        mesh, mats, cnfg, 
        pv_wide, pv_dual, pv_cell, uu_edge, vv_edge, 
        pv_edge, up_edge,
        delta_t, pv_tiny, uu_tiny, 
        cnfg.pv_scheme, cnfg.pv_upwind)

    return rv_dual, pv_dual, rv_wide, pv_wide, \
           rv_cell, pv_cell, \
           pv_edge, up_edge


def _build_qe(mesh, mats, cnfg, 
        qq_cell, uu_edge, vv_edge, delta_t):

#-- compute discrete transport

    ttic = time.time()

    qq_dual = mats.dual_kite_sums * qq_cell
    qq_dual/= mesh.vert.area

    qq_edge = mats.edge_wing_sums * qq_cell
    qq_edge/= mesh.edge.area

    ttoc = time.time()
    tcpu.calc_qmap = tcpu.calc_qmap + (ttoc - ttic)

    return qq_dual, qq_edge


def calc_qmap(mesh, mats, cnfg, 
        qq_cell, uu_edge, vv_edge, delta_t):
  
#-- compute discrete transport
  
    qq_dual, qq_edge = _build_qe(
        mesh, mats, cnfg, 
        qq_cell, uu_edge, vv_edge, delta_t)
    
    up_edge = variables.pv_bias
            
    uu_tiny = cnfg.uu_tiny * 1.
    qq_rms_ = 0.0  # unused
    qq_tint = 0.0
    qq_tiny = max (qq_tiny, 
        2.0 * np.finfo(reals_t).eps * qq_rms_)

    qq_edge, up_edge =  upwinding(
        mesh, mats, cnfg, 
        qq_dual, qq_dual, qq_cell, uu_edge, vv_edge, 
        qq_edge, up_edge,
        delta_t, qq_tiny, uu_tiny, 
        "LAXWENDROFF", 0.5)

    return qq_edge, up_edge
              
              
def calc_perp(mesh, mats, cnfg, uu_edge):

#-- get tangential velocity

    ttic = time.time()

    vv_edge = _calc_perp(mesh, mats, cnfg, uu_edge)

    ttoc = time.time()
    tcpu.calc_perp = tcpu.calc_perp + (ttoc - ttic)

    return vv_edge
              
              
def tend_hadv(mesh, mats, cnfg, hh_edge, hh_cell, 
                                uu_edge,
                                gravity, 
                                hh_tend):

#-- div. for thickness flux

    ttic = time.time()

    hh_tend = _tend_hadv(
        mesh, mats, cnfg, hh_edge, 
            hh_cell, uu_edge, gravity, hh_tend)

    ttoc = time.time()
    tcpu.tend_hadv = tcpu.tend_hadv + (ttoc - ttic)

    return hh_tend


def tend_qadv(mesh, mats, cnfg, hh_edge, hh_cell, 
                                uu_edge, qq_edge,
                                gravity, 
                                qq_tend):

#-- scalar transport flux

    ttic = time.time()

    qq_tend = _tend_qadv(
        mesh, mats, cnfg, 
        hh_edge, hh_cell, uu_edge, qq_edge, gravity, 
        qq_tend)

    ttoc = time.time()
    tcpu.tend_qadv = tcpu.tend_qadv + (ttoc - ttic)

    return qq_tend
    

def tend_uadv(mesh, mats, cnfg, 
        hh_edge, hh_quad, uu_edge, pv_edge, ke_cell,
        ff_dual, ff_edge, ff_cell,
        uu_tend):

#-- energy-neutral UV. flux

    ttic = time.time()

    uu_tend = _tend_uadv(
        mesh, mats, cnfg, 
        hh_edge, hh_quad, uu_edge, pv_edge, ke_cell, 
        ff_dual, ff_edge, ff_cell, 
        uu_tend)

    ttoc = time.time()
    tcpu.tend_uadv = tcpu.tend_uadv + (ttoc - ttic)

    return uu_tend
    
    
def tend_upgf(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                xi_self,  
                                uu_tend):

#-- get z pressure gradient

    ttic = time.time()

    hh_tiny = cnfg.hh_tiny * 1.

    uu_tend = _tend_upgf(
        mesh, mats, cnfg, 
        hh_cell, zb_cell, xi_self, gravity, hh_tiny, 
        uu_tend)
        
    ttoc = time.time()
    tcpu.tend_upgf = tcpu.tend_upgf + (ttoc - ttic)

    return uu_tend

    
def calc_umix(mesh, mats, cnfg, rv_dual, rv_cell):

#-- compute leith viscosities

    nu_turb = variables.nu_turb

    if (cnfg.leith_chi == 0): return nu_turb

    ttic = time.time()

    nu_turb = _calc_umix(
        mesh, mats, cnfg, rv_dual, rv_cell)
    
    ttoc = time.time()
    tcpu.calc_umix = tcpu.calc_umix + (ttoc - ttic)

    return nu_turb


def calc_uwav(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                hh_edge, 
                                uu_edge, vv_edge):

#-- compute waves dissipation

    nu_wave = variables.nu_wave

    if (cnfg.waves_chi == 0): return nu_wave

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    nu_wave = _calc_uwav(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, gravity, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
    
    ttoc = time.time()
    tcpu.calc_uwav = tcpu.calc_uwav + (ttoc - ttic)

    return nu_wave


def calc_hmix(mesh, mats, cnfg, hh_cell, zb_cell,
                                gravity,
                                hh_edge, 
                                uu_edge, vv_edge):

#-- compute shock dissipation

    nu_shoc = variables.nu_shoc

    if (cnfg.shock_chi == 0): return nu_shoc

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    nu_shoc = _calc_hmix(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, gravity, 
            hh_tiny, hh_edge, uu_edge, vv_edge)
    
    ttoc = time.time()
    tcpu.calc_hmix = tcpu.calc_hmix + (ttoc - ttic)

    return nu_shoc


def tend_umix(mesh, mats, cnfg, hh_cell, hh_edge, 
                                hh_quad, hh_dual, 
                                uu_edge,
                                nu_turb, nu_wave,
                                nu_thin,
                                uu_tend):

#-- viscous del^k operators

    if (cnfg.uu_visc_k == 0): return uu_tend

    ttic = time.time()
            
    hh_tiny = cnfg.wetdry_h0 * 100.

    uu_tend = _tend_umix(
        mesh, mats, cnfg, 
            hh_cell, hh_edge, hh_quad, hh_dual, 
            uu_edge, 
            nu_turb, nu_wave, nu_thin, 
            hh_tiny, uu_tend)

    ttoc = time.time()
    tcpu.tend_umix = tcpu.tend_umix + (ttoc - ttic)

    return uu_tend
    
    
def tend_hmix(mesh, mats, cnfg, hh_cell, zb_cell, 
                                gravity,
                                nu_shoc, 
                                hh_tend):

#-- diffusive del^k operators

    if (cnfg.hh_diff_k == 0): return hh_tend

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 100.

    hh_tend = _tend_hmix(
        mesh, mats, cnfg, 
            hh_cell, zb_cell, 
            gravity, nu_shoc, hh_tiny, hh_tend)
    
    ttoc = time.time()
    tcpu.tend_hmix = tcpu.tend_hmix + (ttoc - ttic)

    return hh_tend


def calc_tide(mesh, mats, cnfg, gravity, Xi_tide):

#-- calc. ext. tidal forcing 

    if (cnfg.tidal_frc ==""): return Xi_tide

    ttic = time.time()

    tnow = cnfg.timeisnow

    Xi_tide = _calc_tide(
        mesh, mats, cnfg, tnow, gravity, Xi_tide)
        
    ttoc = time.time()
    tcpu.calc_tide = tcpu.calc_tide + (ttoc - ttic)

    return Xi_tide


def calc_self(mesh, mats, cnfg, hh_cell, zb_cell, 
                                gravity, 
                                Xi_self):

#-- calc. self loads geo-pot.

    if (cnfg.sal_solve ==""): return Xi_self

    ttic = time.time()

    tnow = cnfg.timeisnow

    Xi_self = _calc_self(
        mesh, mats, cnfg, 
        tnow, hh_cell, zb_cell, gravity, Xi_self)
        
    ttoc = time.time()
    tcpu.calc_self = tcpu.calc_self + (ttoc - ttic)

    return Xi_self


def tend_ugeo(mesh, mats, cnfg, Xi_prev, Xi_next,
                                hh_cell, 
                                gravity,
                                uu_tend):

#-- get grad of ext. geo-pot.

    if (Xi_prev is None or
            cnfg.no_geopot): return uu_tend

    ttic = time.time()

    hh_tiny = 1. * cnfg.hh_tiny

    uu_tend = _tend_ugeo(
        mesh, mats, cnfg, 
            gravity, hh_tiny,
            Xi_prev, Xi_next, hh_cell, uu_tend)
        
    ttoc = time.time()
    tcpu.tend_ugeo = tcpu.tend_ugeo + (ttoc - ttic)

    return uu_tend


def tend_utde(mesh, mats, cnfg, Xi_tide,
                                hh_cell, zb_cell,
                                gravity, 
                                uu_tend):

#-- get grad of tide geo-pot.

    if (cnfg.tidal_frc== ""):return uu_tend

    ttic = time.time()

    hh_tiny = cnfg.hh_tiny * 1.

    uu_tend = _tend_utde(
        mesh, mats, cnfg, 
            gravity, hh_tiny,
            Xi_tide, hh_cell, zb_cell, uu_tend)
        
    ttoc = time.time()
    tcpu.calc_tide = tcpu.calc_tide + (ttoc - ttic)

    return uu_tend
    
    
def tend_utau(mesh, mats, cnfg, Tu_prev, Tu_next,
                                hh_edge, 
                                uu_tend):

#-- forcing from ext. stress

    if (Tu_prev is None or
            cnfg.no_stress): return uu_tend

    ttic = time.time()

    hh_tiny = cnfg.wetdry_h0 * 10.0

    uu_tend = _tend_utau(
        mesh, mats, cnfg, hh_tiny,
            Tu_prev, Tu_next, hh_edge, uu_tend)
    
    ttoc = time.time()
    tcpu.tend_utau = tcpu.tend_utau + (ttoc - ttic)

    return uu_tend


def tend_uflt(mesh, mats, cnfg, uu_edge, hh_edge, 
                                uu_tend):

#-- filtered btr-bcl drag c_d

    if (cnfg.fltlaw_cd<=0. or
        cnfg.fltlaw_t0<=0.): return uu_tend

    ttic = time.time()

    uu_filt = variables.uu_filt

    uu_filt, uu_tend = _tend_uflt(
        mesh, mats, cnfg, 
            uu_edge, hh_edge, uu_filt, uu_tend)

    ttoc = time.time()
    tcpu.calc_drag = tcpu.calc_drag + (ttoc - ttic)

    return uu_tend


def calc_drag(mesh, mats, cnfg, gravity, dz_drag,
                                c1_edge, c2_edge,
                                z0_edge, 
                                n0_edge):

#-- composite bottom drag c_d

    ttic = time.time()
    
    ke_cell = variables.ke_cell  # from prev. eval.
    ke_edge = variables.ke_edge

    hh_edge = variables.hh_edge
    hh_quad = variables.hh_quad

    hh_tiny = cnfg.hh_tiny * 1.
    ke_tiny = cnfg.ke_tiny * 1.

    cd_edge = _calc_drag(
        mesh, mats, cnfg, hh_tiny, ke_tiny, 
            gravity, hh_edge, hh_quad,
            ke_cell, ke_edge, dz_drag,
            c1_edge, c2_edge,
            z0_edge, n0_edge)
            
    ttoc = time.time()
    tcpu.calc_drag = tcpu.calc_drag + (ttoc - ttic)

    return cd_edge


try:
    # load cython kernels, if compiled
    from _kx import _calc_obcs
    from _kx import _calc_udry
    from _kx import _upwinding
    from _kx import _calc_hmap
    from _kx import _calc_u_ke
    from _kx import _calc_u_pv
    from _kx import _calc_perp
    from _kx import _tend_hadv
    from _kx import _tend_qadv
    from _kx import _tend_uadv
    from _kx import _tend_upgf
    from _kx import _tend_ugeo
    from _kx import _tend_utde
    from _kx import _calc_umix
    from _kx import _calc_uwav
    from _kx import _calc_hmix
    from _kx import _tend_umix
    from _kx import _tend_hmix
    from _kx import _tend_utau
    from _kx import _tend_uflt
    from _kx import _calc_drag
    from tde import _calc_tide
    from sal import _calc_self

except ImportError:
    raise RuntimeError("Cython back-end not found")

