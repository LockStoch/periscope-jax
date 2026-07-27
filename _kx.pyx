
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: cpow=True

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel
from cython.parallel import threadid

from _fp import flt32_t, flt64_t
from _kp cimport FLT32_t, FLT64_t, LOCAL_t

from _fp import reals_t, index_t, bytes_t
from _kp cimport REALS_t, INDEX_t, BYTES_t

from _fp import udata_t, hdata_t, qdata_t
from _kp cimport UDATA_t, HDATA_t, QDATA_t

from _fp import utend_t, htend_t, qtend_t
from _kp cimport UTEND_t, HTEND_t, QTEND_t

from lib cimport sqrtf as sqrt_r
from lib cimport cbrtf as cbrt_r
from lib cimport logf as log_r
from lib cimport fabsf as fabs_r

from mem import variables
from mem import get_vec_v, get_vec_e, get_vec_c, \
                put_vec_v, put_vec_e, put_vec_c

def _calc_obcs(mesh, mats, cnfg,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
        const REALS_t gravity,
    np.ndarray[REALS_t, ndim=1] hE_prev,
    np.ndarray[REALS_t, ndim=1] uE_prev,
    np.ndarray[REALS_t, ndim=1] hE_next,
    np.ndarray[REALS_t, ndim=1] uE_next
            ):
            
#-- update u, h variables to set open BCs
            
    cdef INDEX_t edge, eidx

    cdef LOCAL_t RADIATE
    cdef REALS_t UU_PRED, UU_RELX, HE_EDGE, UE_EDGE
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0

    cdef REALS_t FAST = 2.0 / 1.0
    cdef REALS_t WAVE = 1.0 / 2.0
    cdef REALS_t BIAS = 1.0 / 2.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t time_step = cnfg.time_step
    cdef REALS_t frc_blend = cnfg.frc_blend

    cdef REALS_t wetdry_h0 = cnfg.wetdry_h0
        
    cdef INDEX_t NBCS = mesh.edge.open.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *HE_PREV = &hE_prev[0]
    cdef REALS_t *UE_PREV = &uE_prev[0]
    cdef REALS_t *HE_NEXT = &hE_next[0]
    cdef REALS_t *UE_NEXT = &uE_next[0]
    
    cdef INDEX_t *MESH_EDGE_OPEN = ptr_index_t(
                  mesh.edge.open)
    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)

    if (NBCS > 0):

        with nogil, parallel(num_threads=numthread):
          
            for edge in prange(0, NBCS, schedule="static", 
                    chunksize=chunkedge):
                    
                eidx = MESH_EDGE_OPEN[edge]
                    
                HE_EDGE = (
                    (ONE_ - frc_blend) * HE_PREV[eidx]
                  + (ZERO + frc_blend) * HE_NEXT[eidx]
                          )
                    
                UE_EDGE = (
                    (ONE_ - frc_blend) * UE_PREV[eidx]
                  + (ZERO + frc_blend) * UE_NEXT[eidx]
                          )
                    
                #-- don't "over-dry" boundaries
                HE_EDGE = max(
                          HE_EDGE, wetdry_h0 * ONE_)

                #-- Flather-type bnd. condition
                RADIATE = sqrt_r(WAVE * gravity / 
                          max(
                    HH_EDGE[eidx], wetdry_h0 * ONE_))

                #-- bind outflow by CFL limiter
                RADIATE = RADIATE * (
                    HH_EDGE[eidx] - HE_EDGE
                    )

                RADIATE = min(RADIATE, +FAST * 
                    MESH_EDGE_CLEN [eidx] / time_step)

                RADIATE = max(RADIATE, -FAST *  
                    MESH_EDGE_CLEN [eidx] / time_step)

                UU_PRED = UE_EDGE + RADIATE

                UU_RELX = \
                    (ONE_ - BIAS) * UE_EDGE + \
                    (ZERO + BIAS) * UU_PRED

                if (UE_EDGE<ZERO): #and UU_RELX<ZERO):
                
                #-- inflow: prescribe mass flux
                    UU_EDGE[eidx] =max(      UE_EDGE,
                                   min(ZERO, UU_RELX) 
                                      )
                   #UU_EDGE[eidx] = UE_EDGE

                    HH_EDGE[eidx] = HE_EDGE
                    
                else:
                
                #-- outflow: radiate deviations
                    UU_EDGE[eidx] =max(ZERO, UU_PRED)
                   #UU_EDGE[eidx] = UU_PRED

                   #HH_EDGE[eidx] = upwind, from scheme
                          
    return hh_edge, uu_edge
    
    
def _calc_udry(mesh, mats, cnfg,
        const REALS_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
            ):
    
#-- limit velocity near wet-dry threshold
    
    cdef INDEX_t edge

    cdef REALS_t UU_RAMP
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TEN_ = 10.
    cdef REALS_t EPS_ = .01
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef np.ndarray[REALS_t] nu_thin = variables.nu_thin

    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    cdef REALS_t *NU_THIN = &nu_thin[0]

    # slen scaling as visc. acts on rot(u)
    cdef REALS_t *MESH_EDGE_SLEN = ptr_reals_t(
                  mesh.edge.slen)

    if (hh_tiny > ZERO):

        with nogil, parallel(num_threads=numthread):
          
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):
                    
                UU_RAMP = HH_EDGE[edge] / hh_tiny - EPS_
                    
                UU_RAMP = min(ONE_, UU_RAMP)
                UU_RAMP = max(ZERO, UU_RAMP)

                UU_EDGE[edge]*= UU_RAMP
                VV_EDGE[edge]*= UU_RAMP

                UU_RAMP = ONE_- UU_RAMP

                NU_THIN[edge] = UU_RAMP * (
                    ONE_ / ONE_ * MESH_EDGE_SLEN [edge])
    
    return uu_edge, vv_edge, variables.nu_thin
            

def _upwinding(mesh, mats, cnfg, 
    np.ndarray[REALS_t, ndim=1] ss_wide,
    np.ndarray[REALS_t, ndim=1] ss_dual, 
    np.ndarray[REALS_t, ndim=1] ss_cell,
    np.ndarray[UDATA_t, ndim=1] uu_edge, 
    np.ndarray[REALS_t, ndim=1] vv_edge,
    np.ndarray[REALS_t, ndim=1] ss_edge,
    np.ndarray[REALS_t, ndim=1] up_bias, 
        const REALS_t delta_t,
        const REALS_t ss_tiny, 
        const REALS_t uu_tiny,  up_kind, 
        const REALS_t up_phi_,
        const REALS_t up_tiny = 1.0E-02
              ):
    
#-- streamline upwinding for a variable S
    
    cdef INDEX_t vert, edge, iptr, xidx
    cdef REALS_t xval

    cdef LOCAL_t dN_EDGE, dP_EDGE, 
    cdef LOCAL_t UP_SUM_, DS_EDGE, SS_BIAS
    cdef LOCAL_t UM_EDGE
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t save_bias = "pv_bias" in cnfg.save_vars
        
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *SS_WIDE = &ss_wide[0]
    cdef REALS_t *SS_DUAL = &ss_dual[0]
    cdef REALS_t *SS_CELL = &ss_cell[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    cdef REALS_t *SS_EDGE = &ss_edge[0]
    cdef REALS_t *UP_BIAS = &up_bias[0]
    
    cdef INDEX_t *GRAD_NORM_XPTR = ptr_index_t(
             mats.edge_grad_norm.indptr)
    cdef INDEX_t *GRAD_NORM_XIDX = ptr_index_t(
             mats.edge_grad_norm.indices)
    cdef REALS_t *GRAD_NORM_XVAL = ptr_reals_t(
             mats.edge_grad_norm.data)

    cdef INDEX_t *GRAD_PERP_XPTR = ptr_index_t(
             mats.edge_grad_perp.indptr)
    cdef INDEX_t *GRAD_PERP_XIDX = ptr_index_t(
             mats.edge_grad_perp.indices)
    cdef REALS_t *GRAD_PERP_XVAL = ptr_reals_t(
             mats.edge_grad_perp.data)

    cdef INDEX_t *VERT_TAIL_XPTR = ptr_index_t(
             mats.dual_tail_sums.indptr)
    cdef INDEX_t *VERT_TAIL_XIDX = ptr_index_t(
             mats.dual_tail_sums.indices)
    cdef REALS_t *VERT_TAIL_XVAL = ptr_reals_t(
             mats.dual_tail_sums.data)

    cdef INDEX_t *EDGE_VERT_XPTR = ptr_index_t(
             mats.edge_vert_sums.indptr)
    cdef INDEX_t *EDGE_VERT_XIDX = ptr_index_t(
             mats.edge_vert_sums.indices)
    cdef REALS_t *EDGE_VERT_XVAL = ptr_reals_t(
             mats.edge_vert_sums.data)

    cdef REALS_t *MESH_EDGE_SLEN = ptr_reals_t(
                  mesh.edge.slen)

    if   (up_kind == "APVM" or 
          up_kind == "LAXWENDROFF"):
              
        #-- APVM: anticipated upstream method; lagrangian
        #-- formulation. Upwind departure points, appears
        #-- to be inconsistent in time for RK integrators
  
        with nogil, parallel(num_threads=numthread):
      
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):        
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + xval * SS_CELL[xidx]
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + xval * SS_DUAL[xidx]
                    
            #-- lagrangian APVM, scale w. flow
                SS_EDGE[edge]-= delta_t * (
                        UU_EDGE [edge] * dN_EDGE +
                        VV_EDGE [edge] * dP_EDGE )
            
    elif (up_kind == "AUST-CONST"):

        #-- AUST: anticipated upstream method; APVM meets
        #-- LUST? Upwinds in multi-dimensional sense, vs.
        #-- LUST, which upwinds via tangential dir. only.

        #-- const. upwinding version

        with nogil, parallel(num_threads=numthread):

            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):      
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + xval * SS_CELL[xidx]
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + xval * SS_DUAL[xidx]

            #-- just a constant upstream bias term
                if save_bias: UP_BIAS[edge] = up_phi_
    
            #-- upwind APVM, scale w. grid spacing
                UM_EDGE = uu_tiny + sqrt_r (
                    UU_EDGE[edge] * UU_EDGE[edge] +
                    VV_EDGE[edge] * VV_EDGE[edge]
                    )

                SS_EDGE[edge]-= up_phi_ / UM_EDGE * (
                    UU_EDGE[edge] * dN_EDGE +
                    VV_EDGE[edge] * dP_EDGE
                    ) * MESH_EDGE_SLEN[edge]
                
    elif (up_kind == "AUST-ADAPT"):
        
        #-- AUST: anticipated upstream method; APVM meets
        #-- LUST? Upwinds in multi-dimensional sense, vs.
        #-- LUST, which upwinds via tangential dir. only.

        #-- adapt. upwinding version

        with nogil, parallel(num_threads=numthread):

            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):
            #-- up_bias+= |large - small| stencils
                UP_SUM_ = ZERO
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                
                    UP_SUM_ = UP_SUM_ + fabs_r ( 
                        SS_WIDE[xidx] - SS_DUAL[xidx])
                    
            #-- dN_edge = edge_grad_norm * ss_cell
                dN_EDGE = ZERO
                for iptr in range(GRAD_NORM_XPTR[edge +0], 
                                  GRAD_NORM_XPTR[edge +1]):
                        
                    xval = GRAD_NORM_XVAL[iptr]
                    xidx = GRAD_NORM_XIDX[iptr]
                        
                    dN_EDGE = \
                        dN_EDGE + xval * SS_CELL[xidx]
                        
            #-- dP_edge = edge_grad_perp * ss_dual
                dP_EDGE = ZERO
                for iptr in range(GRAD_PERP_XPTR[edge +0],
                                  GRAD_PERP_XPTR[edge +1]):
                        
                    xval = GRAD_PERP_XVAL[iptr]
                    xidx = GRAD_PERP_XIDX[iptr]
                        
                    dP_EDGE = \
                        dP_EDGE + xval * SS_DUAL[xidx]
                    
            #-- a measure of "difference" on edges
                DS_EDGE = HALF * (fabs_r (dN_EDGE) +
                                  fabs_r (dP_EDGE))
                DS_EDGE = ss_tiny + \
                   MESH_EDGE_SLEN[edge] * DS_EDGE
                
                SS_BIAS = up_phi_ * UP_SUM_ / DS_EDGE
                
            #-- up^k/(up^k+1.) polynomial limiting
                SS_BIAS = SS_BIAS * SS_BIAS
                SS_BIAS = SS_BIAS /(SS_BIAS + ONE_)

            #-- always need to have some upwinding
                SS_BIAS = SS_BIAS + up_tiny

                if save_bias: UP_BIAS[edge] = SS_BIAS

            #-- upwind APVM, scale w. grid spacing
                UM_EDGE = uu_tiny + sqrt_r (
                    UU_EDGE[edge] * UU_EDGE[edge] +
                    VV_EDGE[edge] * VV_EDGE[edge]
                    )

                SS_EDGE[edge]-= SS_BIAS / UM_EDGE * (
                    UU_EDGE[edge] * dN_EDGE +
                    VV_EDGE[edge] * dP_EDGE
                    ) * MESH_EDGE_SLEN[edge]
                
    return ss_edge, up_bias
    

def _calc_hmap(mesh, mats, cnfg,
        const REALS_t gravity,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
    
#-- compute edge & dual centred thickness
    
    cdef INDEX_t vert, edge, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval

    cdef LOCAL_t HH_SUM_, HH_BIAS
    cdef LOCAL_t C1_WAVE, C2_WAVE, UU_WAVE
    cdef HDATA_t H1_CELL, H2_CELL
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    cdef REALS_t FOUR = 4.0
    cdef REALS_t SIX_ = 6.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t save_bias = "hh_bias" in cnfg.save_vars

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]

    cdef np.ndarray[REALS_t] hh_dual = variables.hh_dual
    cdef np.ndarray[REALS_t] hh_edge = variables.hh_edge
    cdef np.ndarray[REALS_t] hh_quad = variables.hh_quad
    cdef np.ndarray[REALS_t] up_bias = variables.hh_bias
    
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef REALS_t *UP_BIAS = &up_bias[0]
    
    cdef INDEX_t *EDGE_WING_XPTR = ptr_index_t(
             mats.edge_wing_sums.indptr)
    cdef INDEX_t *EDGE_WING_XIDX = ptr_index_t(
             mats.edge_wing_sums.indices)
    cdef REALS_t *EDGE_WING_XVAL = ptr_reals_t(
             mats.edge_wing_sums.data)

    cdef INDEX_t *DUAL_KITE_XPTR = ptr_index_t(
             mats.dual_kite_sums.indptr)
    cdef INDEX_t *DUAL_KITE_XIDX = ptr_index_t(
             mats.dual_kite_sums.indices)
    cdef REALS_t *DUAL_KITE_XVAL = ptr_reals_t(
             mats.dual_kite_sums.data)

    cdef INDEX_t *EDGE_VERT_XPTR = ptr_index_t(
             mats.edge_vert_sums.indptr)
    cdef INDEX_t *EDGE_VERT_XIDX = ptr_index_t(
             mats.edge_vert_sums.indices)
    cdef REALS_t *EDGE_VERT_XVAL = ptr_reals_t(
             mats.edge_vert_sums.data)

    cdef REALS_t *MESH_DUAL_AREA = ptr_reals_t(
                  mesh.vert.area)
    cdef REALS_t *MESH_EDGE_AREA = ptr_reals_t(
                  mesh.edge.area)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell
    
    if (cnfg.hh_scheme == "CENTRE"):
    
        with nogil, parallel(num_threads=numthread):
        
            for vert in prange(0, NVRT, schedule="static", 
                    chunksize=chunkvert):
            #-- compute dual-centred thickness
                HH_SUM_ = ZERO
                for iptr in range(DUAL_KITE_XPTR[vert +0], 
                                  DUAL_KITE_XPTR[vert +1]):
                        
                    xval = DUAL_KITE_XVAL[iptr]
                    xidx = DUAL_KITE_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + xval * HH_CELL[xidx]
                    
                HH_DUAL[vert] = (
                    HH_SUM_ / MESH_DUAL_AREA[vert]
                    )
        
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):
            #-- compute edge-centred thickness
                HH_SUM_ = ZERO
                for iptr in range(EDGE_WING_XPTR[edge +0], 
                                  EDGE_WING_XPTR[edge +1]):
                        
                    xval = EDGE_WING_XVAL[iptr]
                    xidx = EDGE_WING_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + xval * HH_CELL[xidx]
                    
                HH_EDGE[edge] = (
                    HH_SUM_ / MESH_EDGE_AREA[edge]
                    )
                
            #-- compute for PV; simpson's rule
                HH_SUM_ = HH_EDGE[edge] * FOUR
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + ONE_ * HH_DUAL[xidx]
                    
                HH_QUAD[edge] = HH_SUM_ / SIX_
            
    else:  # hh_scheme == "UPWIND"
    
        with nogil, parallel(num_threads=numthread):
        
            for vert in prange(0, NVRT, schedule="static", 
                    chunksize=chunkvert):
            #-- compute dual-centred thickness               
                HH_SUM_ = ZERO
                for iptr in range(DUAL_KITE_XPTR[vert +0], 
                                  DUAL_KITE_XPTR[vert +1]):
                        
                    xval = DUAL_KITE_XVAL[iptr]
                    xidx = DUAL_KITE_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + xval * HH_CELL[xidx]

                HH_DUAL[vert] = (
                    HH_SUM_ / MESH_DUAL_AREA[vert]
                    )
        
            for edge in prange(0, NEDG, schedule="static", 
                    chunksize=chunkedge):
                        
                cel1 = mesh_edge_cell[edge, 0] - 1
                cel2 = mesh_edge_cell[edge, 1] - 1
                
                if (cel1 < 0): cel1 = cel2
                if (cel2 < 0): cel2 = cel1
                        
            #-- compute edge-centred thickness
                HH_SUM_ = ZERO
                for iptr in range(EDGE_WING_XPTR[edge +0], 
                                  EDGE_WING_XPTR[edge +1]):
                        
                    xval = EDGE_WING_XVAL[iptr]
                    xidx = EDGE_WING_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + xval * HH_CELL[xidx]
                
                HH_QUAD[edge] = (
                    HH_SUM_ / MESH_EDGE_AREA[edge]
                    )
                
                H1_CELL = HH_CELL[cel1]
                H2_CELL = HH_CELL[cel2]

            #-- compute upwind thickness blend
               #UU_WAVE = fabs_r(UU_EDGE[edge])
                UU_WAVE = +sqrt_r(
                    UU_EDGE[edge] * UU_EDGE[edge] + 
                    VV_EDGE[edge] * VV_EDGE[edge] )
                
                C1_WAVE = (fabs_r(UU_WAVE)
                        +  sqrt_r(gravity * H1_CELL))
                C2_WAVE = (fabs_r(UU_WAVE)
                        +  sqrt_r(gravity * H2_CELL))

            #-- upwind if wavespeed ratio >> 1
                if (C2_WAVE > C1_WAVE):
                    HH_BIAS = max( sqrt_r(
                              C2_WAVE / C1_WAVE - ONE_),
                              H2_CELL / H1_CELL - ONE_)
                else:
                    HH_BIAS = max( sqrt_r(
                              C1_WAVE / C2_WAVE - ONE_),
                              H1_CELL / H2_CELL - ONE_)

                HH_BIAS = min(ONE_, HH_BIAS)
               #HH_BIAS = max(ZERO, HH_BIAS)

                HH_BIAS = HH_BIAS * HH_BIAS * HH_BIAS

                if save_bias: UP_BIAS[edge] = HH_BIAS

                if (UU_EDGE[edge]>= ZERO):
                    HH_EDGE[edge] = HH_BIAS * H1_CELL + \
                        (ONE_-HH_BIAS) * HH_QUAD[edge]
                else:
                    HH_EDGE[edge] = HH_BIAS * H2_CELL + \
                        (ONE_-HH_BIAS) * HH_QUAD[edge]

            #-- compute for PV; simpson's rule
                HH_SUM_ = HH_QUAD[edge] * FOUR
                for iptr in range(EDGE_VERT_XPTR[edge +0], 
                                  EDGE_VERT_XPTR[edge +1]):
                        
                    xidx = EDGE_VERT_XIDX[iptr]
                        
                    HH_SUM_ = \
                        HH_SUM_ + ONE_ * HH_DUAL[xidx]
                    
                HH_QUAD[edge] = HH_SUM_ / SIX_

    return variables.hh_dual, \
           variables.hh_edge, \
           variables.hh_quad, \
           variables.hh_bias

    
def _calc_u_ke(mesh, mats, cnfg, 
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[REALS_t, ndim=1] hh_quad,
    np.ndarray[REALS_t, ndim=1] hh_dual,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
    
#-- compute the kinetic energy 1/2 |u|^2
    
    cdef INDEX_t vert, edge, cell, iptr, xidx
    cdef REALS_t xval, hFAC

    cdef REALS_t HH_VAL_
    cdef LOCAL_t KE_SUM_, AC_SUM_
    cdef LOCAL_t K1_EDGE, K2_EDGE
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t EPS_ = 1.0E-008
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t method_ke = cnfg.ke_method
    cdef REALS_t weight_ke = cnfg.ke_weight

    cdef REALS_t wetdry_h0 = cnfg.wetdry_h0 * 2.
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    if (weight_ke >= ONE_): NVRT = 0  # don't do ke-dual

    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]

    cdef np.ndarray[REALS_t] ke_edge = variables.ke_edge
    cdef np.ndarray[REALS_t] ke_dual = variables.ke_dual
    cdef np.ndarray[REALS_t] ke_cell = variables.ke_cell

    cdef REALS_t *KE_EDGE = &ke_edge[0]    
    cdef REALS_t *KE_DUAL = &ke_dual[0]
    cdef REALS_t *KE_CELL = &ke_cell[0]
    
    cdef INDEX_t *VERT_TAIL_XPTR = ptr_index_t(
             mats.dual_tail_sums.indptr)
    cdef INDEX_t *VERT_TAIL_XIDX = ptr_index_t(
             mats.dual_tail_sums.indices)
    cdef REALS_t *VERT_TAIL_XVAL = ptr_reals_t(
             mats.dual_tail_sums.data)

    cdef INDEX_t *CELL_WING_XPTR = ptr_index_t(
             mats.cell_wing_sums.indptr)
    cdef INDEX_t *CELL_WING_XIDX = ptr_index_t(
             mats.cell_wing_sums.indices)
    cdef REALS_t *CELL_WING_XVAL = ptr_reals_t(
             mats.cell_wing_sums.data)

    cdef INDEX_t *CELL_KITE_XPTR = ptr_index_t(
             mats.cell_kite_sums.indptr)
    cdef INDEX_t *CELL_KITE_XIDX = ptr_index_t(
             mats.cell_kite_sums.indices)
    cdef REALS_t *CELL_KITE_XVAL = ptr_reals_t(
             mats.cell_kite_sums.data)

    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)
    cdef REALS_t *MESH_VERT_AREA = ptr_reals_t(
                  mesh.vert.area)
    cdef REALS_t *MESH_VERT_PART = ptr_reals_t(
                  mesh.vert.part)
    cdef REALS_t *MESH_EDGE_PART = ptr_reals_t(
                  mesh.edge.part)

    with nogil, parallel(num_threads=numthread):
        
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
        #-- calc. kinetic energy on edges: 1/2 * |u|^2
            K1_EDGE = ONE_ * (
                UU_EDGE[edge] * UU_EDGE[edge])
            
            K2_EDGE = HALF * (
                UU_EDGE[edge] * UU_EDGE[edge] +
                VV_EDGE[edge] * VV_EDGE[edge])

            KE_EDGE[edge] =  (
                (ONE_ - method_ke) * K1_EDGE +
                (ZERO + method_ke) * K2_EDGE
                )

        for vert in prange(0, NVRT, schedule="static", 
                chunksize=chunkvert):
        #-- remap kinetic energy M_(v,e) * 1/2 * |u|^2
            KE_SUM_ = ZERO; AC_SUM_ = ZERO
            for iptr in range(VERT_TAIL_XPTR[vert +0],
                              VERT_TAIL_XPTR[vert +1]):
                    
                xval = VERT_TAIL_XVAL[iptr]
                xidx = VERT_TAIL_XIDX[iptr]

            #-- allow for partial subcell contribution
                xval = xval* MESH_EDGE_PART[xidx]

                AC_SUM_ = AC_SUM_ + xval

                KE_SUM_ = KE_SUM_ \
                    + xval * ONE_ * KE_EDGE[xidx]

            AC_SUM_ = (ONE_ - EPS_) * AC_SUM_ \
                    +  EPS_ * MESH_VERT_AREA[vert]

            KE_DUAL[vert] = KE_SUM_ / AC_SUM_

        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- remap kinetic energy M_(c,e) * 1/2 * |u|^2
            KE_SUM_ = ZERO; AC_SUM_ = ZERO
            HH_VAL_ = HH_CELL[cell]            
            for iptr in range(CELL_WING_XPTR[cell +0],
                              CELL_WING_XPTR[cell +1]):
                    
                xval = CELL_WING_XVAL[iptr]
                xidx = CELL_WING_XIDX[iptr]
   
            #-- allow for partial subcell contribution
                xval = xval* MESH_EDGE_PART[xidx]

                AC_SUM_ = AC_SUM_ + xval

                hFAC = HH_VAL_ / \
                   max (wetdry_h0, HH_QUAD[xidx])
                hFAC = hFAC * hFAC
                
                KE_SUM_ = KE_SUM_ \
                    + xval * hFAC * KE_EDGE[xidx]

            AC_SUM_ = (ONE_ - EPS_) * AC_SUM_ \
                    +  EPS_ * MESH_CELL_AREA[cell]

            KE_CELL[cell] = \
                (ZERO + weight_ke) * KE_SUM_ / AC_SUM_

        #-- remap kinetic energy M_(c,v) * 1/2 * |u|^2
            KE_SUM_ = ZERO; AC_SUM_ = ZERO
            HH_VAL_ = HH_CELL[cell]
            for iptr in range(CELL_KITE_XPTR[cell +0],
                              CELL_KITE_XPTR[cell +1]):
                    
                xval = CELL_KITE_XVAL[iptr]
                xidx = CELL_KITE_XIDX[iptr]

            #-- allow for partial subcell contribution
                xval = xval* MESH_VERT_PART[xidx]
   
            #-- blend cell-edge and vert-edge KE forms
                AC_SUM_ = AC_SUM_ + xval

                hFAC = HH_VAL_ / \
                   max (wetdry_h0, HH_DUAL[xidx])
                hFAC = hFAC * hFAC
                
                KE_SUM_ = KE_SUM_ \
                    + xval * hFAC * KE_DUAL[xidx]

            AC_SUM_ = (ONE_ - EPS_) * AC_SUM_ \
                    +  EPS_ * MESH_CELL_AREA[cell]

            KE_CELL[cell]+= \
                (ONE_ - weight_ke) * KE_SUM_ / AC_SUM_
            
    return variables.ke_cell
    

def _calc_u_pv(mesh, mats, cnfg, 
    np.ndarray[HDATA_t, ndim=1] hh_cell, 
    np.ndarray[REALS_t, ndim=1] hh_quad, 
    np.ndarray[REALS_t, ndim=1] hh_dual,
    np.ndarray[UDATA_t, ndim=1] uu_edge, 
    np.ndarray[REALS_t, ndim=1] vv_edge,
    np.ndarray[FLT32_t, ndim=1] ff_dual,
    np.ndarray[FLT32_t, ndim=1] ff_edge,
    np.ndarray[FLT32_t, ndim=1] ff_cell
              ):

#-- compute potential and relative curl u

#-- pv here is abs. vorticity: curl u + f
    
    cdef INDEX_t vert, edge, cell, iptr, xidx
    cdef REALS_t xval
    
    cdef LOCAL_t RV_SUM_, PV_SUM_

    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert
    cdef REALS_t do_advect = (not cnfg.no_advect)

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]
    cdef FLT32_t *FF_DUAL = &ff_dual[0]
    cdef FLT32_t *FF_EDGE = &ff_edge[0]
    cdef FLT32_t *FF_CELL = &ff_cell[0]

    cdef np.ndarray[REALS_t] rv_dual = variables.rv_dual
    cdef np.ndarray[REALS_t] pv_dual = variables.pv_dual
    
    cdef REALS_t *RV_DUAL = &rv_dual[0]
    cdef REALS_t *PV_DUAL = &pv_dual[0]
    
    cdef np.ndarray[REALS_t] rv_cell = variables.rv_cell
    cdef np.ndarray[REALS_t] pv_cell = variables.pv_cell
        
    cdef REALS_t *RV_CELL = &rv_cell[0]
    cdef REALS_t *PV_CELL = &pv_cell[0]
    
    cdef np.ndarray[REALS_t] rv_wide = variables.rv_wide
    cdef np.ndarray[REALS_t] pv_wide = variables.pv_wide
    
    cdef REALS_t *RV_WIDE = &rv_wide[0]
    cdef REALS_t *PV_WIDE = &pv_wide[0]
    
    cdef np.ndarray[REALS_t] rv_edge = variables.rv_edge
    cdef np.ndarray[REALS_t] pv_edge = variables.pv_edge
        
    cdef REALS_t *RV_EDGE = &rv_edge[0]
    cdef REALS_t *PV_EDGE = &pv_edge[0]

    cdef FLT64_t pv_rms_ = 0.0  # for parallel reduction
    
    cdef INDEX_t *DUAL_CURL_XPTR = ptr_index_t(
             mats.dual_curl_sums.indptr)
    cdef INDEX_t *DUAL_CURL_XIDX = ptr_index_t(
             mats.dual_curl_sums.indices)
    cdef REALS_t *DUAL_CURL_XVAL = ptr_reals_t(
             mats.dual_curl_sums.data)

    cdef INDEX_t *EDGE_VERT_XPTR = ptr_index_t(
             mats.edge_vert_sums.indptr)
    cdef INDEX_t *EDGE_VERT_XIDX = ptr_index_t(
             mats.edge_vert_sums.indices)

    cdef INDEX_t *VERT_EDGE_XPTR = ptr_index_t(
             mats.dual_edge_sums.indptr)
    cdef INDEX_t *VERT_EDGE_XIDX = ptr_index_t(
             mats.dual_edge_sums.indices)
    
    cdef INDEX_t *CELL_KITE_XPTR = ptr_index_t(
             mats.cell_kite_sums.indptr)
    cdef INDEX_t *CELL_KITE_XIDX = ptr_index_t(
             mats.cell_kite_sums.indices)
    cdef REALS_t *CELL_KITE_XVAL = ptr_reals_t(
             mats.cell_kite_sums.data)

    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)
    cdef REALS_t *MESH_DUAL_AREA = ptr_reals_t(
                  mesh.vert.area)
    cdef REALS_t *MESH_QUAD_AREA = ptr_reals_t(
                  mesh.quad.area)

    cdef REALS_t *MESH_VERT_SLIP = ptr_reals_t(
                  mesh.vert.slip)

    with nogil, parallel(num_threads=numthread):
    
        for vert in prange(0, NVRT, schedule="static",
                chunksize=chunkvert):
        #-- compute dual-centred vorticity
            RV_SUM_ = ZERO
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_SUM_ = \
                    RV_SUM_ + xval * UU_EDGE[xidx]

            RV_SUM_ = (
                    RV_SUM_ / MESH_DUAL_AREA[vert])
            
            RV_DUAL[vert] = (
                ONE_ - MESH_VERT_SLIP[vert]
                          ) * RV_SUM_ * do_advect

            PV_DUAL[vert] = (
                (RV_DUAL[vert] + FF_DUAL[vert])
                )  

        # ensure u_i not in curl_i
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
        #-- compute edge-centred vorticity
            RV_SUM_ = ZERO
            for iptr in range(EDGE_VERT_XPTR[edge +0], 
                              EDGE_VERT_XPTR[edge +1]):
                    
                xidx = EDGE_VERT_XIDX[iptr]
                xval = MESH_DUAL_AREA[xidx]
                    
                RV_SUM_ = \
                    RV_SUM_ + xval * RV_DUAL[xidx]
        
            RV_EDGE[edge] = (
                    RV_SUM_ / MESH_QUAD_AREA[edge])

            PV_EDGE[edge] = (
                (RV_EDGE[edge] + FF_EDGE[edge])
                )
            
        for vert in prange(0, NVRT, schedule="static", 
                chunksize=chunkvert):
        #-- average rhombi to dual -- a'la Gassmann
            RV_SUM_ = ZERO
            for iptr in range(VERT_EDGE_XPTR[vert +0], 
                              VERT_EDGE_XPTR[vert +1]):
                
                xidx = VERT_EDGE_XIDX[iptr]
                    
                RV_SUM_ = \
                    RV_SUM_ + ONE_ * RV_EDGE[xidx]
             
            RV_WIDE[vert] = (
                    RV_SUM_ /(VERT_EDGE_XPTR[vert +1] - 
                              VERT_EDGE_XPTR[vert +0] 
                          ) )

            PV_WIDE[vert] = (
                (RV_WIDE[vert] + FF_DUAL[vert])
                )

            pv_rms_ += (
                (PV_WIDE[vert] * PV_WIDE[vert]) / NVRT
                )

        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- compute cell-centred vorticity
            RV_SUM_ = ZERO
            for iptr in range(CELL_KITE_XPTR[cell +0], 
                              CELL_KITE_XPTR[cell +1]):
                    
                xval = CELL_KITE_XVAL[iptr]
                xidx = CELL_KITE_XIDX[iptr]
                    
                RV_SUM_ = \
                    RV_SUM_ + xval * RV_DUAL[xidx]
        
            RV_CELL[cell] = (
                    RV_SUM_ / MESH_CELL_AREA[cell])
        
            PV_CELL[cell] = (
                (RV_CELL[cell] + FF_CELL[cell])
                )
                  
    return variables.rv_dual, \
           variables.pv_dual, \
           variables.rv_wide, \
           variables.pv_wide, \
           sqrt_r  ( pv_rms_ ), \
           variables.rv_cell, \
           variables.pv_cell, \
           variables.rv_edge, \
           variables.pv_edge
    
    
def _tend_hadv(mesh, mats, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
        const REALS_t gravity,
    np.ndarray[HTEND_t, ndim=1] hh_tend
              ):
    
#-- compute thickness advection: div uh
    
    cdef INDEX_t cell, iptr, xidx
    cdef REALS_t xval
  
    cdef LOCAL_t UH_TEND
    
    cdef REALS_t c0 = cnfg.sound_spd
    cdef REALS_t gamma = (c0>0.) * gravity / c0 ** 2

    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef HTEND_t *HH_TEND = &hh_tend[0]

    cdef INDEX_t *CELL_FLUX_XPTR = ptr_index_t(
             mats.cell_flux_sums.indptr)
    cdef INDEX_t *CELL_FLUX_XIDX = ptr_index_t(
             mats.cell_flux_sums.indices)
    cdef REALS_t *CELL_FLUX_XVAL = ptr_reals_t(
             mats.cell_flux_sums.data)

    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)

    with nogil, parallel(num_threads=numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- divergence of edge thickness fluxes D * uh
        #-- with optional weak-compressibility factors
            UH_TEND = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                
                UH_TEND = UH_TEND + xval * ( 
                    UU_EDGE[xidx] * 
                    HH_EDGE[xidx] * 
                   (ONE_ + HALF * gamma * HH_EDGE[xidx])
                )
        
            HH_TEND[cell]+= (
                UH_TEND / MESH_CELL_AREA[cell]
                        / (ONE_ + gamma * HH_CELL[cell])
                )
    
    return hh_tend
    

def _tend_qadv(mesh, mats, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] qq_edge,
        const REALS_t gravity,
    np.ndarray[QTEND_t, ndim=1] qq_tend
              ):
    
#-- compute scalar advection: div (uh)q
    
    cdef INDEX_t cell, iptr, xidx
    cdef REALS_t xval
  
    cdef LOCAL_t UQ_TEND
    
    cdef REALS_t c0 = cnfg.sound_spd
    cdef REALS_t gamma = (c0>0.) * gravity / c0 ** 2

    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *QQ_EDGE = &qq_edge[0]
    cdef QTEND_t *QQ_TEND = &qq_tend[0]

    cdef INDEX_t *CELL_FLUX_XPTR = ptr_index_t(
             mats.cell_flux_sums.indptr)
    cdef INDEX_t *CELL_FLUX_XIDX = ptr_index_t(
             mats.cell_flux_sums.indices)
    cdef REALS_t *CELL_FLUX_XVAL = ptr_reals_t(
             mats.cell_flux_sums.data)

    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)

    with nogil, parallel(num_threads=numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- divergence of edge thickness fluxes D * uh
        #-- with optional weak-compressibility factors
            UQ_TEND = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                
                UQ_TEND = UQ_TEND + xval * ( 
                    QQ_EDGE[xidx] * 
                    UU_EDGE[xidx] * 
                    HH_EDGE[xidx] * 
                   (ONE_ + HALF * gamma * HH_EDGE[xidx])
                )
        
            QQ_TEND[cell]+= (
                UQ_TEND / MESH_CELL_AREA[cell]
                        / (ONE_ + gamma * HH_CELL[cell])
                )
    
    return qq_tend

    
def _tend_uadv(mesh, mats, cnfg, 
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] hh_quad,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] pv_edge,
    np.ndarray[REALS_t, ndim=1] ke_cell,
    np.ndarray[FLT32_t, ndim=1] ff_dual,
    np.ndarray[FLT32_t, ndim=1] ff_edge,
    np.ndarray[FLT32_t, ndim=1] ff_cell,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- mometum advection: qhu^\perp + grad K
    
    cdef INDEX_t edge, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    
    cdef REALS_t K1_CELL, K2_CELL
    cdef LOCAL_t PV_MEAN, UV_FLUX, KE_GRAD
    
    cdef REALS_t HALF = 0.5
    cdef REALS_t ZERO = 0.0
    cdef REALS_t TWO_ = 2.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert
    cdef REALS_t do_advect = (not cnfg.no_advect)

    cdef REALS_t pv_weight = cnfg.pv_weight
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *PV_EDGE = &pv_edge[0]
    cdef REALS_t *KE_CELL = &ke_cell[0]
    cdef FLT32_t *FF_EDGE = &ff_edge[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]

    cdef np.ndarray[REALS_t] uh_flux = get_vec_e()
    cdef np.ndarray[REALS_t] fh_flux = get_vec_e()

    cdef REALS_t *UH_FLUX = &uh_flux[0]
    cdef REALS_t *FH_FLUX = &fh_flux[0]

    cdef INDEX_t *EDGE_FLUX_XPTR = ptr_index_t(
             mats.edge_flux_perp.indptr)
    cdef INDEX_t *EDGE_FLUX_XIDX = ptr_index_t(
             mats.edge_flux_perp.indices)
    cdef REALS_t *EDGE_FLUX_XVAL = ptr_reals_t(
             mats.edge_flux_perp.data)

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

        #-- split linear & nonlinear (curl(u) + f) / h
            PV_EDGE[edge]-= FF_EDGE[edge] * pv_weight
            PV_EDGE[edge]/= HH_QUAD[edge]

            UH_FLUX[edge] = UU_EDGE[edge] * HH_EDGE[edge]

            FH_FLUX[edge] = TWO_ * pv_weight * \
                            FF_EDGE[edge] / HH_QUAD[edge]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- energy neutral flux 1/2 * (W*qhu + q*W*hu)
            UV_FLUX = ZERO
            for iptr in range(EDGE_FLUX_XPTR[edge +0], 
                              EDGE_FLUX_XPTR[edge +1]):
                    
                xval = EDGE_FLUX_XVAL[iptr]
                xidx = EDGE_FLUX_XIDX[iptr]

                PV_MEAN = \
                    FH_FLUX[xidx] + \
                    PV_EDGE[edge] + PV_EDGE[xidx]

                UV_FLUX = UV_FLUX \
                    - xval * UH_FLUX[xidx] * PV_MEAN
        
        #-- gradient of kinetic energy G * 1/2 * |u|^2
            K1_CELL = KE_CELL[cel1]            
            K2_CELL = KE_CELL[cel2]

            KE_GRAD =(K2_CELL - K1_CELL) / MESH_EDGE_CLEN[edge]
                
            KE_GRAD = KE_GRAD * do_advect
        
            UU_TEND[edge]+= \
                MESH_EDGE_MASK[edge] \
                    * (KE_GRAD  + HALF * UV_FLUX)

    put_vec_e   (uh_flux)
    put_vec_e   (fh_flux)
        
    return uu_tend
    

def _tend_upgf(mesh, mats, cnfg, 
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
    np.ndarray[REALS_t, ndim=1] xi_self,
        const REALS_t gravity,
        const REALS_t hh_tiny,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- pressure gradient: grad g * (h + z_b)
    
    cdef INDEX_t edge, cel1, cel2
    cdef REALS_t xval
    
    cdef REALS_t HH_MIN_, H1_CELL, H2_CELL
    cdef FLT64_t ZB_MAX_, Z1_CELL, Z2_CELL
    cdef FLT64_t ZT_GRAD, ZF_GRAD

    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t pgf_limit = cnfg.pgf_limit

    cdef REALS_t sal_scale = cnfg.sal_scale
    cdef REALS_t sal_const = cnfg.sal_const
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *XI_SELF = &xi_self[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- limiter ensures a piecewise linear SSH
        #-- is +ve wrt. a piecewise constant topography
            H1_CELL = HH_CELL[cel1]
            H2_CELL = HH_CELL[cel2]
            HH_MIN_ = TWO_ * (
                H1_CELL * H2_CELL / (H1_CELL + H2_CELL)
                )

           #ZB_MAX_ = max (ZB_CELL[cel1] - HH_MIN_,
           #               ZB_CELL[cel2] - HH_MIN_)

        #-- surface pressure gradient g * G * (h + z_b)
            Z1_CELL = HH_CELL[cel1] + ZB_CELL[cel1]
           #Z1_CELL = max (ZB_MAX_, Z1_CELL)

            Z2_CELL = HH_CELL[cel2] + ZB_CELL[cel2]
           #Z2_CELL = max (ZB_MAX_, Z2_CELL)

            ZT_GRAD =(Z2_CELL - Z1_CELL) / MESH_EDGE_CLEN[edge]

        #-- attract.+loading gradient g * G * filt(z_t)
            Z1_CELL = XI_SELF[cel1]
           #Z1_CELL = max (ZB_MAX_, Z1_CELL)

            Z2_CELL = XI_SELF[cel2]
           #Z2_CELL = max (ZB_MAX_, Z2_CELL)

            ZF_GRAD =(Z2_CELL - Z1_CELL) / MESH_EDGE_CLEN[edge]

        #-- scalar, depth-weighted SAL: alpha * G * xi
            ZT_GRAD = ZT_GRAD - ZF_GRAD * (
                sal_const * min(ONE_, sqrt_r(HH_MIN_/sal_scale)
                ) )

            UU_TEND[edge]+= \
                MESH_EDGE_MASK[edge] * gravity * ZT_GRAD
    
    return uu_tend


def _tend_ugeo(mesh, mats, cnfg, 
        const REALS_t gravity,
        const REALS_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] Xi_prev,
    np.ndarray[REALS_t, ndim=1] Xi_next,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- forcing due to grad of ext. geo-potential
    
    cdef INDEX_t edge, cel1, cel2
    
    cdef LOCAL_t X1_CELL, X2_CELL, XI_GRAD
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t frc_blend = cnfg.frc_blend
    cdef REALS_t frc_start = cnfg.frc_start
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *XI_PREV = &Xi_prev[0]
    cdef REALS_t *XI_NEXT = &Xi_next[0]
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
                
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- gradient of external geo-potential: G * xi
            X1_CELL = (
                (ONE_ - frc_blend) * XI_PREV[cel1]
              + (ZERO + frc_blend) * XI_NEXT[cel1]
                      )
            X2_CELL = (
                (ONE_ - frc_blend) * XI_PREV[cel2]
              + (ZERO + frc_blend) * XI_NEXT[cel2]
                      )
                
            XI_GRAD =(X2_CELL - X1_CELL) / MESH_EDGE_CLEN[edge]

            UU_TEND[edge]+= \
                MESH_EDGE_MASK[edge] * XI_GRAD * frc_start
        
    return uu_tend


def _tend_utde(mesh, mats, cnfg,
        const REALS_t gravity,
        const REALS_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] Xi_tide,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- forcing due to grad of ext. geo-potential
    
    cdef INDEX_t edge, cel1, cel2
    
    cdef REALS_t X1_CELL, X2_CELL
    cdef LOCAL_t XI_GRAD
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t frc_start = cnfg.frc_start
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *XI_TIDE = &Xi_tide[0]
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
                
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- gradient of tide-SAL geo-potential: G * xi
            X1_CELL = XI_TIDE[cel1]
            X2_CELL = XI_TIDE[cel2]
                
            XI_GRAD =(X2_CELL - X1_CELL) / MESH_EDGE_CLEN[edge]

            UU_TEND[edge]-= \
                MESH_EDGE_MASK[edge] * XI_GRAD * frc_start

    return uu_tend

    
def _calc_perp(mesh, mats, cnfg, 
    np.ndarray[UDATA_t, ndim=1] uu_edge
              ):
    
#-- tangent reconstruction: F^\perp <== F
    
    cdef INDEX_t edge, iptr, xidx
    cdef REALS_t xval
    
    cdef LOCAL_t VV_SUM_

    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef UDATA_t *UU_EDGE = &uu_edge[0]

    cdef REALS_t[::1] vv_edge = variables.vv_edge
        
    cdef REALS_t *VV_EDGE = &vv_edge[0]

    cdef INDEX_t *EDGE_LSQR_XPTR = ptr_index_t(
             mats.edge_lsqr_perp.indptr)
    cdef INDEX_t *EDGE_LSQR_XIDX = ptr_index_t(
             mats.edge_lsqr_perp.indices)
    cdef REALS_t *EDGE_LSQR_XVAL = ptr_reals_t(
             mats.edge_lsqr_perp.data)

    cdef REALS_t *MESH_EDGE_PERP = ptr_reals_t(
                  mesh.edge.perp)

    with nogil, parallel(num_threads=numthread):
        
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
        #-- LSQR .^perp reconstruction: V = W_lsqr * U
            VV_SUM_ = ZERO
            for iptr in range(EDGE_LSQR_XPTR[edge +0], 
                              EDGE_LSQR_XPTR[edge +1]):
                    
                xval = EDGE_LSQR_XVAL[iptr]
                xidx = EDGE_LSQR_XIDX[iptr]

                VV_SUM_ = \
                    VV_SUM_ + xval * UU_EDGE[xidx]

            VV_EDGE[edge] = VV_SUM_

        #-- account for free / no-slip at any wall BCs
            VV_EDGE[edge]*= MESH_EDGE_PERP[edge]

    return variables.vv_edge


def _calc_umix(mesh, mats, cnfg, 
    np.ndarray[REALS_t, ndim=1] rv_dual,
    np.ndarray[REALS_t, ndim=1] rv_cell
              ):
          
#-- leith sub-grid model for turb. nu_2^u
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef REALS_t xval

    cdef LOCAL_t dN_EDGE, dP_EDGE
    
    cdef REALS_t ZERO = 0.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    # factor of 2 is from 2 * slen below
    cdef REALS_t leith_max = cnfg.leith_max
    cdef REALS_t leith_chi = \
            cnfg.leith_chi * (2.0 / np.pi) ** 3

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *RV_DUAL = &rv_dual[0]
    cdef REALS_t *RV_CELL = &rv_cell[0]
    
    cdef np.ndarray[REALS_t] nu_turb = variables.nu_turb
    
    cdef REALS_t *NU_TURB = &nu_turb[0]

    cdef np.ndarray[REALS_t] msh_nu2 = variables.msh_nu2
    
    cdef REALS_t *MSH_NU2 = &msh_nu2[0]
    
    cdef INDEX_t *GRAD_NORM_XPTR = ptr_index_t(
             mats.edge_grad_norm.indptr)
    cdef INDEX_t *GRAD_NORM_XIDX = ptr_index_t(
             mats.edge_grad_norm.indices)
    cdef REALS_t *GRAD_NORM_XVAL = ptr_reals_t(
             mats.edge_grad_norm.data)

    cdef INDEX_t *GRAD_PERP_XPTR = ptr_index_t(
             mats.edge_grad_perp.indptr)
    cdef INDEX_t *GRAD_PERP_XIDX = ptr_index_t(
             mats.edge_grad_perp.indices)
    cdef REALS_t *GRAD_PERP_XVAL = ptr_reals_t(
             mats.edge_grad_perp.data)

    # slen scaling as visc. acts on rot(u)
    cdef REALS_t *MESH_EDGE_SLEN = ptr_reals_t(
                  mesh.edge.slen)

    with nogil, parallel(num_threads=numthread):
  
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):        
        #-- dN_edge = edge_grad_norm * rv_cell
            dN_EDGE = ZERO
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                dN_EDGE = \
                    dN_EDGE + xval * RV_CELL[xidx]
                    
        #-- dP_edge = edge_grad_perp * rv_dual
            dP_EDGE = ZERO
            for iptr in range(GRAD_PERP_XPTR[edge +0],
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                    
                dP_EDGE = \
                    dP_EDGE + xval * RV_DUAL[xidx]
        
        #-- nu = chi * len ^ 3 * |grad curl u|
            NU_TURB[edge] = leith_chi * sqrt_r(
                            dN_EDGE * dN_EDGE + 
                            dP_EDGE * dP_EDGE )

            NU_TURB[edge]*= MESH_EDGE_SLEN[edge] * \
                            MESH_EDGE_SLEN[edge] * \
                            MESH_EDGE_SLEN[edge]

            NU_TURB[edge] = min(
                NU_TURB[edge], leith_max * MSH_NU2[edge]
                )
        
    return variables.nu_turb


def _calc_uwav(mesh, mats, cnfg,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const FLT32_t gravity,
        const FLT32_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] hh_edge,    
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
          
#-- Riemann-type "waves" dissipation nu_2^u
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    
    cdef REALS_t H1_CELL, H2_CELL, UU_WAVE
    cdef REALS_t SENSOR_ 
    cdef LOCAL_t C1_WAVE, C2_WAVE, CC_WAVE

    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t waves_max = cnfg.waves_max
    cdef REALS_t waves_chi = cnfg.waves_chi * HALF
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]

    cdef np.ndarray[REALS_t] nu_wave = variables.nu_wave

    cdef REALS_t *NU_WAVE = &nu_wave[0]

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    cdef np.ndarray[REALS_t] msh_fix = variables.msh_fix
    cdef np.ndarray[REALS_t] msh_nu2 = variables.msh_nu2

    cdef REALS_t *MSH_FIX = &msh_fix[0]
    cdef REALS_t *MSH_NU2 = &msh_nu2[0]

    # clen scaling as visc. acts on div(u)
    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)

    with nogil, parallel(num_threads=numthread):

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
            
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- diffusivity = 1/2 * w_chi * sensor * |u_wave|
            H1_CELL = max(hh_tiny,HH_CELL[cel1])
            H2_CELL = max(hh_tiny,HH_CELL[cel2])

            SENSOR_ = ONE_ + MSH_FIX[edge]  # topol. fix

        #-- central-upwind scheme: hrm. average of waves
           #UU_WAVE = fabs_r(UU_EDGE[edge])
            UU_WAVE = sqrt_r (
                UU_EDGE[edge] * UU_EDGE[edge] +
                VV_EDGE[edge] * VV_EDGE[edge] )
            
            C1_WAVE = UU_WAVE + sqrt_r (gravity * H1_CELL)
            C2_WAVE = UU_WAVE + sqrt_r (gravity * H2_CELL)

            CC_WAVE = TWO_ * (
                C1_WAVE * C2_WAVE / (C1_WAVE + C2_WAVE)
                )

            NU_WAVE[edge] = waves_chi * SENSOR_ * CC_WAVE
            NU_WAVE[edge]*= MESH_EDGE_CLEN[edge]
            NU_WAVE[edge] = min(
                NU_WAVE[edge], waves_max * MSH_NU2[edge]
                )

    return variables.nu_wave


def _calc_hmix(mesh, mats, cnfg,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const FLT32_t gravity,
        const FLT32_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] vv_edge
              ):
          
#-- Riemann-type "shock" diffusion nu_2^h
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    
    cdef LOCAL_t Z1_CELL, Z2_CELL, ZB_MAX_
    cdef REALS_t H1_CELL, H2_CELL, UU_WAVE
    cdef LOCAL_t OS_NUM_, OS_DEN_
    cdef REALS_t OS_PROD, SENSOR_ 
    cdef LOCAL_t C1_WAVE, C2_WAVE, CC_WAVE

    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t shock_chi = cnfg.shock_chi * HALF
    cdef REALS_t shock_cut = cnfg.shock_cut
    cdef REALS_t shock_max = cnfg.shock_max
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *VV_EDGE = &vv_edge[0]

    cdef np.ndarray[REALS_t] nu_shoc = variables.nu_shoc
    cdef np.ndarray[REALS_t] os_shoc = variables.os_shoc
    
    cdef REALS_t *NU_SHOC = &nu_shoc[0]
    cdef REALS_t *OS_SHOC = &os_shoc[0]

    cdef np.ndarray[REALS_t] zt_grad = get_vec_e()
        
    cdef REALS_t *ZT_GRAD = &zt_grad[0]

    cdef np.ndarray[REALS_t] msh_fix = variables.msh_fix
    cdef np.ndarray[REALS_t] msh_nu2 = variables.msh_nu2

    cdef REALS_t *MSH_FIX = &msh_fix[0]
    cdef REALS_t *MSH_NU2 = &msh_nu2[0]

    cdef INDEX_t *CELL_FLUX_XPTR = ptr_index_t(
             mats.cell_flux_sums.indptr)
    cdef INDEX_t *CELL_FLUX_XIDX = ptr_index_t(
             mats.cell_flux_sums.indices)
    cdef REALS_t *CELL_FLUX_XVAL = ptr_reals_t(
             mats.cell_flux_sums.data)

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):
  
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
            
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- don't allow gradients across partial layers
            ZB_MAX_ = max (ZB_CELL[cel1], ZB_CELL[cel2])

        #-- surface pressure gradient g * G * (h + z_b)
            H1_CELL = HH_CELL[cel1]
            Z1_CELL = HH_CELL[cel1] + ZB_CELL[cel1]
            Z1_CELL = max (ZB_MAX_, Z1_CELL)

            H2_CELL = HH_CELL[cel2]
            Z2_CELL = HH_CELL[cel2] + ZB_CELL[cel2]
            Z2_CELL = max (ZB_MAX_, Z2_CELL)

            ZT_GRAD[edge] = (
               Z2_CELL - Z1_CELL) / MESH_EDGE_CLEN[edge]

        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- build a Jameson-style shock sensor:
        #-- div grad(z_t) / div abs[grad(z_t)]
            OS_NUM_ = ZERO; OS_DEN_ = \
                min(HH_CELL[cell], shock_cut)
            for iptr in range(CELL_FLUX_XPTR[cell +0],
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]

                OS_PROD = xval * ZT_GRAD[xidx]

                OS_NUM_ = OS_NUM_ + OS_PROD
                OS_DEN_ = OS_DEN_ + OS_PROD * \
                                    OS_PROD
            
            OS_SHOC[cell] = (TWO_ * OS_NUM_ * 
                                    OS_NUM_ / OS_DEN_)

            #-- os^k/(os^k+1.) limit to [0, 1]
            OS_SHOC[cell]*= OS_SHOC[cell]
            OS_SHOC[cell]/= OS_SHOC[cell] + ONE_

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
            
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

        #-- diffusivity = 1/2 * h_chi * sensor * |u_wave|
            H1_CELL = max(hh_tiny,HH_CELL[cel1])
            H2_CELL = max(hh_tiny,HH_CELL[cel2])

            SENSOR_ = ONE_ + MSH_FIX[edge]  # topol. fix
        
            SENSOR_ = SENSOR_ * \
                max(OS_SHOC[cel1],OS_SHOC[cel2])

        #-- central-upwind scheme: hrm. average of waves
            UU_WAVE = sqrt_r (
                UU_EDGE[edge] * UU_EDGE[edge] + 
                VV_EDGE[edge] * VV_EDGE[edge] )

            C1_WAVE = UU_WAVE + sqrt_r (gravity * H1_CELL)
            C2_WAVE = UU_WAVE + sqrt_r (gravity * H2_CELL)

            CC_WAVE = TWO_ * (
                C1_WAVE * C2_WAVE / (C1_WAVE + C2_WAVE)
                )

            NU_SHOC[edge] = shock_chi * SENSOR_ * CC_WAVE
            NU_SHOC[edge]*= MESH_EDGE_CLEN[edge]
            NU_SHOC[edge] = min(
                NU_SHOC[edge], shock_max * MSH_NU2[edge]
                )
            
    put_vec_e  (zt_grad)

    return variables.nu_shoc
        

def _tend_umix(mesh, mats, cnfg,
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] hh_quad,
    np.ndarray[REALS_t, ndim=1] hh_dual,    
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] nu_turb,
    np.ndarray[REALS_t, ndim=1] nu_wave,
    np.ndarray[REALS_t, ndim=1] nu_thin,
        const REALS_t hh_tiny,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- viscosity/dissipation: nu_k^u * del^k
    
    cdef INDEX_t vert, edge, cell, iptr, xidx
    cdef REALS_t xval

    cdef LOCAL_t DU_SUM_, RV_SUM_, DH_SUM_, RH_SUM_
    cdef LOCAL_t DK_SUM_, VK_SUM_
    cdef LOCAL_t UU_VISC
    cdef REALS_t HH_VAL_    

    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    cdef REALS_t FOUR = 4.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t do_dissip = "nu_diss" in cnfg.stat_vars

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef REALS_t *HH_DUAL = &hh_dual[0]
    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *NU_TURB = &nu_turb[0]
    cdef REALS_t *NU_WAVE = &nu_wave[0]
    cdef REALS_t *NU_THIN = &nu_thin[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]

    cdef np.ndarray[REALS_t] ke_diss = variables.ke_diss

    cdef REALS_t *KE_DISS = &ke_diss[0]

    cdef np.ndarray[REALS_t] v2_visc = variables.visc_u2
    cdef np.ndarray[REALS_t] v4_visc = variables.visc_u4

    cdef REALS_t *V2_VISC = &v2_visc[0]
    cdef REALS_t *V4_VISC = &v4_visc[0]

    cdef np.ndarray[REALS_t] vk_edge = get_vec_e()
    
    cdef REALS_t *VK_EDGE = &vk_edge[0]
    
    cdef np.ndarray[REALS_t] rv_dual = get_vec_v()

    cdef REALS_t *RV_DUAL = &rv_dual[0]

    cdef np.ndarray[REALS_t] du_cell = get_vec_c()
    cdef np.ndarray[REALS_t] uh_cell = get_vec_c()
    
    cdef REALS_t *DU_CELL = &du_cell[0]
    cdef REALS_t *UH_CELL = &uh_cell[0]
    
    cdef INDEX_t *CELL_FLUX_XPTR = ptr_index_t(
             mats.cell_flux_sums.indptr)
    cdef INDEX_t *CELL_FLUX_XIDX = ptr_index_t(
             mats.cell_flux_sums.indices)
    cdef REALS_t *CELL_FLUX_XVAL = ptr_reals_t(
             mats.cell_flux_sums.data)

    cdef INDEX_t *DUAL_CURL_XPTR = ptr_index_t(
             mats.dual_curl_sums.indptr)
    cdef INDEX_t *DUAL_CURL_XIDX = ptr_index_t(
             mats.dual_curl_sums.indices)
    cdef REALS_t *DUAL_CURL_XVAL = ptr_reals_t(
             mats.dual_curl_sums.data)

    cdef INDEX_t *GRAD_NORM_XPTR = ptr_index_t(
             mats.edge_grad_norm.indptr)
    cdef INDEX_t *GRAD_NORM_XIDX = ptr_index_t(
             mats.edge_grad_norm.indices)
    cdef REALS_t *GRAD_NORM_XVAL = ptr_reals_t(
             mats.edge_grad_norm.data)

    cdef INDEX_t *GRAD_PERP_XPTR = ptr_index_t(
             mats.edge_grad_perp.indptr)
    cdef INDEX_t *GRAD_PERP_XIDX = ptr_index_t(
             mats.edge_grad_perp.indices)
    cdef REALS_t *GRAD_PERP_XVAL = ptr_reals_t(
             mats.edge_grad_perp.data)

    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)
    cdef REALS_t *MESH_EDGE_AREA = ptr_reals_t(
                  mesh.edge.area)
    cdef REALS_t *MESH_DUAL_AREA = ptr_reals_t(
                  mesh.vert.area)

    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)
    cdef REALS_t *MESH_VERT_SLIP = ptr_reals_t(
                  mesh.vert.slip)

    with nogil, parallel(num_threads=numthread):
        
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- compute div(u.n)
            DU_SUM_ = ZERO; DH_SUM_ = ZERO
            HH_VAL_ = HH_CELL[cell]
            HH_VAL_ = max(HH_VAL_, hh_tiny)  # wet-dry
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                
                DU_SUM_ = \
                    DU_SUM_ + xval * UU_EDGE[xidx]
    
                DH_SUM_ = \
                    DH_SUM_ + xval * UU_EDGE[xidx] \
                                   * HH_EDGE[xidx]
            
            DU_CELL[cell] = (
                DU_SUM_ / MESH_CELL_AREA[cell]
                        )

            UH_CELL[cell] = (
                DH_SUM_ / 
                HH_VAL_ / MESH_CELL_AREA[cell]
                        )
            
        for vert in prange(0, NVRT, schedule="static", 
                chunksize=chunkvert):
        #-- compute rot(u.n)
            RV_SUM_ = ZERO
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_SUM_ = \
                    RV_SUM_ + xval * UU_EDGE[xidx]

            RV_DUAL[vert] = (
                     ONE_ - MESH_VERT_SLIP[vert]
                        ) * RV_SUM_
            RV_DUAL[vert]/= MESH_DUAL_AREA[vert]
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
        #-- V^2 = vk * grad(div(h*u.n) *h^-1) - 
        #--       vk * grad(h*rot(u.n))*h^-1
            DU_SUM_ = ZERO; DH_SUM_ = ZERO
            RV_SUM_ = ZERO; RH_SUM_ = ZERO
            HH_VAL_ = HH_QUAD[edge]
            HH_VAL_ = max(HH_VAL_, hh_tiny)  # wet-dry
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                DU_SUM_ = \
                    DU_SUM_ + xval * DU_CELL[xidx]

                DH_SUM_ = \
                    DH_SUM_ + xval * UH_CELL[xidx]

            for iptr in range(GRAD_PERP_XPTR[edge +0], 
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                
                RV_SUM_ = \
                    RV_SUM_ - xval * RV_DUAL[xidx]
    
                RH_SUM_ = \
                    RH_SUM_ - xval * RV_DUAL[xidx] \
                                   * HH_DUAL[xidx]

            DK_SUM_ = (        # only div^2(u)
                DH_SUM_
                      ) * MESH_EDGE_MASK[edge]

            VK_SUM_ = (        # thick. weight
                DH_SUM_ + 
                RH_SUM_ / HH_VAL_
                      ) * MESH_EDGE_MASK[edge]

            VK_EDGE[edge] = (  # only del^2(u)
                DU_SUM_ + RV_SUM_
                      ) * MESH_EDGE_MASK[edge]

            UU_VISC = (
        #-- constant viscosities
              + V2_VISC[edge] * VK_SUM_
        #-- dissipation adj. to wet-dry areas
              + NU_THIN[edge] * VK_SUM_
        #-- turb. dissipative fluxes
              + NU_TURB[edge] * VK_SUM_
        #-- div u dissipative fluxes
              + NU_WAVE[edge] * DK_SUM_
                )

            UU_TEND[edge]-= UU_VISC

        #-- visc dissipation: u * h * nu * del^k * u
            if (do_dissip): \
            KE_DISS[edge]+= fabs_r(
                UU_VISC * 
                UU_EDGE [edge] * HH_EDGE[edge])

        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- compute div(V^2)
            DU_SUM_ = ZERO; DH_SUM_ = ZERO
            HH_VAL_ = HH_CELL[cell]
            HH_VAL_ = max(HH_VAL_, hh_tiny)  # wet-dry
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                DH_SUM_ = \
                    DH_SUM_ + xval * VK_EDGE[xidx] \
                                   * HH_EDGE[xidx]
         
            UH_CELL[cell] = (
                DH_SUM_ / 
                HH_VAL_ / MESH_CELL_AREA[cell]
                        )

        for vert in prange(0, NVRT, schedule="static", 
                chunksize=chunkvert):
        #-- compute rot(V^2)
            RV_SUM_ = ZERO
            HH_VAL_ = HH_DUAL[vert]
            for iptr in range(DUAL_CURL_XPTR[vert +0], 
                              DUAL_CURL_XPTR[vert +1]):
                    
                xval = DUAL_CURL_XVAL[iptr]
                xidx = DUAL_CURL_XIDX[iptr]
                    
                RV_SUM_ = \
                    RV_SUM_ + xval * VK_EDGE[xidx]
         
            RV_DUAL[vert] = (  # h*rot(V^2)
                HH_VAL_ * 
                RV_SUM_ / MESH_DUAL_AREA[vert]
                        )
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
        #-- V^4 = vk * grad(div(h*V^2) *h^-1) - 
        #--       vk * grad(h*rot(V^2))*h^-1
            DH_SUM_ = ZERO; RH_SUM_ = ZERO
            HH_VAL_ = HH_QUAD[edge]
            HH_VAL_ = max(HH_VAL_, hh_tiny)  # wet-dry
            for iptr in range(GRAD_NORM_XPTR[edge +0], 
                              GRAD_NORM_XPTR[edge +1]):
                    
                xval = GRAD_NORM_XVAL[iptr]
                xidx = GRAD_NORM_XIDX[iptr]
                    
                DH_SUM_ = \
                    DH_SUM_ + xval * UH_CELL[xidx]
    
            for iptr in range(GRAD_PERP_XPTR[edge +0], 
                              GRAD_PERP_XPTR[edge +1]):
                    
                xval = GRAD_PERP_XVAL[iptr]
                xidx = GRAD_PERP_XIDX[iptr]
                    
                RH_SUM_ = \
                    RH_SUM_ - xval * RV_DUAL[xidx]
            
        #   DK_SUM_ = (        # only div^4(u)
        #       DH_SUM_
        #             ) * MESH_EDGE_MASK[edge]
            
            VK_SUM_ = (        # thick. weight
                DH_SUM_ + 
                RH_SUM_ / HH_VAL_
                      ) * MESH_EDGE_MASK[edge]

            UU_VISC = (
        #-- constant viscosities
              - V4_VISC[edge] * VK_SUM_
        #-- turb. dissipative fluxes
        #-- 32 leads to del^2 fraction of approx. 3%
              - NU_TURB[edge] * VK_SUM_
                * MESH_EDGE_AREA[edge]#/ FOUR / TWO_
                                       * FOUR
        #-- div u dissipative fluxes
        #-- 1/8 accounts for nu_4 scaling per MITgcm
             #- NU_WAVE[edge] * DK_SUM_
             #  * MESH_EDGE_AREA[edge] / FOUR / TWO_
                )

            UU_TEND[edge]-= UU_VISC

        #-- visc dissipation: u * h * nu * del^k * u
            if (do_dissip): \
            KE_DISS[edge]+= fabs_r(
                UU_VISC * 
                UU_EDGE [edge] * HH_EDGE[edge])

    put_vec_e  (vk_edge)
    
    put_vec_c  (uh_cell)
    put_vec_c  (du_cell)

    put_vec_v  (rv_dual)
                     
    return uu_tend
    
    
def _tend_hmix(mesh, mats, cnfg, 
    np.ndarray[HDATA_t, ndim=1] hh_cell,
    np.ndarray[FLT32_t, ndim=1] zb_cell,
        const REALS_t gravity,
    np.ndarray[REALS_t, ndim=1] nu_shoc,
        const REALS_t hh_tiny,
    np.ndarray[HTEND_t, ndim=1] hh_tend
              ):
    
#-- thickness dissipation: nu_k^h * del^k
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef INDEX_t cel1, cel2
    cdef REALS_t xval
    
    cdef LOCAL_t OK_SUM_
    cdef LOCAL_t DH_SUM_, GZ_EDGE
    cdef LOCAL_t V2_SUM_, V4_SUM_
    cdef FLT64_t Z1_CELL, Z2_CELL, ZB_MAX_
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t HALF = 0.5
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef HDATA_t *HH_CELL = &hh_cell[0]
    cdef FLT32_t *ZB_CELL = &zb_cell[0]
    cdef REALS_t *NU_SHOC = &nu_shoc[0]
    cdef HTEND_t *HH_TEND = &hh_tend[0]
   
    cdef np.ndarray[REALS_t] v2_diff = variables.diff_h2
    cdef np.ndarray[REALS_t] v4_diff = variables.diff_h4

    cdef REALS_t *V2_DIFF = &v2_diff[0]
    cdef REALS_t *V4_DIFF = &v4_diff[0]
    
    cdef np.ndarray[REALS_t] v2_cell = get_vec_c()
    cdef np.ndarray[REALS_t] v4_cell = get_vec_c()
        
    cdef REALS_t *V2_CELL = &v2_cell[0]
    cdef REALS_t *V4_CELL = &v4_cell[0]
    
    cdef np.ndarray[REALS_t] h2_edge = get_vec_e()
    cdef np.ndarray[REALS_t] h4_edge = get_vec_e()
    
    cdef REALS_t *H2_EDGE = &h2_edge[0]
    cdef REALS_t *H4_EDGE = &h4_edge[0]

    cdef np.ndarray[REALS_t] ok_edge = get_vec_e()
    cdef np.ndarray[REALS_t] ok_cell = get_vec_c()
    
    cdef REALS_t *OK_EDGE = &ok_edge[0]
    cdef REALS_t *OK_CELL = &ok_cell[0]

    cdef INDEX_t *CELL_FLUX_XPTR = ptr_index_t(
             mats.cell_flux_sums.indptr)
    cdef INDEX_t *CELL_FLUX_XIDX = ptr_index_t(
             mats.cell_flux_sums.indices)
    cdef REALS_t *CELL_FLUX_XVAL = ptr_reals_t(
             mats.cell_flux_sums.data)
    
    cdef REALS_t *MESH_CELL_AREA = ptr_reals_t(
                  mesh.cell.area)

    cdef REALS_t *MESH_EDGE_CLEN = ptr_reals_t(
                  mesh.edge.clen)
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell

    with nogil, parallel(num_threads=numthread):
        
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
                
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1
                
        #-- don't allow gradients across partial layers
            ZB_MAX_ = max(ZB_CELL[cel1], ZB_CELL[cel2])

        #-- surface pressure gradient g * G * (h + z_b)
            Z1_CELL = HH_CELL[cel1] + ZB_CELL[cel1]
            Z1_CELL = max(ZB_MAX_, Z1_CELL)

            Z2_CELL = HH_CELL[cel2] + ZB_CELL[cel2]
            Z2_CELL = max(ZB_MAX_, Z2_CELL)

            GZ_EDGE = Z2_CELL - Z1_CELL
            GZ_EDGE = GZ_EDGE / MESH_EDGE_CLEN[edge]
            
        #-- don't add D^2 diffusion into masked layers
            OK_EDGE[edge] = MESH_EDGE_MASK[edge]
            OK_EDGE[edge]*= max (
                Z1_CELL, Z2_CELL)>ZB_MAX_ + hh_tiny
            
            H2_EDGE[edge] = OK_EDGE[edge] * GZ_EDGE
            
        #-- don't add D^4 diffusion into masked layers
            OK_EDGE[edge]*= min (
                Z1_CELL, Z2_CELL)>ZB_MAX_ + hh_tiny

            H4_EDGE[edge] = OK_EDGE[edge] * GZ_EDGE

        #-- edge-centred diffusivity, for conservation
            H4_EDGE[edge]*= V4_DIFF[edge] * gravity

            H2_EDGE[edge]*=(V2_DIFF[edge] * gravity +
                            NU_SHOC[edge] )
            
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- compute div(nu * grad(*))
            V2_SUM_ = ZERO; V4_SUM_ = ZERO
            OK_SUM_ = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                V2_SUM_ = \
                    V2_SUM_ + xval * H2_EDGE[xidx]

                V4_SUM_ = \
                    V4_SUM_ + xval * H4_EDGE[xidx]
                
                # -ve if any edge limits
                OK_SUM_ = \
                    OK_SUM_ - ONE_ + OK_EDGE[xidx]
         
            OK_CELL[cell] = OK_SUM_
            V2_CELL[cell] = (
                V2_SUM_ / MESH_CELL_AREA[cell]
                )

            V4_CELL[cell] = (
                V4_SUM_ / MESH_CELL_AREA[cell]
                )
            
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
                
            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1
                
        #-- pressure gradient tend.: g * G * (hh + zb)
            Z1_CELL = V4_CELL[cel1]
            Z2_CELL = V4_CELL[cel2]
            
            GZ_EDGE = Z2_CELL - Z1_CELL
            GZ_EDGE = GZ_EDGE / MESH_EDGE_CLEN[edge]
            
            OK_SUM_ = OK_EDGE[edge] * \
                (OK_CELL[cel1] >= ZERO) * \
                (OK_CELL[cel2] >= ZERO) 
                
            H4_EDGE[edge] = GZ_EDGE * OK_SUM_
                
        #-- flux limiter: don't allow up-gradient flux
        #-- see M. Xue (2000): MWR.
            H4_EDGE[edge]*= (
                H4_EDGE[edge]*H2_EDGE[edge] > ZERO)

        #-- edge-centred diffusivity, for conservation
            H4_EDGE[edge]*= V4_DIFF[edge] * gravity
            
        for cell in prange(0, NCEL, schedule="static", 
                chunksize=chunkcell):
        #-- compute div(nu * grad(*))
            V4_SUM_ = ZERO
            for iptr in range(CELL_FLUX_XPTR[cell +0], 
                              CELL_FLUX_XPTR[cell +1]):
                    
                xval = CELL_FLUX_XVAL[iptr]
                xidx = CELL_FLUX_XIDX[iptr]
                    
                V4_SUM_ = \
                    V4_SUM_ + xval * H4_EDGE[xidx]
         
            V4_CELL[cell] = (
                V4_SUM_ / MESH_CELL_AREA[cell]
                )
            
            HH_TEND[cell]-= (
               + V2_CELL[cell] - V4_CELL[cell]
                )
    
    put_vec_c  (ok_cell)
    put_vec_e  (ok_edge)

    put_vec_c  (v4_cell)
    put_vec_c  (v2_cell)

    put_vec_e  (h2_edge)
    put_vec_e  (h4_edge)
    
    return hh_tend
    
    
def _tend_utau(mesh, mats, cnfg, 
        const REALS_t hh_tiny,
    np.ndarray[REALS_t, ndim=1] Tu_prev,
    np.ndarray[REALS_t, ndim=1] Tu_next,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[UTEND_t, ndim=1] uu_tend
              ):
    
#-- forcing due to external stresses: tau / h
    
    cdef INDEX_t edge, cell, iptr, xidx
    cdef REALS_t xval

    cdef LOCAL_t TU_EDGE
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t frc_blend = cnfg.frc_blend
    cdef REALS_t frc_start = cnfg.frc_start
    
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *TU_PREV = &Tu_prev[0]
    cdef REALS_t *TU_NEXT = &Tu_next[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]
    
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    with nogil, parallel(num_threads=numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):
                
            TU_EDGE = ( frc_start  * (
                (ONE_ - frc_blend) * TU_PREV[edge]
              + (ZERO + frc_blend) * TU_NEXT[edge]
                      ) )
                
        #-- limit applied stresses in quasi-dry layers
            UU_TEND[edge]-= MESH_EDGE_MASK[edge] * (
                TU_EDGE / max (hh_tiny, HH_EDGE[edge])
                )
        
    return uu_tend


def _tend_uflt(mesh, mats, cnfg,
    np.ndarray[UDATA_t, ndim=1] uu_edge,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] uu_filt,
    np.ndarray[UTEND_t, ndim=1] uu_tend
        ):

#-- btr-bcl dissipation law: cd_flt * |u_flt| * sgn(u)

    cdef REALS_t ZERO = 0.0
    cdef REALS_t ONE_ = 1.0

    cdef INDEX_t edge

    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t fltlaw_cd = cnfg.fltlaw_cd
    cdef REALS_t fltlaw_h0 = cnfg.fltlaw_h0
    cdef LOCAL_t fltlaw_c1
    
    cdef LOCAL_t relax_val
    cdef LOCAL_t relax_tau = cnfg.time_step \
                           / cnfg.fltlaw_t0

    cdef INDEX_t do_dissip = "cd_diss" in cnfg.stat_vars

    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size

    cdef np.ndarray[REALS_t] ke_diss = variables.ke_diss

    cdef UDATA_t *UU_EDGE = &uu_edge[0]
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *UU_FILT = &uu_filt[0]
    cdef UTEND_t *UU_TEND = &uu_tend[0]
    cdef REALS_t *KE_DISS = &ke_diss[0]

    with nogil, parallel(num_threads=numthread):
    
        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

            relax_val = (relax_tau *
                sqrt_r(sqrt_r(max(ONE_,HH_EDGE[edge])
                ) ) )

            UU_FILT[edge] = \
                relax_val * UU_EDGE[edge] + \
                    (ONE_ - relax_val)*UU_FILT[edge]

            fltlaw_c1 = fltlaw_cd * min(
                ONE_, sqrt_r(max(ZERO, 
                    HH_EDGE[edge] / fltlaw_h0 - ONE_)
                ) )

            """
            if (UU_EDGE[edge] * UU_FILT[edge]>=ZERO):
                UU_TEND[edge]+=
                    fltlaw_c1 * UU_FILT[edge]
            """

            if (UU_EDGE[edge] > ZERO):
                UU_TEND[edge]+= \
                    fltlaw_c1 * fabs_r(UU_FILT[edge])
            
            if (UU_EDGE[edge] < ZERO):
                UU_TEND[edge]-= \
                    fltlaw_c1 * fabs_r(UU_FILT[edge])

            # drag dissipation: u * h * cd * u
            if (do_dissip): \
            KE_DISS[edge]+= fltlaw_c1 \
                * UU_FILT[edge] \
                * UU_FILT[edge] * HH_EDGE[edge]
            
    return uu_filt, uu_tend
    
    
def _calc_drag(mesh, mats, cnfg, 
        const REALS_t hh_tiny,
        const REALS_t ke_tiny,
        const REALS_t gravity,
    np.ndarray[REALS_t, ndim=1] hh_edge,
    np.ndarray[REALS_t, ndim=1] hh_quad,
    np.ndarray[REALS_t, ndim=1] ke_cell,
    np.ndarray[REALS_t, ndim=1] ke_edge,
    np.ndarray[FLT32_t, ndim=1] dz_drag,
    np.ndarray[FLT32_t, ndim=1] c1_edge,
    np.ndarray[FLT32_t, ndim=1] c2_edge,
    np.ndarray[FLT32_t, ndim=1] z0_edge,
    np.ndarray[FLT32_t, ndim=1] n0_edge
              ):
    
#-- cd = cd_lin + (cd_sqr + cd_log + cd_man) * |u| / h
    
    cdef REALS_t VONK = 0.4  # von karman
    
    cdef REALS_t ZERO = 0.0
    cdef REALS_t HALF = 0.5
    cdef REALS_t ONE_ = 1.0
    cdef REALS_t TWO_ = 2.0
    cdef REALS_t FOUR = 4.0
    cdef REALS_t SIX_ = 6.0
    
    cdef INDEX_t edge
    cdef INDEX_t cel1, cel2
    
    cdef INDEX_t numthread = cnfg.numthread
    cdef INDEX_t chunksize = cnfg.chunksize
    cdef INDEX_t chunkcell = cnfg.chunkcell
    cdef INDEX_t chunkedge = cnfg.chunkedge
    cdef INDEX_t chunkvert = cnfg.chunkvert

    cdef REALS_t hu_edge
    cdef LOCAL_t ku_edge, ku_tiny, ku_filt, cd_temp
    
    cdef REALS_t loglaw_z0 = cnfg.loglaw_z0
    cdef REALS_t loglaw_hi = cnfg.loglaw_hi
    cdef REALS_t loglaw_lo = cnfg.loglaw_lo
    
    cdef REALS_t manlaw_n2 = cnfg.manlaw_n0
    cdef REALS_t manlaw_hi = cnfg.manlaw_hi
    cdef REALS_t manlaw_lo = cnfg.manlaw_lo

    cdef REALS_t linlaw_cd = cnfg.linlaw_cd
    cdef REALS_t sqrlaw_cd = cnfg.sqrlaw_cd

    # careful not to load arrays to cache if unnecessary
    cdef INDEX_t do_loglaw = loglaw_z0 > 0.
    cdef INDEX_t do_manlaw = manlaw_n2 > 0.
    cdef INDEX_t do_linlaw = linlaw_cd > 0.
    cdef INDEX_t do_sqrlaw = sqrlaw_cd > 0.

    cdef INDEX_t do_dissip = "cd_diss" in cnfg.stat_vars
   
    cdef INDEX_t NVRT = mesh.vert.size
    cdef INDEX_t NEDG = mesh.edge.size
    cdef INDEX_t NCEL = mesh.cell.size
    
    cdef REALS_t *HH_EDGE = &hh_edge[0]
    cdef REALS_t *HH_QUAD = &hh_quad[0]
    cdef REALS_t *KE_CELL = &ke_cell[0]
    cdef REALS_t *KE_EDGE = &ke_edge[0]
    
    cdef FLT32_t *DZ_DRAG = &dz_drag[0]

    cdef FLT32_t *C1_EDGE = &c1_edge[0]
    cdef FLT32_t *C2_EDGE = &c2_edge[0]    
    cdef FLT32_t *Z0_EDGE = &z0_edge[0]
    cdef FLT32_t *N0_EDGE = &n0_edge[0]
    
    cdef np.ndarray[REALS_t] cd_edge = variables.cd_edge
    cdef np.ndarray[REALS_t] ke_diss = variables.ke_diss

    cdef REALS_t *CD_EDGE = &cd_edge[0]
    cdef REALS_t *KE_DISS = &ke_diss[0]
    
    cdef INDEX_t[:, ::1] mesh_edge_cell = mesh.edge.cell
    
    cdef REALS_t *MESH_EDGE_MASK = ptr_reals_t(
                  mesh.edge.fmsk)

    with nogil, parallel(num_threads=numthread):
    
        ku_tiny = sqrt_r(ke_tiny * TWO_)

        for edge in prange(0, NEDG, schedule="static", 
                chunksize=chunkedge):

            cel1 = mesh_edge_cell[edge, 0] - 1
            cel2 = mesh_edge_cell[edge, 1] - 1
            
            if (cel1 < 0): cel1 = cel2
            if (cel2 < 0): cel2 = cel1

            ku_edge = (  # simpson's
                sqrt_r(TWO_ * KE_CELL[cel1]) + 
                sqrt_r(TWO_ * KE_CELL[cel2]) +
                sqrt_r(TWO_ * KE_EDGE[edge]) * FOUR
                ) / SIX_

            hu_edge = min (HH_EDGE[edge], 
                           HH_QUAD[edge])
            hu_edge = max (hh_tiny, hu_edge)
            
            CD_EDGE[edge] = ZERO
       
            if (do_loglaw): loglaw_z0 = Z0_EDGE [edge]
            if (loglaw_z0 > ZERO):
                # NB. log(1+z/z0) "fix" to loglaw
                cd_temp = (VONK / log_r (
                    ONE_ + HALF * hu_edge / loglaw_z0)
                    )
                    
                cd_temp = cd_temp * cd_temp
                    
                cd_temp = min(cd_temp, loglaw_hi)
                cd_temp = max(cd_temp, loglaw_lo)
                
                CD_EDGE[edge]+= cd_temp
            
            if (do_manlaw): manlaw_n2 = N0_EDGE [edge]
            if (manlaw_n2 > ZERO):
                cd_temp = ( 
                gravity * manlaw_n2 *
                          manlaw_n2 / cbrt_r (hu_edge)
                    )
                    
                cd_temp = min(cd_temp, manlaw_hi)
                cd_temp = max(cd_temp, manlaw_lo)

                CD_EDGE[edge]+= cd_temp
            
            if (do_sqrlaw): \
            CD_EDGE[edge]+= C2_EDGE[edge]
            
            # zb-cell vs zb-drag sub-grid effects
            hu_edge = hu_edge- DZ_DRAG[edge]
            hu_edge = max (hh_tiny, hu_edge)

            CD_EDGE[edge]*= (ku_tiny + ku_edge) / hu_edge
            
            if (do_linlaw): \
            CD_EDGE[edge]+= C1_EDGE[edge]

            # drag dissipation: u * h * cd * u
            if (do_dissip): \
            KE_DISS[edge]+= CD_EDGE[edge] \
                   * MESH_EDGE_MASK[edge] \
                * (ku_edge * ku_edge * hu_edge)
    
    return variables.cd_edge


cdef INDEX_t* ptr_index_t(INDEX_t[::1] buffer):
    cdef INDEX_t *BUFFER = &buffer[+0]
    return BUFFER


cdef REALS_t* ptr_reals_t(REALS_t[::1] buffer):
    cdef REALS_t *BUFFER = &buffer[+0]
    return BUFFER

