
import numpy as np

""" Util. to setup memory pools for global & scratch arrays.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t
from _fp import udata_t, hdata_t, qdata_t
from _fp import utend_t, htend_t, qtend_t

class base: pass
variables = base()

ALIGNMENT = 64  # byte alignment of arrays

VERT_SIZE = 4   # number of vec's in pools
EDGE_SIZE = 4
CELL_SIZE = 4
vert_pool = []
edge_pool = []
cell_pool = []

def _aligned(shape, align, dtype):
#-- aligned alloc. for raw arrays
    dtype = np.dtype(dtype)
    nbyte = np.prod(shape) * dtype.itemsize
    array = np.empty(nbyte + align, dtype=np.uint8)
    start = align - (array.ctypes.data % align)
    return array[start:start + nbyte].view(dtype).reshape(shape)

def _allocate(cnfg, size, kind):
#-- parallel alloc. & fill, for NUMA residence
    return set_x_vec(
        cnfg, _aligned(size, align=ALIGNMENT, dtype=kind), +0.0)

def init_pool(cnfg, mesh):
    for i in range(VERT_SIZE):
        vert_pool.append(
            _allocate(cnfg, mesh.vert.size, kind=reals_t))
            
    for i in range(EDGE_SIZE):
        edge_pool.append(
            _allocate(cnfg, mesh.edge.size, kind=reals_t))
            
    for i in range(CELL_SIZE):
        cell_pool.append(
            _allocate(cnfg, mesh.cell.size, kind=reals_t))
    
    variables.is_zero = \
            _allocate(cnfg, mesh._max_size, kind=reals_t)

    variables.hh_cell = \
            _allocate(cnfg, mesh.cell.size, kind=hdata_t)
    variables.uu_edge = \
            _allocate(cnfg, mesh.edge.size, kind=udata_t)
    variables.qq_cell = \
            _allocate(cnfg, mesh.cell.size, kind=qdata_t)

    variables.ff_cell = \
            _allocate(cnfg, mesh.cell.size, kind=flt32_t)
    variables.ff_edge = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)
    variables.ff_vert = \
            _allocate(cnfg, mesh.vert.size, kind=flt32_t)

    variables.zb_cell = \
            _allocate(cnfg, mesh.cell.size, kind=flt32_t)

    variables.hh_min_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.uu_min_ = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.qq_min_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.hh_max_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.uu_max_ = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.qq_max_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)

    variables.zt_rms_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.ke_ave_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.ke_rms_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.ke_max_ = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.dk_ave_ = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.dk_rms_ = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.dk_max_ = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.hh_tend = \
            _allocate(cnfg, mesh.cell.size, kind=htend_t)
    variables.uu_tend = \
            _allocate(cnfg, mesh.edge.size, kind=utend_t)
    variables.qh_tend = \
            _allocate(cnfg, mesh.cell.size, kind=qtend_t)

    variables.h0_tend = \
            _allocate(cnfg, mesh.cell.size, kind=htend_t)
    variables.u0_tend = \
            _allocate(cnfg, mesh.edge.size, kind=utend_t)
    variables.s0_tend = \
            _allocate(cnfg, mesh.edge.size, kind=utend_t)

    variables.hb_cell = \
            _allocate(cnfg, mesh.cell.size, kind=hdata_t)

    variables.h1_cell = \
            _allocate(cnfg, mesh.cell.size, kind=hdata_t)
    variables.h2_cell = \
            _allocate(cnfg, mesh.cell.size, kind=hdata_t)
    variables.h3_cell = \
            _allocate(cnfg, mesh.cell.size, kind=hdata_t)
    
    variables.uk_edge = \
            _allocate(cnfg, mesh.edge.size, kind=udata_t)
    
    variables.hh_dual = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)
            
    variables.rv_dual = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)
    variables.pv_dual = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)

    variables.rv_wide = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)
    variables.pv_wide = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)
            
    variables.vv_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
            
    variables.hh_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)  
    variables.hh_quad = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    
    variables.rv_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)               
    variables.pv_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.hh_bias = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.pv_bias = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.ke_bias = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.msh_fix = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.msh_nu2 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.msh_nu4 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.visc_u2 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.visc_u4 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.diff_h2 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.diff_h4 = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.nu_turb = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.nu_wave = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.nu_shoc = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.nu_thin = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.os_wave = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.os_shoc = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
        
    variables.cd_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.cd_save = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.dz_drag = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)
    variables.c1_edge = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)
    variables.c2_edge = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)
    variables.z0_edge = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)
    variables.n0_edge = \
            _allocate(cnfg, mesh.edge.size, kind=flt32_t)

    variables.ke_diss = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
            
    variables.rv_cell = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.pv_cell = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)

    variables.ke_edge = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)
    variables.ke_dual = \
            _allocate(cnfg, mesh.vert.size, kind=reals_t)
    variables.ke_cell = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)

    variables.uu_filt = \
            _allocate(cnfg, mesh.edge.size, kind=reals_t)

    variables.Xi_tide = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)
    variables.Xi_self = \
            _allocate(cnfg, mesh.cell.size, kind=reals_t)

def get_vec_v():
    if (len(vert_pool) > 0):
        return vert_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: verts")
        
def put_vec_v(vec): vert_pool.append(vec)

def get_vec_e():
    if (len(edge_pool) > 0):
        return edge_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: edges")
        
def put_vec_e(vec): edge_pool.append(vec)

def get_vec_c():
    if (len(cell_pool) > 0):
        return cell_pool.pop()
    else:
        raise RuntimeError("Pool exhausted: cells")
        
def put_vec_c(vec): cell_pool.append(vec)

try:
    # load cython kernels, if compiled
    from _kt import _set_x_vec as set_x_vec
    from _kt import _cpy_x_vec as cpy_x_vec

except ImportError:
    raise RuntimeError("Cython back-end not found")

