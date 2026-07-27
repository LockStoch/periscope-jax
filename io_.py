
import time
import numpy as np
import netCDF4 as nc

""" NETCDF-4 output for SWE-solver; MPAS-style variables  
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from log import tcpu

from mem import variables

from _dx import calc_vars

class base: pass
out_ = base()
out_.uu_edge = False  # True to write to file
out_.vv_edge = False
out_.hh_bias = False
out_.hh_cell = False
out_.hh_edge = False
out_.hh_dual = False
out_.zt_cell = False
out_.qq_cell = False
out_.du_cell = False
out_.uh_cell = False
out_.ke_bias = False
out_.ke_cell = False
out_.pv_bias = False
out_.pv_dual = False
out_.rv_dual = False
out_.pv_cell = False
out_.rv_cell = False
out_.ux_cell = False
out_.uy_cell = False
out_.uz_cell = False
out_.nu_turb = False
out_.nu_wave = False
out_.nu_shoc = False
out_.nu_thin = False
out_.xi_tide = False
out_.xi_self = False
out_.uu_filt = False
out_.ke_filt = False

def save_step(save, mesh, mats, flow, cnfg, step, hh_cell, uu_edge,
                                                  qq_cell):

    hh_edge, hh_dual, hh_bias, \
    ke_cell, ke_bias, \
    rv_cell, pv_cell, rv_dual, pv_dual, pv_edge, pv_bias, \
    vv_edge, nu_turb, nu_wave, os_wave, nu_shoc, os_shoc, \
    nu_thin, uu_filt, Xi_tide, Xi_self = calc_vars (
        mesh, mats, flow, cnfg, hh_cell, uu_edge, qq_cell
        )

    ttic = time.time()

    data = nc.Dataset(save, "a", format="NETCDF4")

    # seconds elapsed since epoch
    data.timeisnow = cnfg.timeisnow

    # xt variables are tmp scratch

    if (out_.uu_edge):
        data.variables["uu_edge"][step, :, :] = \
            np.reshape(uu_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
            
    if (out_.vv_edge):
        data.variables["vv_edge"][step, :, :] = \
            np.reshape(vv_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
                
    if (out_.hh_bias):
        _t_dual = mats.dual_tail_sums * hh_bias
        _t_dual/= mesh.vert.area

        data.variables["hh_bias"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
                
    if (out_.hh_cell):         
        data.variables["hh_cell"][step, :, :] = \
            np.reshape(hh_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
                
    if (out_.hh_edge):         
        data.variables["hh_edge"][step, :, :] = \
            np.reshape(hh_edge[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))
                
    if (out_.hh_dual):         
        data.variables["hh_dual"][step, :, :] = \
            np.reshape(hh_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    
    if (out_.zt_cell):
        _t_cell = flow.zb_cell + hh_cell
    
        data.variables["zt_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.qq_cell):
        data.variables["qq_cell"][step, :, :] = \
            np.reshape(qq_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.du_cell):
        _t_cell = mats.cell_flux_sums * uu_edge
        _t_cell/= mesh.cell.area

        data.variables["du_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.uh_cell):
        _t_edge = uu_edge * hh_edge
        _t_cell = mats.cell_flux_sums * _t_edge
        _t_cell/= mesh.cell.area

        data.variables["uh_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.ke_cell):
        data.variables["ke_cell"][step, :, :] = \
            np.reshape(ke_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1)) 

    if (out_.pv_bias):
        _t_dual = mats.dual_tail_sums * pv_bias
        _t_dual/= mesh.vert.area

        data.variables["pv_bias"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    
    if (out_.pv_dual):        
        data.variables["pv_dual"][step, :, :] = \
            np.reshape(pv_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
    
    if (out_.rv_dual):
        data.variables["rv_dual"][step, :, :] = \
            np.reshape(rv_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))
                
    if (out_.pv_cell):        
        data.variables["pv_cell"][step, :, :] = \
            np.reshape(pv_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
    
    if (out_.rv_cell):
        data.variables["rv_cell"][step, :, :] = \
            np.reshape(rv_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
                
    if (out_.ux_cell):
        _t_cell = mats.cell_lsqr_xnrm * uu_edge

        data.variables["ux_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
                
    if (out_.uy_cell):
        _t_cell = mats.cell_lsqr_ynrm * uu_edge

        data.variables["uy_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))
                
    if (out_.uz_cell):
        _t_cell = mats.cell_lsqr_znrm * uu_edge

        data.variables["uz_cell"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))   
                
    if (out_.nu_turb):
        _t_dual = mats.dual_tail_sums * nu_turb
        _t_dual/= mesh.vert.area
    
        data.variables["nu_turb"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

    if (out_.nu_thin):
        _t_dual = mats.dual_tail_sums * nu_thin
        _t_dual/= mesh.vert.area
    
        data.variables["nu_thin"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

    if (out_.nu_wave):
        _t_dual = mats.dual_tail_sums * nu_wave
        _t_dual/= mesh.vert.area
    
        data.variables["nu_wave"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

        _t_dual = mats.dual_tail_sums * os_wave
        _t_dual/= mesh.vert.area
    
        data.variables["os_wave"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

    if (out_.nu_shoc):
        _t_dual = mats.dual_tail_sums * nu_shoc
        _t_dual/= mesh.vert.area
    
        data.variables["nu_shoc"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

        _t_dual = mats.dual_kite_sums * os_shoc
        _t_dual/= mesh.vert.area
    
        data.variables["os_shoc"][step, :, :] = \
            np.reshape(_t_dual[
                mesh.vert.irev - 1], (1, mesh.vert.size, 1))

    if (out_.xi_tide):
        data.variables["Xi_tide"][step, :, :] = \
            np.reshape(Xi_tide[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.xi_self):
        data.variables["Xi_self"][step, :, :] = \
            np.reshape(Xi_self[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    if (out_.uu_filt):
        data.variables["uu_filt"][step, :, :] = \
            np.reshape(uu_filt[
                mesh.edge.irev - 1], (1, mesh.edge.size, 1))

    if (out_.ke_filt):
        vv_filt = mats.edge_lsqr_perp * uu_filt

        _t_edge = .5 * uu_filt ** 2 + \
                  .5 * vv_filt ** 2

        _t_cell = mats.cell_wing_sums * _t_edge
        _t_cell/= mesh.cell.area

        data.variables["ke_filt"][step, :, :] = \
            np.reshape(_t_cell[
                mesh.cell.irev - 1], (1, mesh.cell.size, 1))

    data.close()
    
    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)


def save_last(save, mesh, mats, flow, cnfg, step, kp_sum_, en_sum_,
                                                  hh_min_, hh_max_,
                                                  uu_min_, uu_max_,
                                                  qq_min_, qq_max_,
                                                  zt_rms_,
                                         ke_ave_, ke_rms_, ke_max_,
                                         dk_ave_, dk_rms_, dk_max_):

    ttic = time.time()

    data = nc.Dataset(save, "a", format="NETCDF4")

    data.variables["subdomn"][:] = \
        mesh.cell.subd[mesh.cell.irev - 1]

    # xt variables are tmp scratch

    data.variables["kp_sums"][:] = kp_sum_
    data.variables["en_sums"][:] = en_sum_
    
    data.variables["hh_min_"][:] = \
               hh_min_[mesh.cell.irev - 1]
    data.variables["hh_max_"][:] = \
               hh_max_[mesh.cell.irev - 1]
               
    data.variables["uu_min_"][:] = \
               uu_min_[mesh.edge.irev - 1]
    data.variables["uu_max_"][:] = \
               uu_max_[mesh.edge.irev - 1]

    data.variables["zt_rms_"][:] = np.sqrt(
    1./ step * zt_rms_[mesh.cell.irev - 1])

    data.variables["ke_ave_"][:] = \
    1./ step * ke_ave_[mesh.cell.irev - 1]
    data.variables["ke_rms_"][:] = np.sqrt(
    1./ step * ke_rms_[mesh.cell.irev - 1])
    data.variables["ke_max_"][:] = \
               ke_max_[mesh.cell.irev - 1]

    xt_dual = mats.dual_tail_sums * dk_ave_
    xt_dual/= mesh.vert.area

    data.variables["dk_ave_"][:] = \
               xt_dual[mesh.vert.irev - 1]

    xt_dual = mats.dual_tail_sums * dk_rms_
    xt_dual/= mesh.vert.area

    data.variables["dk_rms_"][:] = np.sqrt(
               xt_dual[mesh.vert.irev - 1])

    xt_dual = mats.dual_tail_sums * dk_max_
    xt_dual/= mesh.vert.area

    data.variables["dk_max_"][:] = \
               xt_dual[mesh.vert.irev - 1]
    
    xt_dual = mats.dual_tail_sums * variables.visc_u2
    xt_dual/= mesh.vert.area
    
    data.variables["u2_visc"][:] = \
               xt_dual[mesh.vert.irev - 1]
    
    xt_dual = mats.dual_tail_sums * variables.visc_u4
    xt_dual/= mesh.vert.area
    
    data.variables["u4_visc"][:] = \
               xt_dual[mesh.vert.irev - 1]
    
    xt_dual = mats.dual_tail_sums * variables.diff_h2
    xt_dual/= mesh.vert.area

    data.variables["h2_diff"][:] = \
               xt_dual[mesh.vert.irev - 1]

    xt_dual = mats.dual_tail_sums * variables.diff_h4
    xt_dual/= mesh.vert.area

    data.variables["h4_diff"][:] = \
               xt_dual[mesh.vert.irev - 1] ** 2
    
    data.close()

    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)
    
    
def init_file(name, cnfg, save, mesh, flow):

    ttic = time.time()
    
    TIMECHUNK = 4
    MESHCHUNK = 4096
    LVLSCHUNK = 1

    data = nc.Dataset(save, "w", format="NETCDF4")
    if (mesh.rsph is not None):
        data.on_a_sphere = "YES"
        data.sphere_radius = mesh.rsph
    else:
        data.on_a_sphere = "NO"
    data.sphere_flatten = mesh.flat
    data.config_gravity = flow.gravity
    if (mesh.wrap [0] is None and 
        mesh.wrap [1] is None):
        data.is_periodic = "NO"
    else:
        data.is_periodic = "YES"
        if (mesh.wrap[0] is not None):
            data.x_period = mesh.wrap[0]
        if (mesh.wrap[1] is not None):
            data.y_period = mesh.wrap[1]
    data.source = "PERISCOPE"

    for attr in vars(cnfg):  # add user-opts to output
        vals = getattr(cnfg, attr)
        if (vals is None): continue
        if (type(vals) == list): continue
        if (type(vals) == bool): vals = int(vals)
        setattr (data, attr, vals)

    data.createDimension("Time", None)
    data.createDimension("Step", None)
    data.createDimension("TWO", 2)
    data.createDimension("nCells", mesh.cell.size)
    data.createDimension("nEdges", mesh.edge.size)
    data.createDimension("nVertices", mesh.vert.size)
    data.createDimension("nVertLevels", 1)
    data.createDimension("maxEdges", np.max(mesh.cell.topo) * 1)
    data.createDimension("maxEdges2", np.max(mesh.cell.topo) * 2)
    data.createDimension("vertexDegree", 3)

    data.createVariable("lonCell", "f8", ("nCells"))
    data["lonCell"][:] = mesh.cell.xlon
    data.createVariable("latCell", "f8", ("nCells"))
    data["latCell"][:] = mesh.cell.ylat
    data.createVariable("xCell", "f8", ("nCells"))
    data["xCell"][:] = mesh.cell.xpos
    data.createVariable("yCell", "f8", ("nCells"))
    data["yCell"][:] = mesh.cell.ypos
    data.createVariable("zCell", "f8", ("nCells"))
    data["zCell"][:] = mesh.cell.zpos
    data.createVariable("areaCell", "f4", ("nCells"))
    data["areaCell"][:] = mesh.cell.area
    data.createVariable(
        "verticesOnCell", "i4", ("nCells", "maxEdges"))
    data["verticesOnCell"][:, :] = mesh.cell.vert
    data.createVariable(
        "edgesOnCell", "i4", ("nCells", "maxEdges"))
    data["edgesOnCell"][:, :] = mesh.cell.edge
    data.createVariable(
        "cellsOnCell", "i4", ("nCells", "maxEdges"))
    data["cellsOnCell"][:, :] = mesh.cell.cell
    data.createVariable("nEdgesOnCell", "i4", ("nCells"))
    data["nEdgesOnCell"][:] = mesh.cell.topo

    data.createVariable("lonEdge", "f8", ("nEdges"))
    data["lonEdge"][:] = mesh.edge.xlon
    data.createVariable("latEdge", "f8", ("nEdges"))
    data["latEdge"][:] = mesh.edge.ylat
    data.createVariable("xEdge", "f8", ("nEdges"))
    data["xEdge"][:] = mesh.edge.xpos
    data.createVariable("yEdge", "f8", ("nEdges"))
    data["yEdge"][:] = mesh.edge.ypos
    data.createVariable("zEdge", "f8", ("nEdges"))
    data["zEdge"][:] = mesh.edge.zpos
    data.createVariable("dvEdge", "f8", ("nEdges"))
    data["dvEdge"][:] = mesh.edge.vlen
    data.createVariable("dcEdge", "f8", ("nEdges"))
    data["dcEdge"][:] = mesh.edge.clen
    data.createVariable("angleEdge", "f8", ("nEdges"))
    data["angleEdge"][:] = mesh.edge.beta
    data.createVariable(
        "verticesOnEdge", "i4", ("nEdges", "TWO"))
    data["verticesOnEdge"][:, :] = mesh.edge.vert
    data.createVariable(
        "weightsOnEdge", "f4", ("nEdges", "maxEdges2"))
    data["weightsOnEdge"][:, :] = mesh.edge.wmul
    data.createVariable(
        "cellsOnEdge", "i4", ("nEdges", "TWO"))
    data["cellsOnEdge"][:, :] = mesh.edge.cell
    data.createVariable(
        "edgesOnEdge", "i4", ("nEdges", "maxEdges2"))
    data["edgesOnEdge"][:, :] = mesh.edge.edge
    data.createVariable("nEdgesOnEdge", "i4", ("nEdges"))
    data["nEdgesOnEdge"][:] = mesh.edge.topo

    data.createVariable("lonVertex", "f8", ("nVertices"))
    data["lonVertex"][:] = mesh.vert.xlon
    data.createVariable("latVertex", "f8", ("nVertices"))
    data["latVertex"][:] = mesh.vert.ylat
    data.createVariable("xVertex", "f8", ("nVertices"))
    data["xVertex"][:] = mesh.vert.xpos
    data.createVariable("yVertex", "f8", ("nVertices"))
    data["yVertex"][:] = mesh.vert.ypos
    data.createVariable("zVertex", "f8", ("nVertices"))
    data["zVertex"][:] = mesh.vert.zpos
    data.createVariable("areaTriangle", "f4", ("nVertices"))
    data["areaTriangle"][:] = mesh.vert.area
    data.createVariable(
        "kiteAreasOnVertex", "f4", ("nVertices", "vertexDegree"))
    data["kiteAreasOnVertex"][:, :] = mesh.vert.kite
    data.createVariable(
        "edgesOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["edgesOnVertex"][:, :] = mesh.vert.edge
    data.createVariable(
        "cellsOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["cellsOnVertex"][:, :] = mesh.vert.cell
   
    data.createVariable("is_mask", "i1", ("nCells"))
    data["is_mask"].long_name = "TRUE for cells masked out of flow"
    data["is_mask"][:] = flow.is_mask
    
    data.createVariable("is_open", "i1", ("nEdges"))
    data["is_open"].long_name = "TRUE for edges on open boundaries"
    data["is_open"][:] = flow.is_open
    
    data.createVariable("bc_slip", "f4", ("nEdges"))
    data["bc_slip"].long_name = "Wall slip coefficient on edges"
    data["bc_slip"][:] = flow.bc_slip
   
    data.createVariable("zb_cell", "f4", ("nCells"))
    data["zb_cell"].long_name = "Elevation of lower surface"
    data["zb_cell"][:] = flow.zb_cell
    data.createVariable("zb_drag", "f4", ("nCells"))
    data["zb_drag"].long_name = \
        "Elevation of lower surface, hrm. mean for subgrid drag"
    data["zb_drag"][:] = flow.zb_drag

    data.createVariable("ff_cell", "f4", ("nCells"))
    data["ff_cell"].long_name = "Coriolis parameter on cells"
    data["ff_cell"][:] = flow.ff_cell
    data.createVariable("ff_edge", "f4", ("nEdges"))
    data["ff_edge"].long_name = "Coriolis parameter on edges"
    data["ff_edge"][:] = flow.ff_edge
    data.createVariable("ff_vert", "f4", ("nVertices"))
    data["ff_vert"].long_name = "Coriolis parameter on duals"
    data["ff_vert"][:] = flow.ff_vert

    data.createVariable("c1_edge", "f4", ("nEdges"))
    data["c1_edge"].long_name = "Drag coefficient (linlaw) on edges"
    data["c1_edge"][:] = flow.c1_edge
    data.createVariable("c2_edge", "f4", ("nEdges"))
    data["c2_edge"].long_name = "Drag coefficient (sqrlaw) on edges"
    data["c2_edge"][:] = flow.c2_edge

    data.createVariable("z0_edge", "f4", ("nEdges"))
    data["z0_edge"].long_name = "Drag roughness (loglaw) on edges"
    data["z0_edge"][:] = flow.z0_edge
    data.createVariable("n0_edge", "f4", ("nEdges"))
    data["n0_edge"].long_name = "Manning coeff. (manlaw) on edges"
    data["n0_edge"][:] = flow.n0_edge

    data.createVariable("subdomn", "i4", ("nCells"))
    data.variables["subdomn"].long_name = "Subdomain ID, on cells"

    data.createVariable("h2_diff", "f4", ("nVertices"))
    data["h2_diff"].long_name = \
        "DEL^2(H) diffusion coefficient, remapped to duals"
    data.createVariable("h4_diff", "f4", ("nVertices"))
    data["h4_diff"].long_name = \
        "DEL^4(H) diffusion coefficient, remapped to duals"
    
    data.createVariable("u2_visc", "f4", ("nVertices"))
    data["u2_visc"].long_name = \
        "DEL^2(U) viscosity coefficient, remapped to duals"
    data.createVariable("u4_visc", "f4", ("nVertices"))
    data["u4_visc"].long_name = \
        "DEL^4(U) viscosity coefficient, remapped to duals"
    
    data.createVariable(
        "u0_edge", "f8", ("nEdges", "nVertLevels"))
    data["u0_edge"].long_name = "Normal velocity initial conditions" 
    data["u0_edge"][:] = flow.uu_edge
    data.createVariable(
        "h0_cell", "f8", ("nCells", "nVertLevels"))    
    data["h0_cell"].long_name = "Layer thickness initial conditions"
    data["h0_cell"][:] = flow.hh_cell

    data.createVariable(
        "uu_min_", "f4", ("nEdges", "nVertLevels"))
    data["uu_min_"].long_name = "Min. normal velocity for all steps"
    data.createVariable(
        "uu_max_", "f4", ("nEdges", "nVertLevels"))
    data["uu_max_"].long_name = "Max. normal velocity for all steps"
    
    data.createVariable(
        "zt_rms_", "f4", ("nCells", "nVertLevels"))    
    data["zt_rms_"].long_name = "RMS. of free surface for all steps"

    data.createVariable(
        "ke_ave_", "f4", ("nCells", "nVertLevels"))    
    data["ke_ave_"].long_name = "Mean kinetic energy for all steps"
    data.createVariable(
        "ke_rms_", "f4", ("nCells", "nVertLevels"))    
    data["ke_rms_"].long_name = "RMS. kinetic energy for all steps"
    data.createVariable(
        "ke_max_", "f4", ("nCells", "nVertLevels"))    
    data["ke_max_"].long_name = "Max. kinetic energy for all steps"

    data.createVariable(
        "dk_ave_", "f4", ("nVertices", "nVertLevels"))    
    data["dk_ave_"].long_name = "Mean dissipation for all steps"
    data.createVariable(
        "dk_rms_", "f4", ("nVertices", "nVertLevels"))    
    data["dk_rms_"].long_name = "RMS. dissipation for all steps"
    data.createVariable(
        "dk_max_", "f4", ("nVertices", "nVertLevels"))    
    data["dk_max_"].long_name = "Max. dissipation for all steps"

    data.createVariable(
        "hh_min_", "f4", ("nCells", "nVertLevels"))    
    data["hh_min_"].long_name = "Min. layer thickness for all steps"
    data.createVariable(
        "hh_max_", "f4", ("nCells", "nVertLevels"))    
    data["hh_max_"].long_name = "Max. layer thickness for all steps"

    data.createVariable("kp_sums", "f4", ("Step"))
    data["kp_sums"].long_name = \
        "Energetics invariant: total KE+PE over time"
    data.createVariable("en_sums", "f4", ("Step"))
    data["en_sums"].long_name = \
        "Rotational invariant: total PV**2 over time"

    if ("uu_edge" in cnfg.save_vars):
        data.createVariable(
            "uu_edge", "f8", ("Time", "nEdges", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["uu_edge"].long_name = "Normal velocity on edges" 
        out_.uu_edge = True

    if ("vv_edge" in cnfg.save_vars):
        data.createVariable(
            "vv_edge", "f4", ("Time", "nEdges", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["vv_edge"].long_name = "Tangential velocity on edges" 
        out_.vv_edge = True
    
    if ("hh_cell" in cnfg.save_vars):   
        data.createVariable(
            "hh_cell", "f8", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["hh_cell"].long_name = "Layer thickness on cells"
        out_.hh_cell = True
        
    if ("hh_edge" in cnfg.save_vars):   
        data.createVariable(
            "hh_edge", "f4", ("Time", "nEdges", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["hh_edge"].long_name = "Layer thickness on edges"
        out_.hh_edge = True
        
    if ("hh_dual" in cnfg.save_vars):   
        data.createVariable(
            "hh_dual", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["hh_dual"].long_name = "Layer thickness on vertices"
        out_.hh_dual = True
        
    if ("zt_cell" in cnfg.save_vars):
        data.createVariable(
            "zt_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["zt_cell"].long_name = "Elevation of upper surface"
        out_.zt_cell = True

    if ("qq_cell" in cnfg.save_vars):   
        data.createVariable(
            "qq_cell", "f8", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["qq_cell"].long_name = "Tracer value on cells"
        out_.qq_cell = True

    if ("du_cell" in cnfg.save_vars):
        data.createVariable(
            "du_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["du_cell"].long_name = \
            "Divergence of velocity on cells"
        out_.du_cell = True
    
    if ("uh_cell" in cnfg.save_vars):
        data.createVariable(
            "uh_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["uh_cell"].long_name = \
            "Divergence of thickness flux on cells"
        out_.uh_cell = True

    if ("hh_bias" in cnfg.save_vars):
        data.createVariable(
            "hh_bias", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["hh_bias"].long_name = \
            "Upwind-bias for HH, remapped to duals"
        out_.hh_bias = True

    if ("ke_bias" in cnfg.save_vars):
        data.createVariable(
            "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["ke_bias"].long_name = \
            "Upwind-bias for KE, remapped to duals"
        out_.ke_bias = True
    
    if ("pv_bias" in cnfg.save_vars):
        data.createVariable(
            "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["pv_bias"].long_name = \
            "Upwind-bias for PV, remapped to duals"
        out_.pv_bias = True

    if ("ke_cell" in cnfg.save_vars):
        data.createVariable(
            "ke_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["ke_cell"].long_name = "Kinetic energy on cells"
        out_.ke_cell = True
        
    if ("pv_dual" in cnfg.save_vars):
        data.createVariable(
            "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["pv_dual"].long_name = \
            "Absolute vorticity curl(u) + f, on duals"
        out_.pv_dual = True       
    
    if ("rv_dual" in cnfg.save_vars):
        data.createVariable(
            "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["rv_dual"].long_name = "Relative vorticity on duals"
        out_.rv_dual = True
        
    if ("pv_cell" in cnfg.save_vars):
        data.createVariable(
            "pv_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["pv_cell"].long_name = \
            "Absolute vorticity curl(u) + f, on cells"
        out_.pv_cell = True       
    
    if ("rv_cell" in cnfg.save_vars):
        data.createVariable(
            "rv_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["rv_cell"].long_name = "Relative vorticity on cells"
        out_.rv_cell = True
        
    if ("ux_cell" in cnfg.save_vars):
        data.createVariable(
            "ux_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["ux_cell"].long_name = \
            "Reconstructed velocity on cells in x-axis direction"
        out_.ux_cell = True
        
    if ("uy_cell" in cnfg.save_vars):
        data.createVariable(
            "uy_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["uy_cell"].long_name = \
            "Reconstructed velocity on cells in y-axis direction"
        out_.uy_cell = True
        
    if ("uz_cell" in cnfg.save_vars):
        data.createVariable(
            "uz_cell", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["uz_cell"].long_name = \
            "Reconstructed velocity on cells in z-axis direction"
        out_.uz_cell = True
        
    if ("nu_turb" in cnfg.save_vars):
        data.createVariable(
            "nu_turb", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["nu_turb"].long_name = \
            "Turbulent eddy viscosity, remapped to duals"
        out_.nu_turb = True

    if ("nu_thin" in cnfg.save_vars):
        data.createVariable(
            "nu_thin", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["nu_thin"].long_name = \
            "Wet-dry region viscosity, remapped to duals"
        out_.nu_thin = True

    if ("nu_wave" in cnfg.save_vars):
        data.createVariable(
            "nu_wave", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["nu_wave"].long_name = \
            "Artificial diffusivity (waves), remapped to duals"
        data.createVariable(
            "os_wave", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["os_wave"].long_name = \
            "Oscillation sensor (waves), remapped to duals"
        out_.nu_wave = True

    if ("nu_shoc" in cnfg.save_vars):
        data.createVariable(
            "nu_shoc", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["nu_shoc"].long_name = \
            "Artificial diffusivity (shock), remapped to duals"
        data.createVariable(
            "os_shoc", "f4", ("Time", "nVertices", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["os_shoc"].long_name = \
            "Oscillation sensor (shock), remapped to duals"
        out_.nu_shoc = True

    if ("xi_tide" in cnfg.save_vars):
        data.createVariable(
            "Xi_tide", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["Xi_tide"].long_name = \
            "Applied tidal potential on cells"
        out_.xi_tide = True

    if ("xi_self" in cnfg.save_vars):
        data.createVariable(
            "Xi_self", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["Xi_self"].long_name = \
            "Self attraction & loading potential on cells"
        out_.xi_self = True

    if ("uu_filt" in cnfg.save_vars):
        data.createVariable(
            "uu_filt", "f4", ("Time", "nEdges", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["uu_filt"].long_name = \
            "Time-filtered normal velocity on edges"
        out_.uu_filt = True

    if ("ke_filt" in cnfg.save_vars):
        data.createVariable(
            "ke_filt", "f4", ("Time", "nCells", "nVertLevels"), 
            chunksizes=(TIMECHUNK, MESHCHUNK, LVLSCHUNK))
        data["ke_filt"].long_name = \
            "Time-filtered kinetic energy on cells"
        out_.ke_filt = True

    data.close()
    
    ttoc = time.time()
    tcpu.filewrite = tcpu.filewrite + (ttoc - ttic)
    
