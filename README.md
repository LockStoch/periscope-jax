## `PERISCOPE`

`PERISCOPE` is a numerical toolbox for the simulation of geophysical flows using 
variable resolution unstructured meshes. 

<p align="center"> <img src = "../main/img/merger_3.png"> </p>

Presently, a nonlinear rotating shallow water model is available &mdash; solved on 
planar and spherical domains (either the full sphere or regional pieces thereof), with 
support for various boundary conditions, drag laws and forcing types. 

$$\begin{gather}
\frac{\partial h}{\partial t} + \nabla \cdot (u h) = D_{h} + S_{h}\ , 
\\\\\\
\frac{\partial u}{\partial t} + (u \cdot \nabla) u + f u^{\perp} = 
    -\nabla \Big(g(h + z_{b}) + \xi_{u}\Big) - c_{d} u + D_{u} + \frac{1}{h} \tau_{u} + S_{u}\ .
\end{gather}$$

See `MODELS.md` for additional detail regarding the formulation, numerics and model 
output.

A staggered, unstructured mesh mimetic finite-volume / difference discretisation 
is employed that maintains the energy (and to a lesser extent enstrophy) balances 
associated with geophysical flows.

One way to install the various dependencies needed is via `conda`:

    conda create --name periscope-utils python=X.Y
    conda activate periscope-utils
    pip install -r requirements.txt

`pip install -r requirements.txt` installs the CPU build of `jax`. To run on a
GPU, follow up with the appropriate CUDA/ROCm wheel for your system, e.g.:

    pip install -U "jax[cuda12]"

The (presently reduced, PGF + continuity) solver core is implemented in `JAX`
and runs on CPU or GPU via the same code path -- no separate build step is
required.

To run the shallow-water solver (see `swe.py --help`):

    python swe.py \
    --mesh-file="path+name-to-mpas-mesh+init-file" \
    --numthread=N \
    --time-span="WdXhYmZs" \ 
    --save-time="WdXhYmZs"

Solver output is saved to an MPAS-'ish' NetCDF file that can be visualised using
e.g. [paraview](https://www.paraview.org/download/).

Input files for various cases can be built from 
[MPAS-format](https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf) mesh 
files using the utilities provided &mdash; see the cases in `run` for details.

For example, to build and run the barotopic jet of 
[Galewesky et al](https://doi.org/10.1111/j.1600-0870.2004.00071.x) 
(meshes available from the repository release assets):

    python ICs/jet.py \
    --mesh-file="mesh_w_elev_cvt_7.nc" \
    --init-file="jet_cvt_7.nc" \
    --with-pert=True --radius=6371220.

    python swe.py \
    --mesh-file="jet_cvt_7.nc" \
    --numthread=N \
    --time-span="6d" --save-time="1d" --stat-time="1d"

Output is saved to the `out_jet_cvt_7.nc` file, which can be opened for visualisation 
using e.g. paraview.

