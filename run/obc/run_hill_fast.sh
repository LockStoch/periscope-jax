
if [ ! -n "${BINDIR+z}" ]; then
  echo "BINDIR=path/to/periscope is required"; exit 1
fi

if [ ! -n "${MSHDIR+z}" ]; then
  echo "MSHDIR=path/to/mesh-dirs is required"; exit 1
fi

if [ ! -n "${NUMCPU+z}" ]; then
  echo "NUMCPU=N is required"; exit 1
fi

if [ ! -n "${PYTHON+z}" ]; then PYTHON="python3" ; fi
if [ ! -n "${SCHEME+z}" ]; then SCHEME="RK33-FB" ; fi

# NB: deliberately not using taskset to re-pin CPU affinity here.
# Under SLURM with shared node access, the job's cgroup-assigned CPU
# IDs don't necessarily start at 0, so a hardcoded 0-N range can fall
# entirely outside what the job is actually allowed to use. SLURM's
# cgroup already restricts this job correctly on its own -- OMP_PLACES
# / OMP_PROC_BIND below handle thread placement within that.
RUNNER=""

export OMP_PLACES=cores
export OMP_PROC_BIND=true

if [ -f ${BINDIR}/swe.py ]
then

  opts=(
    --mesh-file=${MSHDIR}/"hill_3.nc"
    --forc-file=${MSHDIR}/"frc_hill_3.nc"
    --numthread=${NUMCPU}
    --integrate=${SCHEME}
    --num-steps=100 --save-freq=100 --stat-freq=100
    --wetdry-h0=1.E-03
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1.
    --uu-visc-4=1.E+11 --hh-diff-2=1.E+02
    --leith-chi=1.E+00 --leith-max=1.E+04
  )

  ${RUNNER} ${PYTHON} ${BINDIR}/swe.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

