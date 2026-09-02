
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
if [ ! -n "${SOLNFILE+z}" ]; then SOLNFILE="" ; fi

# NB: deliberately not using taskset to re-pin CPU affinity here.
# Job schedulers already restrict a job to its allotted CPUs; a
# hardcoded 0-N range assumes those IDs start at 0, which isn't
# guaranteed (e.g. under shared-node scheduling, a job can land on
# any slice of a node's CPUs). OMP_PLACES/OMP_PROC_BIND below handle
# thread placement within whatever the scheduler already grants.
RUNNER=""

export OMP_PLACES=cores
export OMP_PROC_BIND=true

if [ -f ${BINDIR}/swe.py ]
then

  opts=(
    --mesh-file=${MSHDIR}/"jet_cvt_7.nc"
    --soln-file=${SOLNFILE}
    --numthread=${NUMCPU}
    --integrate=${SCHEME}
    --time-step="60" --time-span="3d" --save-time="1d" --stat-time="1d"
  )

  ${RUNNER} ${PYTHON} ${BINDIR}/swe.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

