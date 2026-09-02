
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

if command -v taskset >/dev/null 2>&1; then
  RUNNER="taskset --cpu-list 0-$((NUMCPU-1))"
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=true

if [ -f ${BINDIR}/swe.py ]
then

  opts=(
    --mesh-file=${MSHDIR}/"wake_3.nc"
    --forc-file=${MSHDIR}/"frc_wake_3.nc"
    --numthread=${NUMCPU}
    --integrate=${SCHEME}
    --num-steps=1000 --save-freq=100 --stat-freq=100
    --wetdry-h0=1.E-03
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1.
    --uu-visc-4=1.E+11 --hh-diff-2=1.E+02
    --leith-chi=1.E+00 --leith-max=1.E+04
  )

  ${RUNNER} ${PYTHON} ${BINDIR}/swe.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

