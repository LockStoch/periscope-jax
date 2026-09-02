
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
    --mesh-file=${MSHDIR}/"tsu_cvt_8.nc"
    --numthread=${NUMCPU}
    --integrate=${SCHEME}
    --time-span="8h" --save-time="30m" --stat-time="30m"
    --wetdry-h0=1.E-03
    --loglaw-z0=0.0100 --loglaw-lo=0.0025 --loglaw-hi=1.
    --uu-visc-4=1.E+11 --hh-diff-2=1.E+02
  )

  ${RUNNER} ${PYTHON} ${BINDIR}/swe.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

