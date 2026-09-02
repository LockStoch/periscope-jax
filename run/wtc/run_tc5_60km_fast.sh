
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
if [ ! -n "${SCHEME+z}" ]; then SCHEME="RK43-FB" ; fi

if command -v taskset >/dev/null 2>&1; then
  RUNNER="taskset --cpu-list 0-$((NUMCPU-1))"
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=true

if [ -f ${BINDIR}/swe.py ]
then

  opts=(
    --mesh-file=${MSHDIR}/"tc5_cvt_7.nc"
    --numthread=${NUMCPU}
    --integrate=${SCHEME}
    --time-span="1d" --save-time="1d" --stat-time="1d"
  )

  ${RUNNER} ${PYTHON} ${BINDIR}/swe.py "${opts[@]}"

else
  echo "PERISCOPE not found"
fi

