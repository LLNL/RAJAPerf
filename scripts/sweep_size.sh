#!/usr/bin/env bash

EXECUTABLES=()
SIZE_MIN=10000
SIZE_MAX=1000000
SIZE_RATIO=2

################################################################################
#
# Usage:
#     srun -n1 --exclusive sweep.sh -x raja-perf.exe [-- <raja perf args>]
#
# Parse any args for this script and consume them using shift
# leave the raja perf arguments if any for later use
#
# Examples:
#     lalloc 1 lrun -n1 sweep.sh -x raja-perf.exe -- <args>
#       # run a sweep of default problem sizes with executable `raja-perf.exe`
#       # with args `args`
#
#     srun -n1 --exclusive sweep.sh -x raja-perf.exe --size-min 1000
#            --size-max 10000 --size-ratio 2 -- <args>
#       # run a sweep of problem sizes 1K to 10K with ratio 2 (1K, 2K, 4K, 8K)
#       # with executable `raja-perf.exe` with args `args`
#
################################################################################
while [ "$#" -gt 0 ]; do

  if [[ "$1" =~ ^\-.* ]]; then

    if [[ "x$1" == "x-x" || "x$1" == "x--executable" ]]; then

      EXECUTABLES+=("$2")
      shift

    elif [[ "x$1" == "x-m" || "x$1" == "x--size-min" ]]; then

      SIZE_MIN="$2"
      shift

    elif [[ "x$1" == "x-M" || "x$1" == "x--size-max" ]]; then

      SIZE_MAX="$2"
      shift

    elif [[ "x$1" == "x-r" || "x$1" == "x--size-ratio" ]]; then

      SIZE_RATIO="$2"
      shift

    elif [[ "x$1" == "x--" ]]; then

      shift
      break

    else

      echo "Unknown arg: $1" 1>&2
      exit 1

    fi

  else
    break
  fi

  shift

done

echo "Running sweep with executables: ${EXECUTABLES[@]}"
echo "Sweeping from size $SIZE_MIN to $SIZE_MAX with ratio $SIZE_RATIO"
echo "extra args to executables are: $@"


################################################################################
# check sizes and ratio
################################################################################
if [[ "$SIZE_MIN" -le 0 ]]; then
  echo "Invalid size-min: $SIZE_MIN" 1>&2
  exit 1
fi
if [[ "$SIZE_MAX" -le 0 ]]; then
  echo "Invalid size-max: $SIZE_MAX" 1>&2
  exit 1
fi
if [[ "$SIZE_RATIO" -le 1 ]]; then
  echo "Invalid size-ratio: $SIZE_RATIO" 1>&2
  exit 1
fi
if [[ "$SIZE_MIN" -gt "$SIZE_MAX" ]]; then
  echo "Invalid sizes size-min: $SIZE_MIN, size-max: $SIZE_MAX" 1>&2
  exit 1
fi

EXEC_I=0
for exec in "${EXECUTABLES[@]}"; do

  echo "$exec"

  mkdir "RAJAPerf_$EXEC_I" || exit 1

  let EXEC_I=EXEC_I+1

done

SIZE="$SIZE_MIN"
while [[ "$SIZE" -le "$SIZE_MAX" ]]; do

  EXEC_I=0
  for exec in "${EXECUTABLES[@]}"; do

    SIZE_DIR="$(printf "SIZE_%09d" $SIZE)"
    OUT_DIR="RAJAPerf_$EXEC_I/$SIZE_DIR"

    mkdir "${OUT_DIR}" || exit 1

    echo "$exec -od ${OUT_DIR} --size $SIZE $@" | tee -a "${OUT_DIR}/raja-perf-sweep.txt"
          $exec -od ${OUT_DIR} --size $SIZE $@ &>> "${OUT_DIR}/raja-perf-sweep.txt"

    let EXEC_I=EXEC_I+1

  done

  let SIZE=SIZE*SIZE_RATIO

done
