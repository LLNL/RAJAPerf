#!/usr/bin/env bash
#
# Run a matrix of RAJAPerf builds and runs from a compiler/version list.
#
# This script is a thin driver around the existing lc-build scripts. It reads a
# list of compiler configurations, sources the appropriate build script for each
# entry, builds the tree, and optionally runs either a normal benchmark command
# or a throughput factor sweep.

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/lc-builds/run_compiler_matrix.sh <build_script.sh|from-list> <compiler_list.txt> [options] -- [raja-perf args...]

Description:
  Reads <compiler_list.txt> (one entry per line) and, for each line, invokes
  a build script with the tokens on that line as its arguments, then builds and
  runs the suite from that build directory.

  The build script is sourced (not executed) so any 'module load ...' it does
  applies to the subsequent build+run.

List file format:
  - Blank lines and lines starting with '#' are ignored.
  - By default, each line is treated as arguments to the build script.
    Examples (for toss4_gcc.sh):
      12.3.0
      12.3.0 3.27.4 -DENABLE_OPENMP=On
    Examples (for MPI lc-build scripts):
      # toss4_mvapich2_gcc.sh rows are: mvapich2_version gcc_version [cmake_version] [cmake args...]
      2.3.7 12.3.0
      2.3.7 12.3.0 3.27.4 -DENABLE_OPENMP=On
    Examples (for MPI + device-arch lc-build scripts):
      # toss4_cray-mpich_amdclang.sh rows are:
      #   cray_mpich_version hip_compiler_version hip_arch [cmake_version] [cmake args...]
      9.0.1 7.2.1 gfx942
      9.0.1 7.2.1 gfx942 3.27.4 -DCMAKE_CXX_FLAGS="-munsafe-fp-atomics"
      # toss4_mvapich2_nvcc_gcc.sh rows are:
      #   mvapich2_version nvcc_version cuda_arch gcc_version [cmake_version] [cmake args...]
      2.3.7 12.2.2 90 10.3.1
      2.3.7 12.2.2 90 10.3.1 3.27.4 -DENABLE_OPENMP=On

  - If <build_script.sh> is 'from-list', the first token on each line is the
    build script and the remaining tokens are arguments to that script. Use
    this to mix non-MPI and MPI compiler builds in one run:
      toss4_nvcc_gcc.sh 12.3.0 sm_80
      toss4_mvapich2_gcc.sh 2.3.7 12.3.0
      toss4_mvapich2_icpx.sh 2.3.7 2022.1.0
      toss4_cray-mpich_amdclang.sh 9.0.1 7.2.1 gfx942
      toss4_mvapich2_nvcc_gcc.sh 2.3.7 12.2.2 90 10.3.1

Options:
  --kernel <name>        Add a kernel or group to run (repeatable)
  --kernel-file <path>   File with kernels/groups (one per line; '#' comments ok)
  --run-cmd <string>     Required to run; launcher prefix. End launcher options
                         with `--` so RAJAPerf args can be appended (e.g.
                         "srun -N1 -n1 -c 64 --")
  --throughput           Run a factor sweep for each requested kernel. Each run
                         gets --memory-allocated BASEMEM*factor and an outfile
                         named <kernel>_factor_<factor>.
  --base-mem <bytes>     Base memory for --throughput (default: 50000)
  --factor <N>           Add one throughput factor (repeatable)
  --factors <list>       Add throughput factors from a comma/space separated list
  --throughput-outdir <path>
                         Output directory for throughput files, relative to each
                         build dir unless absolute (default: throughput)
  --no-warmup-same       Do not add --warmup-perfrun-same
  --no-skip-existing     Always reconfigure (do not reuse existing build dirs)
  -j, --jobs <N>        Parallel build jobs (default: nproc)
  --configure-only      Only run configure step (skip build+run)
  --build-only          Configure (if needed) + build (skip run)
  --keep-going          Continue after failures (default: on)
  --fail-fast           Stop at first failure
  --log-dir <path>      Log directory (default: logs/<script>-YYYYmmdd-HHMMSS)
  -h, --help            Show this help

Example:
  scripts/lc-builds/run_compiler_matrix.sh toss4_amdclang.sh compiler_list.txt --run-cmd "srun -N1 -n1 -c 64 --" --kernel INIT3 -- -i 10
  scripts/lc-builds/run_compiler_matrix.sh toss4_amdclang.sh compiler_list.txt --kernel-file kernels.txt -- --dryrun
  scripts/lc-builds/run_compiler_matrix.sh toss4_mvapich2_gcc.sh mpi_gcc.txt --run-cmd "srun -N1 -n2 --" --kernel-file kernels.txt -- --variants OpenMP
  scripts/lc-builds/run_compiler_matrix.sh from-list compiler_matrix.txt --run-cmd "srun -N1 -n4 --" --throughput --factors "1 4 16 64 256 1024" --kernel-file kernels.txt -- --npasses 1 --variants CUDA
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

build_script="$1"
list_file="$2"
shift 2

resolve_build_script_path() {
  # Allow bare script names like "toss4_amdclang.sh" by resolving them relative
  # to scripts/lc-builds, while still accepting explicit absolute/relative paths.
  local candidate="$1"
  if [[ "${candidate}" != /* && "${candidate}" != ./* && "${candidate}" != ../* ]]; then
    if [[ -f "${script_dir}/${candidate}" ]]; then
      candidate="${script_dir}/${candidate}"
    fi
  fi

  if [[ ! -f "${candidate}" ]]; then
    return 1
  fi

  echo "${candidate}"
}

jobs=""
configure_only=0
build_only=0
keep_going=1
log_dir=""
requested_kernels=()
skip_existing=1
warmup_same=1
run_cmd=""
run_args=()
throughput=0
throughput_base_mem=50000
throughput_factors=()
throughput_outdir="throughput"

case "${build_script}" in
  from-list|from_list|FROM-LIST|FROM_LIST|-)
    build_script_from_list=1
    ;;
  *)
    build_script_from_list=0
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel)
      requested_kernels+=("${2:-}")
      shift 2
      ;;
    --kernel-file)
      kernel_file="${2:-}"
      if [[ -z "${kernel_file}" || ! -f "${kernel_file}" ]]; then
        echo "Kernel file not found: ${kernel_file}" >&2
        exit 2
      fi
      while IFS= read -r raw_kline || [[ -n "${raw_kline}" ]]; do
        kline="${raw_kline#"${raw_kline%%[![:space:]]*}"}" # ltrim
        kline="${kline%"${kline##*[![:space:]]}"}"         # rtrim
        if [[ -z "${kline}" || "${kline:0:1}" == "#" ]]; then
          continue
        fi
        requested_kernels+=("${kline}")
      done < "${kernel_file}"
      shift 2
      ;;
    --no-skip-existing)
      skip_existing=0
      shift
      ;;
    --no-warmup-same)
      warmup_same=0
      shift
      ;;
    --run-cmd)
      run_cmd="${2:-}"
      shift 2
      ;;
    --throughput)
      throughput=1
      shift
      ;;
    --base-mem)
      throughput_base_mem="${2:-}"
      shift 2
      ;;
    --factor)
      throughput_factors+=("${2:-}")
      shift 2
      ;;
    --factors)
      factor_list="${2:-}"
      factor_list="${factor_list//,/ }"
      for factor in ${factor_list}; do
        throughput_factors+=("${factor}")
      done
      shift 2
      ;;
    --throughput-outdir)
      throughput_outdir="${2:-}"
      shift 2
      ;;
    -j|--jobs)
      jobs="${2:-}"
      shift 2
      ;;
    --configure-only)
      configure_only=1
      shift
      ;;
    --build-only)
      build_only=1
      shift
      ;;
    --keep-going)
      keep_going=1
      shift
      ;;
    --fail-fast)
      keep_going=0
      shift
      ;;
    --log-dir)
      log_dir="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      run_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

# If we're going to run, require an explicit run command prefix.
if [[ "${configure_only}" -eq 0 && "${build_only}" -eq 0 && -z "${run_cmd}" ]]; then
  echo "Missing required --run-cmd (use e.g. --run-cmd \"srun -N1 -n1 -c 64 --\")." >&2
  exit 2
fi

if [[ "${run_cmd}" == "direct" ]]; then
  echo "The --run-cmd direct shortcut is no longer supported; pass an explicit launcher prefix." >&2
  exit 2
fi

# Translate requested kernels into raja-perf args (prepend so user can still override after '--').
kernel_args=()
if [[ ${#requested_kernels[@]} -gt 0 ]]; then
  kernel_args=(--kernels "${requested_kernels[@]}")
fi

# Add warmup behavior unless user explicitly controls warmup in run args.
warmup_args=()
if [[ "${warmup_same}" -eq 1 ]]; then
  has_warmup_control=0
  for arg in "${run_args[@]}"; do
    case "${arg}" in
      --warmup-kernels|-wk|--warmup-disable|--warmup-perfrun-same)
        has_warmup_control=1
        break
        ;;
    esac
  done
  if [[ "${has_warmup_control}" -eq 0 ]]; then
    warmup_args=(--warmup-perfrun-same)
  fi
fi

if [[ "${build_script_from_list}" -eq 0 ]]; then
  if ! build_script="$(resolve_build_script_path "${build_script}")"; then
    echo "Build script not found: ${build_script}" >&2
    exit 2
  fi
fi

if [[ ! -f "${list_file}" ]]; then
  echo "List file not found: ${list_file}" >&2
  exit 2
fi

if [[ -z "${jobs}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  else
    jobs="1"
  fi
fi

if [[ "${throughput}" -eq 1 && ${#throughput_factors[@]} -eq 0 ]]; then
  throughput_factors=(1 4 16 32 64 128 256 512 1024)
fi

if [[ -z "${log_dir}" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  if [[ "${build_script_from_list}" -eq 1 ]]; then
    base="compiler-matrix"
  else
    base="$(basename -- "${build_script}")"
    base="${base%.sh}"
  fi
  log_dir="${repo_root}/logs/${base}-${ts}"
fi
mkdir -p "${log_dir}"

# Print the resolved run configuration once up front so each log directory is
# self-describing even if the caller did not capture the command line elsewhere.
echo "Repo root    : ${repo_root}"
if [[ "${build_script_from_list}" -eq 1 ]]; then
  echo "Build script : from list"
else
  echo "Build script : ${build_script}"
fi
echo "List file    : ${list_file}"
echo "Jobs         : ${jobs}"
echo "Log dir      : ${log_dir}"
echo "Kernels      : ${requested_kernels[*]:-(all)}"
echo "Skip existing: ${skip_existing}"
echo "Warmup same  : ${warmup_same}"
echo "Run cmd      : ${run_cmd:-"(none)"}"
echo "Run args     : ${run_args[*]:-(none)}"
echo "Throughput   : ${throughput}"
if [[ "${throughput}" -eq 1 ]]; then
  echo "Base memory  : ${throughput_base_mem}"
  echo "Factors      : ${throughput_factors[*]}"
  echo "Output dir   : ${throughput_outdir}"
fi
echo

failures=0
total=0

line_no=0
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  line_no=$((line_no + 1))

  line="${raw_line#"${raw_line%%[![:space:]]*}"}" # ltrim
  line="${line%"${line##*[![:space:]]}"}"         # rtrim

  if [[ -z "${line}" || "${line:0:1}" == "#" ]]; then
    continue
  fi

  total=$((total + 1))
  entry_id="$(printf "%03d" "${total}")"

  # Each non-comment line becomes one matrix entry. In normal mode the line is
  # passed as arguments to a single build script. In from-list mode the first
  # token selects the build script and the remaining tokens are its arguments.
  IFS=$' \t' read -r -a line_tokens <<<"${line}"
  if [[ "${build_script_from_list}" -eq 1 ]]; then
    entry_build_script="${line_tokens[0]:-}"
    script_args=("${line_tokens[@]:1}")
    if [[ -z "${entry_build_script}" ]]; then
      echo "FAILED: line ${line_no}: missing build script" >&2
      failures=$((failures + 1))
      if [[ "${keep_going}" -eq 0 ]]; then
        break
      fi
      continue
    fi
    if ! entry_build_script="$(resolve_build_script_path "${entry_build_script}")"; then
      echo "FAILED: line ${line_no}: build script not found: ${line_tokens[0]}" >&2
      failures=$((failures + 1))
      if [[ "${keep_going}" -eq 0 ]]; then
        break
      fi
      continue
    fi
  else
    entry_build_script="${build_script}"
    script_args=("${line_tokens[@]}")
  fi
  log_prefix="${log_dir}/${entry_id}"

  echo "==> [${entry_id}] line ${line_no}: $(basename -- "${entry_build_script}") ${script_args[*]}"

  # Execute each matrix entry in a fresh login shell so module operations and
  # sourced build scripts do not leak state across compiler configurations.
  # Positional arguments are marshaled across the bash -lc boundary with a
  # delimiter because script_args and perf_args are both variable length.
  if ! THROUGHPUT_FACTORS="${throughput_factors[*]}" bash -lc '
    set -euo pipefail
    repo_root="$0"
    build_script="$1"; shift
    jobs="$1"; shift
    configure_only="$1"; shift
    build_only="$1"; shift
    skip_existing="$1"; shift
    run_cmd="$1"; shift
    throughput="$1"; shift
    throughput_base_mem="$1"; shift
    throughput_outdir="$1"; shift
    delim="$1"; shift

    script_args=()
    while [[ $# -gt 0 && "$1" != "$delim" ]]; do
      script_args+=("$1")
      shift
    done
    if [[ $# -gt 0 && "$1" == "$delim" ]]; then
      shift
    fi
    perf_args=("$@")

    run_raja_perf() {
      # Treat the user-supplied run command as a launcher prefix such as srun,
      # then append the RAJAPerf executable and shell-quoted arguments.
      local quoted=()
      local a q
      for a in "$@"; do
        printf -v q "%q" "$a"
        quoted+=("$q")
      done
      eval "${run_cmd} ./bin/raja-perf.exe ${quoted[*]}"
    }

    cd "$repo_root"

    # Many existing lc-build scripts reference optional positional args (e.g. "$3")
    # without guarding for nounset. Disable it while sourcing for compatibility.
    #
    # If requested, probe for an existing configured build without deleting it.
    if [[ "$skip_existing" == "1" ]]; then
      build_preexisted=0

      # Some lc-build scripts always call rm/mkdir/cmake while setting up the
      # build dir. During the reuse probe, temporarily replace those commands so
      # we can discover the intended build directory without deleting anything or
      # re-running configuration.
      cmake() { :; }
      rm() { :; }
      mkdir() {
        local arg
        for arg in "$@"; do
          [[ "$arg" == -* ]] && continue
          if [[ -d "$arg" ]]; then
            build_preexisted=1
          fi
        done
        command mkdir -p "$@"
      }

      set +u
      # shellcheck disable=SC1090
      source "$build_script" "${script_args[@]}"
      set -u

      build_dir="$PWD"
      configured_present=0
      if [[ "$build_preexisted" == "1" && -f "${build_dir}/CMakeCache.txt" && ( -f "${build_dir}/Makefile" || -f "${build_dir}/build.ninja" ) ]]; then
        configured_present=1
      fi

      cd "$repo_root"
      unset -f cmake rm mkdir

      # If the probed directory already contains a configured build tree, reuse
      # it. Otherwise source the build script again normally to configure it.
      if [[ "$configured_present" == "1" ]]; then
        cd "$build_dir"
      else
        set +u
        # shellcheck disable=SC1090
        source "$build_script" "${script_args[@]}"
        set -u
      fi
    else
      set +u
      # shellcheck disable=SC1090
      source "$build_script" "${script_args[@]}"
      set -u
    fi

    if [[ "$configure_only" == "0" ]]; then
      cmake --build . --parallel "$jobs"
      if [[ "$build_only" == "0" ]]; then
        if [[ "$throughput" == "1" ]]; then
          # Throughput mode expands one logical entry into many RAJAPerf runs by
          # pairing each selected kernel with each requested factor. The kernel
          # selection options are peeled out of perf_args so the generated runs
          # can inject their own per-kernel outfile and memory size.
          kernels_to_run=()
          passthrough_args=()
          while [[ ${#perf_args[@]} -gt 0 ]]; do
            case "${perf_args[0]}" in
              --kernels|-k|--kernel)
                shift_count=1
                if [[ "${perf_args[0]}" == "--kernels" ]]; then
                  shift_count=1
                fi
                perf_args=("${perf_args[@]:${shift_count}}")
                while [[ ${#perf_args[@]} -gt 0 && "${perf_args[0]}" != -* ]]; do
                  kernels_to_run+=("${perf_args[0]}")
                  perf_args=("${perf_args[@]:1}")
                done
                ;;
              *)
                passthrough_args+=("${perf_args[0]}")
                perf_args=("${perf_args[@]:1}")
                ;;
            esac
          done
          if [[ ${#kernels_to_run[@]} -eq 0 ]]; then
            kernels_to_run=("all")
          fi

          factors=()
          throughput_factor_text="${THROUGHPUT_FACTORS:-}"
          throughput_factor_text="${throughput_factor_text//,/ }"
          read -r -a factors <<<"${throughput_factor_text}"
          if [[ ${#factors[@]} -eq 0 ]]; then
            factors=(1 4 16 32 64 128 256 512 1024)
          fi

          for kernel_name in "${kernels_to_run[@]}"; do
            for factor in "${factors[@]}"; do
              # Keep output filenames predictable so downstream merge/plot tools
              # can recover the factor directly from the CSV path.
              mem=$(( factor * throughput_base_mem ))
              outfile_kernel="${kernel_name//\//_}"
              echo "Running throughput: kernel=${kernel_name} factor=${factor} memory=${mem}"
              generated_args=(
                --npasses 1
                --npasses-combiners Average Minimum Maximum
                --outdir "$throughput_outdir"
                --outfile "${outfile_kernel}_factor_${factor}"
                --memory-allocated "$mem"
              )
              if [[ "$kernel_name" != "all" ]]; then
                generated_args=(-k "$kernel_name" "${generated_args[@]}")
              fi
              run_raja_perf "${generated_args[@]}" "${passthrough_args[@]}"
            done
          done
        else
          run_raja_perf "${perf_args[@]}"
        fi
      fi
    fi
  ' "${repo_root}" "${entry_build_script}" "${jobs}" "${configure_only}" "${build_only}" "${skip_existing}" "${run_cmd}" \
      "${throughput}" "${throughput_base_mem}" "${throughput_outdir}" \
      "__RAJAPERF_MATRIX_DELIM__" "${script_args[@]}" "__RAJAPERF_MATRIX_DELIM__" "${warmup_args[@]}" "${kernel_args[@]}" "${run_args[@]}" \
      </dev/null \
      2>&1 | tee "${log_prefix}.log"; then
    echo "FAILED: line ${line_no}: ${line}" >&2
    failures=$((failures + 1))
    if [[ "${keep_going}" -eq 0 ]]; then
      break
    fi
  fi
  echo
done < "${list_file}"

echo "Done. Total entries: ${total}; failures: ${failures}"
if [[ "${failures}" -ne 0 ]]; then
  exit 1
fi
