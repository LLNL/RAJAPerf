#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
truehostname=${hostname//[0-9]/}
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}
job_unique_id=${CI_JOB_ID:-""}
raja_version=${UPDATE_RAJA:-""}

sys_type=${SYS_TYPE:-""}
py_env_path=${PYTHON_ENVIRONMENT_PATH:-""}

# Dependencies
date
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt=""

    if [[ -d /dev/shm ]]
    then
        prefix="/dev/shm/${hostname}"
        if [[ -z ${job_unique_id} ]]; then
          job_unique_id=manual_job_$(date +%s)
          while [[ -d ${prefix}/${job_unique_id} ]] ; do
              sleep 1
              job_unique_id=manual_job_$(date +%s)
          done
        fi

        prefix="${prefix}/${job_unique_id}"
        mkdir -p ${prefix}
        prefix_opt="--prefix=${prefix}"
    fi

    python3 tpl/RAJA/scripts/uberenv/uberenv.py --project-json=".uberenv_config.json" --spec="${spec}" ${prefix_opt}

    mv ${project_dir}/tpl/RAJA/hc-*.cmake ${project_dir}/.

fi
date

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"hc-*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/hc-*.cmake"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/hc-*.cmake"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

# Build Directory
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

build_dir="${build_root}/build_${hostconfig//.cmake/}"

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ Host-config: ${hostconfig_path}"
    echo "~ Build Dir:   ${build_dir}"
    echo "~ Project Dir: ${project_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~ ENV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building RAJA PerfSuite"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Map CPU core allocations
    declare -A core_counts=(["lassen"]=40 ["ruby"]=28 ["corona"]=32)

    # If using Multi-project, set up the submodule
    if [[ -n ${raja_version} ]]
    then
      cd tpl/RAJA  
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      echo "~~~~ Updating RAJA Submodule to develop ~~~"
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      git pull origin develop
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      echo "~~~~ Updating Submodules within RAJA ~~~~~~"
      echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
      git submodule update --init --recursive
      cd -
    fi

    # If building, then delete everything first
    # NOTE: 'cmake --build . -j core_counts' attempts to reduce individual build resources.
    #       If core_counts does not contain hostname, then will default to '-j ', which should
    #       use max cores.
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    date

    if [[ "${truehostname}" == "corona" ]]
    then
        module unload rocm
    fi

    cmake \
      -C ${hostconfig_path} \
      ${project_dir}
    if echo ${spec} | grep -q "intel" ; then
        cmake --build . -j 16
        echo "~~~~~~~~~ Build Command: ~~~~~~~~~~~~~~~~~~~~~"
        echo "cmake --build . -j 16"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    else
        cmake --build . -j ${core_counts[$truehostname]}
        echo "~~~~~~~~~ Build Command: ~~~~~~~~~~~~~~~~~~~~~"
        echo "cmake --build . -j ${core_counts[$truehostname]}"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    fi
    date
fi

if [[ ! -d ${build_dir} ]]
then
    echo "ERROR: Build directory not found : ${build_dir}" && exit 1
fi

cd ${build_dir}

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ TESTING RAJAPERF SUITE"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

if grep -q -i "ENABLE_TESTS.*ON" ${hostconfig_path}
then

    #
    # Maintaining separate, but identical release and debug sections 
    # in case we want to make them disctinct in the future.
    #

    if echo ${sys_type} | grep -q "blueos" && echo ${spec} | grep -q "cuda" ; then
        if grep -q -i "CMAKE_BUILD_TYPE.*Release" ${hostconfig_path}
        then
            echo "~~~~~~~~~ Run Command: ~~~~~~~~~~~~~~~~~~~~~"
            echo "lrun -n1 ... ctest --output-on-failure -T test"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            lrun -n1 --smpiargs="-disable_gpu_hooks" ctest --output-on-failure -T test
        else
            echo "~~~~~~~~~ Run Command: ~~~~~~~~~~~~~~~~~~~~~"
            echo "lrun -n1 ... ctest --output-on-failure -T test"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            lrun -n1 --smpiargs="-disable_gpu_hooks" ctest --output-on-failure -T test
        fi
    else
        if grep -q -i "CMAKE_BUILD_TYPE.*Release" ${hostconfig_path}
        then
            echo "~~~~~~~~~ Run Command: ~~~~~~~~~~~~~~~~~~~~~"
            echo "ctest --output-on-failure -T test 2>&1 | tee tests_output.txt"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            ctest --output-on-failure -T test 2>&1 | tee tests_output.txt
        else
            echo "~~~~~~~~~ Run Command: ~~~~~~~~~~~~~~~~~~~~~"
            echo "ctest --output-on-failure -T test 2>&1 | tee tests_output.txt"
            echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            ctest --output-on-failure -T test 2>&1 | tee tests_output.txt
        fi
    fi

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    xsltproc -o junit.xml ${project_dir}/blt/tests/ctest-to-junit.xsl Testing/*/Test.xml
    mv junit.xml ${project_dir}/junit.xml

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ CLEAN UP"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
make clean

