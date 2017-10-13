#!/bin/bash

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

source ~/.bashrc
cd ${TRAVIS_BUILD_DIR}
or_die mkdir travis-build
cd travis-build
if [[ "$DO_BUILD" == "yes" ]] ; then
    or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} ../
    cat CMakeCache.txt
    or_die make -j 3 VERBOSE=1
    if [[ "${DO_TEST}" == "yes" ]] ; then
        or_die ./bin/raja-perf.exe --checkrun
    fi
fi

exit 0
