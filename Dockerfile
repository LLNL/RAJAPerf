###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM axom/compilers:gcc-5 AS gcc5
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN ls
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_WARNINGS=On -DENABLE_OPENMP=On -DRAJA_DEPRECATED_TESTS=On ..
RUN cd build && make -j 16
RUN cd build && ./bin/raja-perf.exe

FROM axom/compilers:gcc-5 AS gcc5-debug
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DENABLE_WARNINGS=On -DENABLE_COVERAGE=On -DENABLE_OPENMP=On ..
RUN cd build && make -j 16

FROM axom/compilers:gcc-6 AS gcc6
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_WARNINGS=On -DENABLE_OPENMP=On -DRAJA_ENABLE_RUNTIME_PLUGINS=On ..
RUN cd build && make -j 16
RUN cd build && ./bin/raja-perf.exe

FROM axom/compilers:gcc-7 AS gcc7
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_WARNINGS=On -DENABLE_OPENMP=On ..
RUN cd build && make -j 16
RUN cd build && ./bin/raja-perf.exe

FROM axom/compilers:gcc-8 AS gcc8
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_WARNINGS=On -DENABLE_OPENMP=On -DRAJA_ENABLE_BOUNDS_CHECK=ON ..
RUN cd build && make -j 16
RUN cd build && ./bin/raja-perf.exe

FROM axom/compilers:clang-9 AS clang9
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-fmodules -DENABLE_OPENMP=On ..
RUN cd build && make -j 16
RUN cd build && ./bin/raja-perf.exe

FROM axom/compilers:clang-9 AS clang9-debug
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=On -DCMAKE_CXX_FLAGS=-fsanitize=address ..
RUN cd build && make -j 16

FROM axom/compilers:nvcc-10.2 AS nvcc10
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 ..
RUN cd build && make -j 2

FROM axom/compilers:nvcc-10.2 AS nvcc10-debug
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 ..
RUN cd build && make -j 2

FROM axom/compilers:rocm AS hip
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
ENV HCC_AMDGPU_TARGET=gfx900
RUN mkdir build && cd build && cmake -DROCM_ROOT_DIR=/opt/rocm/include -DHIP_RUNTIME_INCLUDE_DIRS="/opt/rocm/include;/opt/rocm/hip/include" -DENABLE_HIP=On -DENABLE_OPENMP=Off -DENABLE_CUDA=Off -DENABLE_WARNINGS_AS_ERRORS=Off -DHIP_HIPCC_FLAGS=-fPIC ..
RUN cd build && make -j 16

FROM axom/compilers:oneapi AS sycl
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN /bin/bash -c "source /opt/intel/inteloneapi/setvars.sh && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=dpcpp -DENABLE_SYCL=On .."
RUN /bin/bash -c "source /opt/intel/inteloneapi/setvars.sh && cd build && make -j 16"
