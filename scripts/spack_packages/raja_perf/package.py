# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import socket
import glob

from spack.package import *
from spack.pkg.builtin.camp import hip_repair_cache


class RajaPerf(CMakePackage, CudaPackage, ROCmPackage):
    """RAJA Perf Suite Framework."""

    homepage = "http://software.llnl.gov/RAJAPerf/"
    git      = "https://github.com/LLNL/RAJAPerf.git"

    maintainers = ["richhornung"]

    version('develop', branch='develop', submodules='True')
    version('main',  branch='main',  submodules='True')
    version('0.12.0', tag='v0.12.0', submodules="True")
    version('0.11.0', tag='v0.11.0', submodules="True")
    version('0.10.0', tag='v0.10.0', submodules="True")
    version('0.9.0', tag='v0.9.0', submodules="True")
    version('0.8.0', tag='v0.8.0', submodules="True")
    version('0.7.0', tag='v0.7.0', submodules="True")
    version('0.6.0', tag='v0.6.0', submodules="True")
    version('0.5.2', tag='v0.5.2', submodules="True")
    version('0.5.1', tag='v0.5.1', submodules="True")
    version('0.5.0', tag='v0.5.0', submodules="True")
    version('0.4.0', tag='v0.4.0', submodules="True")

    variant('openmp', default=True, description='Build OpenMP backend')
    variant('openmp_target', default=False, description='Build with OpenMP target support')
    variant('shared', default=False, description='Build Shared Libs')
    variant('libcpp', default=False, description='Uses libc++ instead of libstdc++')
    variant('tests', default='basic', values=('none', 'basic', 'benchmarks'),
            multi=False, description='Tests to run')

    depends_on("blt")
    depends_on("blt@0.5.0:", type="build", when="@0.12.0:")
    depends_on("blt@0.4.1:", type="build", when="@0.11.0:")
    depends_on("blt@0.4.0:", type="build", when="@0.8.0:")
    depends_on("blt@0.3.0:", type="build", when="@:0.7.0")

    depends_on("cmake@3.20:", when="@0.12.0:", type="build")
    depends_on("cmake@3.23:", when="@0.12.0: +rocm", type="build")
    depends_on("cmake@3.14:", when="@:0.12.0:", type="build")

    depends_on("llvm-openmp", when="+openmp %apple-clang")

    depends_on("rocprim", when="+rocm")
 
    conflicts('+openmp', when='+rocm')
    conflicts('~openmp', when='+openmp_target', msg='OpenMP target requires OpenMP')

    def _get_sys_type(self, spec):
        sys_type = str(spec.architecture)
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    @property
    # TODO: name cache file conditionally to cuda and libcpp variants
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            hostname = hostname.rstrip("1234567890")
        return "{0}-{1}-{2}@{3}-{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            self.spec.dag_hash(8)
        )

    def initconfig_compiler_entries(self):
        spec = self.spec
        # Default entries are already defined in CachedCMakePackage, inherit them:
        entries = super(RajaPerf, self).initconfig_compiler_entries()

        # Switch to hip as a CPP compiler.
        # adrienbernede-22-11:
        #   This was only done in upstream Spack raja package.
        #   I could not find the equivalent logic in Spack source, so keeping it.
        if "+rocm" in spec:
            entries.insert(0, cmake_cache_path("CMAKE_CXX_COMPILER", spec["hip"].hipcc))

        # Override CachedCMakePackage CMAKE_C_FLAGS and CMAKE_CXX_FLAGS add
        # +libcpp specific flags
        flags = spec.compiler_flags

        # use global spack compiler flags
        cppflags = " ".join(flags["cppflags"])
        if cppflags:
            # avoid always ending up with " " with no flags defined
            cppflags += " "

        cflags = cppflags + " ".join(flags["cflags"])
        if "+libcpp" in spec:
            cflags += " ".join([cflags,"-DGTEST_HAS_CXXABI_H_=0"])
        if cflags:
            entries.append(cmake_cache_string("CMAKE_C_FLAGS", cflags))

        cxxflags = cppflags + " ".join(flags["cxxflags"])
        if "+libcpp" in spec:
            cxxflags += " ".join([cxxflags,"-stdlib=libc++ -DGTEST_HAS_CXXABI_H_=0"])
        if cxxflags:
            entries.append(cmake_cache_string("CMAKE_CXX_FLAGS", cxxflags))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(RajaPerf, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("ENABLE_OPENMP", "+openmp" in spec))

        if "+cuda" in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))

            if not spec.satisfies("cuda_arch=none"):
                cuda_arch = spec.variants["cuda_arch"].value
                entries.append(cmake_cache_string("CUDA_ARCH", "sm_{0}".format(cuda_arch[0])))
                entries.append(
                    cmake_cache_string("CMAKE_CUDA_ARCHITECTURES", "{0}".format(cuda_arch[0]))
                )
        else:
            entries.append(cmake_cache_option("ENABLE_CUDA", False))

        if "+rocm" in spec:
            entries.append(cmake_cache_option("ENABLE_HIP", True))
            entries.append(cmake_cache_path("HIP_ROOT_DIR", "{0}".format(spec["hip"].prefix)))
            hip_repair_cache(entries, spec)
            archs = self.spec.variants["amdgpu_target"].value
            if archs != "none":
                arch_str = ",".join(archs)
                entries.append(
                    cmake_cache_string("HIP_HIPCC_FLAGS", "--amdgpu-target={0}".format(arch_str))
                )
                entries.append(
                    cmake_cache_string("CMAKE_HIP_ARCHITECTURES", arch_str)
                )
        else:
            entries.append(cmake_cache_option("ENABLE_HIP", False))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        option_prefix = "RAJA_" if spec.satisfies("@0.11.0:") else ""

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        entries.append(cmake_cache_path("BLT_SOURCE_DIR", spec["blt"].prefix))

        # Build options
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Build Options")
        entries.append("#------------------{0}\n".format("-" * 60))

        entries.append(cmake_cache_string(
            "CMAKE_BUILD_TYPE", spec.variants["build_type"].value))
        entries.append(cmake_cache_option("BUILD_SHARED_LIBS", "+shared" in spec))
        if not self.run_tests and not "+tests" in spec:
            entries.append(cmake_cache_option("ENABLE_TESTS", False))
        else:
            entries.append(cmake_cache_option("ENABLE_TESTS", True))

        return entries

    def cmake_args(self):
        options = []
        return options

    @property
    def build_relpath(self):
        """Relative path to the cmake build subdirectory."""
        return join_path("..", self.build_dirname)

    @run_after("install")
    def setup_build_tests(self):
        """Copy the build test files after the package is installed to a
        relative install test subdirectory for use during `spack test run`."""
        # Now copy the relative files
        self.cache_extra_test_sources(self.build_relpath)

        # Ensure the path exists since relying on a relative path at the
        # same level as the normal stage source path.
        mkdirp(self.install_test_root)

    @property
    def _extra_tests_path(self):
        # TODO: The tests should be converted to re-build and run examples
        # TODO: using the installed libraries.
        return join_path(self.install_test_root, self.build_relpath, "bin")
