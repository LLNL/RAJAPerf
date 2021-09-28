from spack import *


class Hip(CMakePackage):
    """HIP is a C++ Runtime API and Kernel Language that allows developers to
       create portable applications for AMD and NVIDIA GPUs from
       single source code."""

    homepage = "https://github.com/ROCm-Developer-Tools/HIP"
    url      = "https://github.com/ROCm-Developer-Tools/HIP/archive/refs/tags/rocm-4.0.0.tar.gz"

    maintainers = ['srekolam', 'arjun-raj-kuppala']

    version('4.1.0', sha256='25ad58691456de7fd9e985629d0ed775ba36a2a0e0b21c086bd96ba2fb0f7ed1')
    version('4.0.0', sha256='0082c402f890391023acdfd546760f41cb276dffc0ffeddc325999fd2331d4e8')

    depends_on('cmake@3:', type='build')
    depends_on('perl@5.10:', type=('build', 'run'))
    depends_on('mesa~llvm@18.3:')

    for ver in ['4.0.0', '4.1.0']:
        depends_on('rocclr@' + ver,  type='build', when='@' + ver)
        depends_on('hsakmt-roct@' + ver, type='build', when='@' + ver)
        depends_on('hsa-rocr-dev@' + ver, type='link', when='@' + ver)
        depends_on('comgr@' + ver, type='build', when='@' + ver)
        depends_on('llvm-amdgpu@' + ver, type='build', when='@' + ver)
        depends_on('rocm-device-libs@' + ver, type='build', when='@' + ver)
        depends_on('rocminfo@' + ver, type='build', when='@' + ver)

    def setup_dependent_package(self, module, dependent_spec):
        self.spec.hipcc = join_path(self.prefix.bin, 'hipcc')

    @run_before('install')
    def filter_sbang(self):
        perl = self.spec['perl'].command
        kwargs = {'ignore_absent': False, 'backup': False, 'string': False}

        with working_dir('bin'):
            match = '^#!/usr/bin/perl'
            substitute = "#!{perl}".format(perl=perl)
            files = [
                'hipify-perl', 'hipcc', 'extractkernel',
                'hipconfig', 'hipify-cmakefile'
            ]
            filter_file(match, substitute, *files, **kwargs)

    def cmake_args(self):
        args = [
            '-DHIP_COMPILER=clang',
            '-DHIP_PLATFORM=rocclr',
            '-DHSA_PATH={0}'.format(self.spec['hsa-rocr-dev'].prefix),
            '-DLIBROCclr_STATIC_DIR={0}/lib'.format(self.spec['rocclr'].prefix)
        ]
        return args

