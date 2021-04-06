#ifndef RAJAPERFSUITE_QUICKKERNELBASE_HPP
#define RAJAPERFSUITE_QUICKKERNELBASE_HPP

#include "KernelBase.hpp"
#include <utility>

namespace rajaperf {

    struct SureBuddyOkay {
        bool validate_checksum(double reference, double variant) {
            return true;
        }
    };

    template<typename SetUp, typename Execute, typename Checksum = SureBuddyOkay>
    class QuickKernelBase : public rajaperf::KernelBase {
        SetUp m_setup;
        Execute m_execute;
        Checksum m_checksum;
        using runData = decltype(m_setup(0, 0));
        runData rd;
    public:
        QuickKernelBase(std::string &name, const RunParams &params, SetUp se, Execute ex, Checksum ch) : KernelBase(
                name,
                params),
                                                                                                         m_setup(se),
                                                                                                         m_execute(ex),
                                                                                                         m_checksum(
                                                                                                                 ch) {}

        QuickKernelBase(std::string &name, const RunParams &params, SetUp se, Execute ex) : KernelBase(name,
                                                                                                       params),
                                                                                            m_setup(se),
                                                                                            m_execute(ex),
                                                                                            m_checksum(
                                                                                                    SureBuddyOkay()) {}

        Real_type m_y;

        void setUp(VariantID vid) override { rd = m_setup(0, 0); }

        void updateChecksum(VariantID vid) override {
            checksum[vid] += m_y;
        }

        void tearDown(VariantID vID) override {}

        void runSeqVariant(VariantID vID) override {}

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
        void runOpenMPVariant(VariantID vid) override {
           auto size = getRunSize();
           for(int x =0; x< getRunReps(); ++x){
              m_execute(x, size)
           }
        }
#endif
#if defined(RAJA_ENABLE_CUDA)
        void runCudaVariant(VariantID vid) override {}
#endif
#if defined(RAJA_ENABLE_HIP)
        void runHipVariant(VariantID vid) override {}
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
        void runOpenMPTargetVariant(VariantID vid) override {}
#endif

#if defined(RUN_KOKKOS)
        using index_seq = std::make_index_sequence<std::tuple_size<runData>::value>;

        template<size_t... Is>
        void rkv_helper(std::index_sequence<Is...>) {
            auto size = getRunSize();
            for (int x = 0; x < getRunReps(); ++x) {
                m_execute(x, size, std::get<Is>(rd)...);
            }
        }

        void runKokkosVariant(VariantID vid) override {
            rkv_helper(index_seq());
        }

#endif // RUN_KOKKOS
    };

    template<class... Lambdas>
    KernelBase *make_kernel_base(std::string name, const RunParams &params, Lambdas... lambdas) {
        return new QuickKernelBase<Lambdas...>(name, params, lambdas...);
    }

} // end namespace rajaperf
#endif //RAJAPERFSUITE_QUICKKERNELBASE_HPP
