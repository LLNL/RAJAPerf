#include "common/Benchmark.hxx"

#include "add.hxx"

int main(int argc, char* argv[])
{

  std::vector<RAJA::Benchmark*> variants;

  variants.push_back(new AddBenchmark(100, 10));
#if defined(ADD_RAJA_SEQ)
#endif

  for (benchmark : variants) {
    benchmark->execute();
  }
}
