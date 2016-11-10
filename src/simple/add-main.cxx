#include "common/Benchmark.hxx"

#include "add.hxx"

int main(int argc, char* argv[])
{

  RAJA::Benchmark* add = new AddBenchmark(100, 10);
  add->execute();

}
