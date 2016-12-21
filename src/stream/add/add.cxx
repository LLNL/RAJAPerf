#include "add.hxx"

AddBenchmark::AddBenchmark(size_t size, size_t iterations):
  RAJA::Benchmark(size, iterations)
{
}

AddBenchmark::~AddBenchmark()
{
}

void AddBenchmark::setUp()
{
  x = new double[size];
  y = new double[size];

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 25);

  for (int i = 0; i < size; ++i) {
    x[i] = dist(e2);
    y[i] = dist(e2);
  }
}

void AddBenchmark::executeBenchmark()
{
  for( int i = 0; i < size; ++i) {
    y[i] = x[i] + y[i];
  }
}

void AddBenchmark::tearDown()
{
  // delete[] x;
  // delete[] y;
}
