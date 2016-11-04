#include "RAJA/RAJA.hxx"

#include <random>

namespace RAJA {
  namespace perfsuite {
    struct raw {};
  }
}

template <typename POLICY, typename NUMERIC_T>
class AddKernel {
public:
  virtual void SetUp(const size_t size) {
      this->size = size;
      x = new NUMERIC_T[size];
      y = new NUMERIC_T[size];

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 25);

      for (int i = 0; i < size; ++i) {
        x[i] = dist(e2);
        y[i] = dist(e2);
      }
  }

  virtual void Execute()
  {
    for( int i = 0; i < size; ++i) {
      y[i] = x[i] + y[i];
    }
  }

  virtual void TearDown()
  {
      delete[] x;
      delete[] y;
  }

  virtual void Run(const size_t size){
    SetUp(size);
    Execute();
    TearDown();
  }

protected:
  NUMERIC_T* x;
  NUMERIC_T* y;

  size_t size;
};
