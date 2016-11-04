#include "add.hxx"

int main(int argc, char* argv[])
{
  AddKernel<RAJA::perfsuite::raw, float>().Run(101);
}
