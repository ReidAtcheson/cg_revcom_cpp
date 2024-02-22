#include <random>
#include <cstdint>

#include "cg.h"
#include "sparse.h"


int main(int argc,char** argv){
  std::mt19937 rng(23947);
  int64_t nrows=1024;
  int64_t spread=32;
  int64_t nnz_per_row=10;
  double eps = 1e-4;

  sparse_t<int,float> A(nrows,spread,nnz_per_row,eps,rng);

  return 0;
}
