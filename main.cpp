#include <vector>
#include <random>
#include <cstdint>

#include "cg.h"
#include "sparse.h"


int main(int argc,char** argv){
  std::mt19937 rng(23947);
  int64_t nrows=1024;
  int64_t spread=32;
  int64_t nnz_per_row=10;
  int64_t maxiter = 1000;
  double eps = 1e-4;

  sparse_t<int64_t,double> A(nrows,spread,nnz_per_row,eps,rng);
  std::vector<double> b(nrows,1.0);

  cg_plain(nrows,
      [&A](std::span<const double> in, std::span<double> out){
        A.matvec(in,out);
      },std::span<const double>(b),maxiter);

  return 0;
}
