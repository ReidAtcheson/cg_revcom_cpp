#include <vector>
#include <random>
#include <cstdint>

#include "cg.h"
#include "power.h"
#include "sparse.h"


int main(int argc,char** argv){
  std::mt19937 rng(23947);
  int64_t nrows=200000;
  int64_t spread=32;
  int64_t nnz_per_row=10;
  int64_t maxiter = 50;
  double eps = 1.0;

  sparse_t<int64_t,double> A(nrows,spread,nnz_per_row,eps,rng);
  std::vector<double> b(nrows,1.0);
  std::vector<double> r(nrows,0.0);

  auto x = cg_plain(nrows,
      [&A](std::span<const double> in, std::span<double> out){
        A.matvec(in,out);
      },std::span<const double>(b),maxiter);

  auto [lambda,y] = power(nrows,[&A](std::span<const double> in, std::span<double> out){
        A.matvec(in,out);
      },
      std::span<const double>(b),
      maxiter);

  /*primitive check that Ax=b.*/
  A.matvec(x,r);
  for(size_t i=0;i<nrows;i++){
    assert(std::abs(r[i]-b[i])/std::abs(b[i]) < 1e-6);
  }
  /*
   * Primitive check that Ay = lambda * y.
   *
   * Power iteration converges slowly so I use a really
   * weak check here.
   */
  A.matvec(y,r);
  for(size_t i=0;i<nrows;i++){
    double relerr = std::abs(lambda*y[i]-r[i])/std::abs(lambda*y[i]);
    if(std::abs(y[i])>1e-2 && relerr>1e-2){
      std::print("{}  ,   {},    {}\n",lambda*y[i],r[i],relerr);
    }
    if(std::abs(y[i])>1e-2){
      assert(relerr<1e-2);
    }
  }




  return 0;
}
