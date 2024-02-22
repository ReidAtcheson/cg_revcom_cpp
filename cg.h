#ifndef __CG_H_
#define __CG_H_
#include <vector>
#include <span>
#include <print>
#include <cassert>

template<typename T,typename Lambda>
std::vector<T> cg_plain(size_t nrows,Lambda A, std::span<const T> b,size_t maxiter){
  bool verbose = false;
  auto dot = [](std::span<const T> x, std::span<const T> y) -> T{
    assert(x.size() == y.size());
    T out = 0.0;
    for(size_t i=0;i<x.size();i++){
      out += x[i]*y[i];
    }
    return out;
  };
  std::vector<T> x(nrows,0.0);
  std::vector<T> r(nrows,0.0);
  std::vector<T> p(nrows,0.0);
  std::vector<T> q(nrows,0.0);
  /*Calculate initial residual.*/
  A(x,r);
  for(size_t i=0;i<r.size();i++){
    r[i] = b[i] - r[i];
  }
  std::copy(r.begin(),r.end(),p.begin());
  T beta = 0.0;
  T rho_last = 0.0;
  T rho_current = 0.0;
  for(size_t it=0;it<maxiter;it++){
    rho_current = dot(r,r);
    if(verbose){
      std::print("iteration: {}, residual: {}\n",it,std::sqrt(rho_current));
    }

    if(it>0){
      beta =  rho_current/rho_last;
      for(size_t i=0;i<p.size();i++){
        p[i] = r[i] + beta*p[i];
      }
    }
    A(p,q);
    T alpha = rho_current / dot(p,q);
    /*Update.*/
    for(size_t i=0;i<x.size();i++){
      x[i] = x[i] + alpha*p[i];
      r[i] = r[i] - alpha*q[i];
    }
    rho_last = rho_current;
  }
  return x;
}



#endif
