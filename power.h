#ifndef __POWER_H_
#define __POWER_H_
#include <print>
#include <span>

template<typename T,typename Lambda>
std::pair<T,std::vector<T>> power(size_t nrows,Lambda A,std::span<const T> x,size_t maxiter){
  bool verbose = false;
  auto dot = [](std::span<const T> x, std::span<const T> y) -> T{
    assert(x.size() == y.size());
    T out = 0.0;
    for(size_t i=0;i<x.size();i++){
      out += x[i]*y[i];
    }
    return out;
  };
  std::vector<T> y(x.begin(),x.end());
  std::vector<T> Ay(y.size(),0.0);
  T lambda = 0.0;
  for(size_t it=0;it<maxiter;it++){
    A(y,Ay);
    lambda = dot(y,Ay);
    if(verbose){
      std::print("iteration: {},  lambda = {}\n",it,lambda);
    }
    T normAy = std::sqrt(dot(Ay,Ay));
    for(size_t i=0;i<Ay.size();i++){
      Ay[i]/=normAy;
    }
    std::copy(Ay.begin(),Ay.end(),y.begin());
  }
  return {lambda,Ay};
}


#endif
