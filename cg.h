#ifndef __CG_H_
#define __CG_H_
#include <span>
#include <mdspan>

template<typename Lambda>
void cg_plain(size_t nrows,size_t ncols,Lambda A, std::span<double> b, std::span<double> r, std::span<double> p){
}



#endif
