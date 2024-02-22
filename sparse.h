#ifndef __SPARSE_H_
#define __SPARSE_H_
#include <vector>
#include <random>
#include <map>


/**
 * 
 *
 * Minimal implementation of some sparse functionality
 * just for the purposes of testing the ideas in the repo.
 *
 */


template<typename I,typename F>
class sparse_t{
  private:
    I nrows;
    I ncols;
    std::vector<I> cids;
    std::vector<I> offs;
    std::vector<F> vals;
  public:
    sparse_t() = default;
    /*Randomly generate a sparse symmetric positive definite matrix.*/
    sparse_t(I nrows,I ncols,I spread,I nnz_per_row,F eps,std::mt19937& rng){
      std::map<std::pair<I,I>,F> A;
      double mean=0.0;
      double std=spread;
      std::normal_distribution<> d(mean,std);
      for(I r = 0; r < nrows; r++){
      }
    }
};


#endif
