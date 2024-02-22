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
    sparse_t(I nrows,I spread,I nnz_per_row,F eps,std::mt19937& rng){
      /*Hold initially in an ordered map just to make things easy.*/
      std::map<std::pair<I,I>,F> A;

      /*The distributions that will pull the column indices and nonzero values.*/
      double mean=0.0;
      double std=spread;
      std::normal_distribution<> d(mean,std);
      std::uniform_real_distribution<> v(-1.0,1.0);

      /*user inputs nnz_per_row but since we are symmetrizing this it will roughly
       * double this value so I just half it here.*/
      nnz_per_row/=2;


      for(I r = 0; r < nrows; r++){
        for(I nz = 0; nz < nnz_per_row; nz++){
          /*Make sure we don't get indices below 0 or above nrows.*/
          I c = std::max(0,std::min(nrows - 1,d(rng) + r));
          F val = v(rng);
          A[{r,c}] = val;
          A[{c,r}] = val;
        }
        /*Zero out the diagonal as I will modify this
         * later to ensure positive definiteness.*/
        A[{r,r}] = 0.0;
      }
    }
};


#endif
