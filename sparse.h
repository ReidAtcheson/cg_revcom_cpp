#ifndef __SPARSE_H_
#define __SPARSE_H_
#include <vector>
#include <random>
#include <map>
#include <cassert>
#include <thread>
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
      assert(nrows>0);
      assert(nnz_per_row>0);
      assert(eps>0.0);
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
          I c = std::max(I(0),std::min(nrows - 1,I(d(rng) + r)));
          F val = v(rng);
          A[{r,c}] = val;
          A[{c,r}] = val;
        }
        /*Zero out the diagonal as I will modify this
         * later to ensure positive definiteness.*/
        A[{r,r}] = 0.0;
      }


      /*Now construct the compressed-sparse-row format.*/
      I off = 0;
      this->offs.push_back(off);
      for(I r = 0;r < nrows; r++){
        /*First determine the diagonal value.*/
        F diag=eps;
        for(auto it = A.lower_bound({r,0}); it != A.upper_bound({r,nrows-1}); it++){
          diag += std::abs(it->second);
        }
        A[{r,r}] = diag;
        /*Now add to CSR matrix.*/
        for(auto it = A.lower_bound({r,0}); it != A.upper_bound({r,nrows-1}); it++){
          off+=1;
          this->vals.push_back(it->second);
          this->cids.push_back(it->first.second);
        }
        this->offs.push_back(off);
      }

      this->nrows = nrows;
      this->ncols = nrows;

      /*Some post-conditions.*/
      /*Symmetric matrix. nrows==ncols.*/
      assert(this->nrows == this->ncols);
      /*Offsets array should be sorted.*/
      assert(std::is_sorted(this->offs.begin(),this->offs.end()));
      /*Offsets array should have nrows+1 offsets.*/
      assert(this->offs.size() == size_t(this->nrows+1));
      /*Column indices should be sorted for each row.*/
      for(size_t i = 0;i < this->offs.size()-1; i++){
        I beg = this->offs[i];
        I end = this->offs[i+1];
        assert(std::is_sorted(&this->cids[beg], &this->cids[end]));
      }
      /*Last offset value should equal total number of nonzeros.*/
      assert(size_t(this->offs[this->offs.size()-1]) == this->vals.size());
      /*Finally: check symmetry.*/
      std::map<std::pair<I,I>,F> check_symmetry;
      for(I r =0;r<nrows;r++){
        I beg = this->offs[r];
        I end = this->offs[r+1];
        for(I coff = beg; coff < end; coff++){
          I c = this->cids[coff];
          /*Indices in CSR format should be unique and only occur once in a full scan
           * of the datastructure.*/
          assert( check_symmetry.find({r,c}) == check_symmetry.end() );
          check_symmetry[{r,c}] = this->vals[coff];
        }
      }
      for(auto& [rc,val] : check_symmetry){
        auto [r,c] = rc;
        F val0 = check_symmetry[{r,c}];
        F val1 = check_symmetry[{c,r}];
        assert(val0 == val1);
      }
    }


    void matvec(std::span<const F> in,std::span<F> out) const{
      assert(in.size() == this->ncols);
      assert(out.size() == this->nrows);
      for(I r=0;r<this->nrows;r++){
        I beg = this->offs[r];
        I end = this->offs[r+1];
        out[r]=0.0;
        for(I coff = beg; coff < end; coff++){
          I c = this->cids[coff];
          F v = this->vals[coff];
          out[r] += v * in[c];
        }
      }
    }

  void matvec_batch2(
      std::span<const F> in0,
      std::span<const F> in1,
      std::span<F> out0,
      std::span<F> out1
      ) const{
        assert(in0.size() == this->ncols);
        assert(out0.size() == this->nrows);
        assert(in1.size() == this->ncols);
        assert(out1.size() == this->nrows);
        for(I r=0;r<this->nrows;r++){
          I beg = this->offs[r];
          I end = this->offs[r+1];
          out0[r]=0.0;
          out1[r]=0.0;
          for(I coff = beg; coff < end; coff++){
            I c = this->cids[coff];
            F v = this->vals[coff];
            out0[r] += v * in0[c];
            out1[r] += v * in1[c];
          }
        }
      }

  void matvec_batch2_multithreaded(
      std::span<const F> in0,
      std::span<const F> in1,
      std::span<F> out0,
      std::span<F> out1
      ) const{
        assert(in0.size() == this->ncols);
        assert(out0.size() == this->nrows);
        assert(in1.size() == this->ncols);
        assert(out1.size() == this->nrows);
        std::thread first_half([&](){
          for(I r=0;r<this->nrows/2;r++){
            I beg = this->offs[r];
            I end = this->offs[r+1];
            out0[r]=0.0;
            out1[r]=0.0;
            for(I coff = beg; coff < end; coff++){
              I c = this->cids[coff];
              F v = this->vals[coff];
              out0[r] += v * in0[c];
              out1[r] += v * in1[c];
            }
          }
      });

      std::thread second_half([&](){
        for(I r=this->nrows/2;r<this->nrows;r++){
          I beg = this->offs[r];
          I end = this->offs[r+1];
          out0[r]=0.0;
          out1[r]=0.0;
          for(I coff = beg; coff < end; coff++){
            I c = this->cids[coff];
            F v = this->vals[coff];
            out0[r] += v * in0[c];
            out1[r] += v * in1[c];
          }
        }
    });

    first_half.join();
    second_half.join();

  }



};


#endif
