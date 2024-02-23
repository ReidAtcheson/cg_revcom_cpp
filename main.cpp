#include <vector>
#include <random>
#include <cstdint>
#include <thread>

#include "cg.h"
#include "power.h"
#include "sparse.h"
#include "timer.h"

template<typename I,typename T>
void plain_sequential(
    int64_t nrows,
    int64_t maxiter,
    const sparse_t<I,T>& A,
    const std::vector<T>& b){
  std::vector<double> r(nrows,0.0);
  {
    scope_timer_t timer("plain sequential");
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
  }

}

template<typename I,typename T>
void plain_multithreaded(
    int64_t nrows,
    int64_t maxiter,
    const sparse_t<I,T>& A,
    const std::vector<T>& b){
  std::vector<double> r(nrows,0.0);
  {
    scope_timer_t timer("plain multithreaded");

    std::vector<double> x;
    double lambda;
    std::vector<double> y;
    {
      std::thread run_cg([&](){
        auto tmp_x = cg_plain(nrows,
            [&A](std::span<const double> in, std::span<double> out){
              A.matvec(in,out);
            },std::span<const double>(b),maxiter);
        x = tmp_x;
      });

      std::thread run_power([&](){
      auto [tmp_lambda,tmp_y] = power(nrows,[&A](std::span<const double> in, std::span<double> out){
            A.matvec(in,out);
          },
          std::span<const double>(b),
          maxiter);
      lambda = tmp_lambda;
      y = tmp_y;
      });

      run_cg.join();
      run_power.join();
    }

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
  }
}

template<typename I,typename T>
void plain_revcomm(
    int64_t nrows,
    int64_t maxiter,
    const sparse_t<I,T>& A,
    const std::vector<T>& b){
  std::vector<double> r(nrows,0.0);
  {
    scope_timer_t timer("plain reverse communication");

    std::vector<double> x;
    for(auto deferred_op : cg_revcomm(nrows, std::span<const double>(b), maxiter)){

      std::visit([&](auto&& arg){
            using DT = std::decay_t<decltype(arg)>;
            /*CG has requested ("reverse-communicated") a sparse matrix-vector product
             * so now we evaluate that here.*/
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              auto [in, out] = arg;
              A.matvec(in,out);
            }
            /*Algorithm has produced an answer which we capture.*/
            else{
              auto [tmp_x] = arg;
              x = tmp_x;
            }
          }, deferred_op);
    }
    double lambda;
    std::vector<double> y;
    for(auto deferred_op : power_revcomm(nrows, std::span<const double>(b), maxiter)){
      std::visit([&](auto&& arg){
                using DT = std::decay_t<decltype(arg)>;
                /*Power iteration has requested ("reverse-communicated") a sparse matrix-vector product
                 * so now we evaluate that here.*/
                if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                  auto [in, out] = arg;
                  A.matvec(in,out);
                }
                /*Algorithm has produced an answer which we capture.*/
                else{
                  auto [result] = arg;
                  auto [tmp_lambda,tmp_y] = result;
                  y = tmp_y;
                  lambda = tmp_lambda;
                }
              }, deferred_op);
    }
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
  }

}

template<typename I,typename T>
void batched_revcomm(
    int64_t nrows,
    int64_t maxiter,
    const sparse_t<I,T>& A,
    const std::vector<T>& b){
  std::vector<double> r(nrows,0.0);
  {
    scope_timer_t timer("batched reverse communication");

    /**
     * The idea here is instead of simply sequentially iterating both coroutines
     * to completion we can co-iterate them and batch up any matching
     * matrix-vector products involving `A`, greatly reducing
     * the redundant memory reads while still keeping
     * the algorithms logically separate.
     */

    auto cg_eval = cg_revcomm(nrows, std::span<const double>(b),maxiter);
    auto power_eval = power_revcomm(nrows, std::span<const double>(b),maxiter);


    auto cg_it = cg_eval.begin();
    auto power_it = power_eval.begin();
    std::vector<double> x;
    double lambda;
    std::vector<double> y;
    while(cg_it != cg_eval.end() && power_it != power_eval.end()){
      auto cg_op = *cg_it;
      auto power_op = *power_it;

      bool cg_is_matvec = std::visit([&](auto&& arg) -> bool{
            using DT = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              return true;
            }
            return false;
          },cg_op);

      bool power_is_matvec = std::visit([&](auto&& arg) -> bool{
            using DT = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              return true;
            }
            return false;
          },power_op);

      /*We have four posibilities: 
       * Either both CG and power iteration have requested a
       * matrix-vector product,one of them is done,
       * or both of them are done.
       *
       * But in all other cases we just handle them with the
       * same visitor pattern as before.
       * */
      if(cg_is_matvec && power_is_matvec){
        /*In this case since both algorithms want matrix-vector product
         * with the same matrix `A` we batch them up.*/
        auto [in0,out0] = std::get<deferred_matvec<double>>(cg_op);
        auto [in1,out1] = std::get<deferred_matvec<double>>(power_op);
        A.matvec_batch2(in0,in1,out0,out1);
      }
      else{
        std::visit([&](auto&& arg){
                  using DT = std::decay_t<decltype(arg)>;
                  /*Power iteration has requested ("reverse-communicated") a sparse matrix-vector product
                   * so now we evaluate that here.*/
                  if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                    auto [in, out] = arg;
                    A.matvec(in,out);
                  }
                  /*Algorithm has produced an answer which we capture.*/
                  else{
                    auto [result] = arg;
                    auto [tmp_lambda,tmp_y] = result;
                    y = tmp_y;
                    lambda = tmp_lambda;
                  }
                }, power_op);
        std::visit([&](auto&& arg){
              using DT = std::decay_t<decltype(arg)>;
              /*CG has requested ("reverse-communicated") a sparse matrix-vector product
               * so now we evaluate that here.*/
              if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                auto [in, out] = arg;
                A.matvec(in,out);
              }
              /*Algorithm has produced an answer which we capture.*/
              else{
                auto [tmp_x] = arg;
                x = tmp_x;
              }
            }, cg_op);


      }
      cg_it++;
      power_it++;
    }
    /*Clean up any remaining steps from both algorithms.*/
    while(cg_it != cg_eval.end()){
      auto cg_op = *cg_it;
      std::visit([&](auto&& arg){
            using DT = std::decay_t<decltype(arg)>;
            /*CG has requested ("reverse-communicated") a sparse matrix-vector product
             * so now we evaluate that here.*/
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              auto [in, out] = arg;
              A.matvec(in,out);
            }
            /*Algorithm has produced an answer which we capture.*/
            else{
              auto [tmp_x] = arg;
              x = tmp_x;
            }
          }, cg_op);
      cg_it++;
    }
    while(power_it != power_eval.end()){
      auto power_op = *power_it;
      std::visit([&](auto&& arg){
                using DT = std::decay_t<decltype(arg)>;
                /*Power iteration has requested ("reverse-communicated") a sparse matrix-vector product
                 * so now we evaluate that here.*/
                if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                  auto [in, out] = arg;
                  A.matvec(in,out);
                }
                /*Algorithm has produced an answer which we capture.*/
                else{
                  auto [result] = arg;
                  auto [tmp_lambda,tmp_y] = result;
                  y = tmp_y;
                  lambda = tmp_lambda;
                }
              }, power_op);
      power_it++;
    }




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
  }

}

template<typename I,typename T>
void batched_multithreaded_revcomm(
    int64_t nrows,
    int64_t maxiter,
    const sparse_t<I,T>& A,
    const std::vector<T>& b){
  std::vector<double> r(nrows,0.0);
  {
    scope_timer_t timer("batched reverse communication");

    /**
     * The idea here is instead of simply sequentially iterating both coroutines
     * to completion we can co-iterate them and batch up any matching
     * matrix-vector products involving `A`, greatly reducing
     * the redundant memory reads while still keeping
     * the algorithms logically separate.
     */

    auto cg_eval = cg_revcomm(nrows, std::span<const double>(b),maxiter);
    auto power_eval = power_revcomm(nrows, std::span<const double>(b),maxiter);


    auto cg_it = cg_eval.begin();
    auto power_it = power_eval.begin();
    std::vector<double> x;
    double lambda;
    std::vector<double> y;
    while(cg_it != cg_eval.end() && power_it != power_eval.end()){
      auto cg_op = *cg_it;
      auto power_op = *power_it;

      bool cg_is_matvec = std::visit([&](auto&& arg) -> bool{
            using DT = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              return true;
            }
            return false;
          },cg_op);

      bool power_is_matvec = std::visit([&](auto&& arg) -> bool{
            using DT = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              return true;
            }
            return false;
          },power_op);

      /*We have four posibilities: 
       * Either both CG and power iteration have requested a
       * matrix-vector product,one of them is done,
       * or both of them are done.
       *
       * But in all other cases we just handle them with the
       * same visitor pattern as before.
       * */
      if(cg_is_matvec && power_is_matvec){
        /*In this case since both algorithms want matrix-vector product
         * with the same matrix `A` we batch them up.*/
        auto [in0,out0] = std::get<deferred_matvec<double>>(cg_op);
        auto [in1,out1] = std::get<deferred_matvec<double>>(power_op);
        A.matvec_batch2_multithreaded(in0,in1,out0,out1);
      }
      else{
        std::visit([&](auto&& arg){
                  using DT = std::decay_t<decltype(arg)>;
                  /*Power iteration has requested ("reverse-communicated") a sparse matrix-vector product
                   * so now we evaluate that here.*/
                  if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                    auto [in, out] = arg;
                    A.matvec(in,out);
                  }
                  /*Algorithm has produced an answer which we capture.*/
                  else{
                    auto [result] = arg;
                    auto [tmp_lambda,tmp_y] = result;
                    y = tmp_y;
                    lambda = tmp_lambda;
                  }
                }, power_op);
        std::visit([&](auto&& arg){
              using DT = std::decay_t<decltype(arg)>;
              /*CG has requested ("reverse-communicated") a sparse matrix-vector product
               * so now we evaluate that here.*/
              if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                auto [in, out] = arg;
                A.matvec(in,out);
              }
              /*Algorithm has produced an answer which we capture.*/
              else{
                auto [tmp_x] = arg;
                x = tmp_x;
              }
            }, cg_op);


      }
      std::thread cg_advance([&](){cg_it++;});
      std::thread power_advance([&](){power_it++;});
      cg_advance.join();
      power_advance.join();
    }
    /*Clean up any remaining steps from both algorithms.*/
    while(cg_it != cg_eval.end()){
      auto cg_op = *cg_it;
      std::visit([&](auto&& arg){
            using DT = std::decay_t<decltype(arg)>;
            /*CG has requested ("reverse-communicated") a sparse matrix-vector product
             * so now we evaluate that here.*/
            if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
              auto [in, out] = arg;
              A.matvec(in,out);
            }
            /*Algorithm has produced an answer which we capture.*/
            else{
              auto [tmp_x] = arg;
              x = tmp_x;
            }
          }, cg_op);
      cg_it++;
    }
    while(power_it != power_eval.end()){
      auto power_op = *power_it;
      std::visit([&](auto&& arg){
                using DT = std::decay_t<decltype(arg)>;
                /*Power iteration has requested ("reverse-communicated") a sparse matrix-vector product
                 * so now we evaluate that here.*/
                if constexpr (std::is_same_v<DT, deferred_matvec<double>>){
                  auto [in, out] = arg;
                  A.matvec(in,out);
                }
                /*Algorithm has produced an answer which we capture.*/
                else{
                  auto [result] = arg;
                  auto [tmp_lambda,tmp_y] = result;
                  y = tmp_y;
                  lambda = tmp_lambda;
                }
              }, power_op);
      power_it++;
    }




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
  }

}






int main(int argc,char** argv){
  std::mt19937 rng(23947);
  int64_t nrows=200000;
  int64_t spread=32;
  int64_t nnz_per_row=30;
  int64_t maxiter = 50;
  double eps = 1.0;

  sparse_t<int64_t,double> A(nrows,spread,nnz_per_row,eps,rng);
  std::vector<double> b(nrows,1.0);
  plain_sequential(nrows,maxiter,A,b);
  plain_multithreaded(nrows,maxiter,A,b);
  plain_revcomm(nrows,maxiter,A,b);
  batched_revcomm(nrows,maxiter,A,b);
  batched_multithreaded_revcomm(nrows,maxiter,A,b);




  return 0;
}
