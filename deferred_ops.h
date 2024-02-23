#ifndef __DEFERRED_OPS_H_
#define __DEFERRED_OPS_H_


template<typename T>
struct deferred_matvec{
  std::span<const T> in;
  std::span<T> out;
};

template<typename T>
struct done{
  T result;
};

#endif
