# cg_revcom_cpp (PROOF OF CONCEPT CODE!)
Looking at reverse communication using std::generator in c++.

This is a proof-of-concept. I cut a lot of corners in the name of getting some interesting results.

# Implementation of std::generator

As of this writing: the inclusion of `std::generator` in standard C++ is very new so I used an
available [reference implementation](https://github.com/lewissbaker/generator) which is *not* 
my own. As far as I can tell only libstdc++ contains an implementation of `std::generator`
but libc++ is easier to download because they give binaries for common platforms. It is
in the [C++23 standard](https://en.cppreference.com/w/cpp/coroutine/generator) however 
so I expect this code should eventually work as the libraries catch up to this.

# References

 * https://www.netlib.org/lapack/lawnspdf/lawn99.pdf
