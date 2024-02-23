#ifndef __TIMER_H_
#define __TIMER_H_

#include <iostream>
#include <chrono>
#include <string>
#include <cstring>

class scope_timer_t {
private:
    std::string label;
    std::chrono::high_resolution_clock::time_point start;
public:
    scope_timer_t(const std::string& label) : label(label), start(std::chrono::high_resolution_clock::now()) {}

    ~scope_timer_t() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << label << ": " << duration << " milliseconds" << std::endl;
    }
};



#endif
