#ifndef _RANDOMUTIL_HPP
#define _RANDOMUTIL_HPP

#include <random>
#include <algorithm>
#include <MIOpenGEMM/error.hpp>

namespace MIOpenGEMM{

class RandomUtil {

private:
  std::random_device rd;
  std::default_random_engine gen;
  std::uniform_int_distribution<unsigned> unidis;

public:
  RandomUtil();
  unsigned get_from_range(unsigned upper);  
  template <typename T>
  void shuffle(unsigned start_index, unsigned end_index, T & t){
    if (end_index > t.size() || start_index > end_index){
      throw miog_error("problem in template function RandomUtil::shuffle");
    }
    std::shuffle(t.begin() + start_index, t.begin() + end_index, gen);
  }
};
  
}


#endif
