#ifndef TINYGEMM_RANDOMUTIL_HPP
#define TINYGEMM_RANDOMUTIL_HPP

#include <random>
#include <algorithm>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{

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
        throw tinygemm_error("problem in shuffle");
      }
      std::shuffle(t.begin() + start_index, t.begin() + end_index, gen);
    }
  };
  
}


#endif
