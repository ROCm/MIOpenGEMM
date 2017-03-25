#include <tinygemm/randomutil.hpp>


namespace tinygemm{

RandomUtil::RandomUtil():rd(), gen(rd()) {} 

unsigned RandomUtil::get_from_range(unsigned upper){
  return unidis(gen) % upper;
}
  


}
