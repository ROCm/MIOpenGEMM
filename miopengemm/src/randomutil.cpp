#include <miopengemm/randomutil.hpp>


namespace MOOMOOMOOGEMM{

RandomUtil::RandomUtil():rd(), gen(rd()) {} 

unsigned RandomUtil::get_from_range(unsigned upper){
  return unidis(gen) % upper;
}
  


}
