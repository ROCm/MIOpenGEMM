#ifndef TINYGEMMERROR_HPP
#define TINYGEMMERROR_HPP

#include <stdexcept>

namespace tinygemm{

class tinygemm_error : public std::runtime_error{
  public:
    tinygemm_error(const std::string& what_arg);
};

void tinygemm_warning(const std::string & warning);

}


#endif
