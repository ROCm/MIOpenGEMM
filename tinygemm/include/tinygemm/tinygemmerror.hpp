#ifndef TINYGEMMERROR_HPP
#define TINYGEMMERROR_HPP

#include <stdexcept>

namespace tinygemm{

class tinygemm_error : public std::runtime_error{
  public:
    tinygemm_error(const std::string& what_arg);
    tinygemm_error(const char* what_arg );  
};

}


#endif
