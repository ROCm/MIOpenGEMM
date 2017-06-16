#ifndef TINYGEMMERROR_HPP
#define TINYGEMMERROR_HPP

#include <stdexcept>

namespace MIOpenGEMM{

class miog_error : public std::runtime_error{
  public:
    miog_error(const std::string& what_arg);
};

void tinygemm_warning(const std::string & warning);

}


#endif
