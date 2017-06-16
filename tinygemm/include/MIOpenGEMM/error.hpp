#ifndef ERROR_HPP
#define ERROR_HPP

#include <stdexcept>

namespace MIOpenGEMM{

class miog_error : public std::runtime_error{
  public:
    miog_error(const std::string& what_arg);
};

void miog_warning(const std::string & warning);

}


#endif
