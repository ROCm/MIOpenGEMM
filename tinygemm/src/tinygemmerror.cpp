#include <tinygemm/tinygemmerror.hpp>

#include <iostream>

namespace tinygemm{
  tinygemm_error::tinygemm_error(const std::string& what_arg):std::runtime_error("from tinygemm. " + what_arg){}

  void tinygemm_warning(const std::string & warning){
    std::cerr << "TINYGEMM WARNING:\n" << warning << std::flush;
  }

}



