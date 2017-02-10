#ifndef GENERATORUTIL_HPP
#define GENERATORUTIL_HPP

#include <string>

namespace tinygemm{
namespace genutil{

std::string get_time_string(const std::string & type); 

std::string get_what_string();

std::string get_how_string();

std::string get_derived_string();

  
}
}

#endif
