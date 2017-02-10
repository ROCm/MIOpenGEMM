#include <tinygemm/generatorutil.hpp>

#include <chrono>
#include <ctime>
#include <sstream>

namespace tinygemm{
namespace genutil{

std::string get_generic_kernelname(const std::string & type){
  return "tg_" + type;
}


std::string get_time_string(const std::string & type){  

  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t generation_time = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss <<  
R"(/* ****************************************************************************
* This )" << type << " kernel string was generated on " << std::ctime(&generation_time) << 
R"(**************************************************************************** */)";
  return ss.str();
}

std::string get_what_string(){
  return R"(/* ***********************************************
* These parameters define WHAT this kernel does *
* *********************************************** */)";
}

std::string get_how_string(){
  return R"(/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/)";
}

std::string get_derived_string(){
  return R"(/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */)";
}
  
}
}
