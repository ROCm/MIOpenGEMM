#ifndef TINYGEMM_MAPKEYCHECK_HPP
#define TINYGEMM_MAPKEYCHECK_HPP


#include <tinygemm/tinygemmerror.hpp>
#include <sstream>
#include <algorithm>


#include <iostream>
namespace tinygemm{
namespace mapkeycheck{


template <typename ValType>
void check_map_keys(const std::map<std::string, ValType> & params, const std::vector<std::string>  & names, const std::string & hash){
  std::stringstream ss;
  ss << "while in check_map_keys (" << hash << "). ";
  for (auto & x : names){
    if (params.count(x) == 0){
      ss << "The parameter `" << x << "', should appear as a key but appears not to. \n";
      throw tinygemm_error(ss.str());
    }
  }
  
  for (auto & x : params){
    //std::cout << x.first << std::endl;
    auto blip = std::find(names.cbegin(), names.cend(), x.first);
    if (blip == names.cend()) {
      ss << "The parameter `" << x.first << "', which appears in the map, is not recognised in the list of prescribed names.\n";
      throw tinygemm_error(ss.str());
    }
  }
}

}
}

#endif
