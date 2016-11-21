#ifndef MAKEKERNELSOURCE_HPP
#define MAKEKERNELSOURCE_HPP

#include <map>
#include <vector>
#include <string>

namespace mkkern{
//if filename is "", default name is used. 
int make_kernel_via_python(std::string dir_name, std::string t_float, std::map<std::string, unsigned> all_int_parms, std::string kernelname);

}

#endif
