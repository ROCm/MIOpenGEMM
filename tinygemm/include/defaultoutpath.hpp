#ifndef DEFAULTOUTPATH_HPP
#define DEFAULTOUTPATH_HPP

#include <string>

namespace tinygemm{
namespace defpaths{

struct tmp_dir {
    std::string name;
    tmp_dir();
    tmp_dir(const tmp_dir&) = delete;
    ~tmp_dir();
};
}
}


#endif 
