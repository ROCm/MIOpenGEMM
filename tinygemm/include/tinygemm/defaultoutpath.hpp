#ifndef DEFAULTOUTPATH_HPP
#define DEFAULTOUTPATH_HPP

#include <string>
#include <boost/filesystem.hpp>

namespace tinygemm{
namespace defpaths{

struct tmp_dir {
    boost::filesystem::path name;
    tmp_dir();
    tmp_dir(const tmp_dir&) = delete;
    ~tmp_dir();
};
}
}


#endif 
