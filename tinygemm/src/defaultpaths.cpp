#include <tinygemm/defaultoutpath.hpp>
#include <array>
#include <stdlib.h>

namespace tinygemm{
namespace defpaths{

template<std::size_t N>
std::array<char, N> make_str_array(const char(&a)[N])
{
    std::array<char, N> result;
    std::copy(a, a+N, result.begin());
    return result;
}


tmp_dir::tmp_dir()
{
    name = boost::filesystem::temp_directory_path()/"tinygemm"/boost::filesystem::unique_path();
    boost::filesystem::create_directories(name);
}

tmp_dir::~tmp_dir()
{
    // TODO: Delete directory and all files in it
    // std::remove(name.c_str());
}

}}