#include "tinygemmerror.hpp"

namespace tinygemm{
tinygemm_error::tinygemm_error(const std::string& what_arg):std::runtime_error(what_arg){};
tinygemm_error::tinygemm_error(const char* what_arg):std::runtime_error(what_arg){};
}
