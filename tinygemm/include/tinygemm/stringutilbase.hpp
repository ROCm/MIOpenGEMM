#ifndef STRINGUTILBASE_HPP
#define STRINGUTILBASE_HPP


#include <string>
#include <vector>
#include <iso646.h>

namespace tinygemm{
namespace stringutil{
//split the string tosplit by delim. With x appearances of delim in tosplit, the returned vector will have length x + 1 (even if appearances at the start, end, contiguous.
std::vector<std::string> split(const std::string & tosplit, const std::string & delim);

//split on whitespaces
std::vector<std::string> split(const std::string & tosplit);

std::string getdirfromfn(const std::string & fn);

//split something like QWE111 into QWE and 111.
std::tuple<std::string, unsigned> splitnumeric(std::string alphanum);

}
}


#endif
