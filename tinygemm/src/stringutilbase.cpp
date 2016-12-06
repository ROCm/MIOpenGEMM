
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <iostream>
#include <tuple>
#include <algorithm>

namespace tinygemm{
namespace stringutil{
//split the string tosplit by delim. With x appearances of delim in tosplit, the returned vector will have length x + 1 (even if appearances at the start, end, contiguous.
std::vector<std::string> split(const std::string & tosplit, const std::string & delim){
	
	std::vector<std::string> spv; //vector to return
	if (delim.length() > tosplit.length()){
		return spv;
	}


	std::vector<size_t> splitposstarts {0};		
	std::vector<size_t> splitposends;
	
	for (size_t x = 0; x <  tosplit.length() - delim.length() + 1; ++x){		
		auto res = std::mismatch(delim.begin(), delim.end(), tosplit.begin() + x);
		if (res.first == delim.end()){
			splitposends.push_back(x);
			splitposstarts.push_back(x + delim.length());

		}
	}
	
	splitposends.push_back(tosplit.length());

	for (unsigned i = 0; i < splitposends.size(); ++i){
		spv.push_back(tosplit.substr(splitposstarts[i], splitposends[i] - splitposstarts[i] ));
	}
	
	return spv;
}


std::tuple<std::string, unsigned> splitnumeric(std::string alphanum){
	size_t split_point = alphanum.find_first_of("0123456789");
	return  std::make_tuple<std::string, unsigned>(alphanum.substr(0, split_point), std::stoi(alphanum.substr(split_point, alphanum.size() )));
}

bool isws(const char & c){
	return (c == ' ' ||  c == '\t' || c == '\n');
}

std::vector<std::string> split(const std::string & tosplit){
	
	std::vector<std::string> spv2;
	
	unsigned it = 0;	
	
	while (it != tosplit.size()){
		while (isws(tosplit[it]) and it != tosplit.size()){
			++it;
		}
		unsigned start = it;
		
		while (!isws(tosplit[it]) and it != tosplit.size()){
			++it;
		}
		unsigned end = it;
		
		if (!isws(tosplit[end -1])){
			spv2.push_back(tosplit.substr(start, end - start));
		}
	}
		
		
		
	
	return spv2;
} 


std::string getdirfromfn(const std::string & fn){
	auto morcels = split(fn, "/");

	if (morcels[0].compare("") != 0){
		throw tinygemm_error("The string passed to getdirfromfn is not a valid path as there is no leading / .");
	}

	std::string dir = "/";
		
	for (unsigned i = 1; i < morcels.size() - 1; ++i){
		dir = dir + morcels[i] + "/";
	}
	return dir;
}


}
}
