#ifndef _KERNELCACHE_HPP
#define _KERNELCACHE_HPP


#include <miopengemm/solution.hpp>
namespace MIOpenGEMM{
 
std::string get_cache_keys_string(std::string k_dev, std::string k_con, std::string k_geo, std::string k_comment);


class CachedSolution {
  public:
    std::string hyperstring;
    SolutionStatistics stats;    
    CachedSolution(std::string hyperstring_, SolutionStatistics stats_ ):hyperstring(hyperstring_), stats(stats_) {}//, stats_string(stats_string_) {}
    CachedSolution() = default;    
    
    std::string get_string();
    
};


/* TODO : unordered maps are faster */
using KernelCache = std::map< std::string, std::map< std::string, std::map<std::string, std::map<std::string, CachedSolution> > > >;

KernelCache get_kernel_cache();


CachedSolution get_generic_cached_solution(const std::string & constraints_string, const Geometry & gg);


/* [device][constraint][geometry][further_comment] -> cached solution */
extern const KernelCache kernel_cache;

void add_entry(KernelCache & kc, const std::string & k_dev, const std::string & k_con,  const std::string k_geo, const std::string k_comment, CachedSolution tgcs);


}


#endif
