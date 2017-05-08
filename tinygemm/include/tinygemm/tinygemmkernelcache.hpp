#ifndef TINYGEMM_TINYGEMMKERNELCACHE_HPP
#define TINYGEMM_TINYGEMMKERNELCACHE_HPP

namespace tinygemm{
  

class TinygemmCachedSolution {
  public:
    std::string hyperstring;
    std::string stats_string;
    TinygemmCachedSolution(std::string hyperstring_, std::string stats_string_):hyperstring(hyperstring_), stats_string(stats_string_) {}
    TinygemmCachedSolution() = default;
    
};

using KernelCache = std::map< std::string, std::map< std::string, std::map<std::string, tinygemm::TinygemmCachedSolution> > >;

KernelCache get_kernel_cache();

/* [device][constraint][geometry] -> cached solution */
extern const KernelCache kernel_cache;


}


#endif
