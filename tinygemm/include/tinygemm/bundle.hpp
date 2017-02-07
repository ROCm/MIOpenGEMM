#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <string>

#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/kernelstring.hpp>

namespace tinygemm{
namespace kerngen{



class Bundle{
  public:

    const std::vector<KernelString> v_tgks;
    derivedparams::DerivedParams dp;

    Bundle(std::vector<KernelString> && v_tgks_, const derivedparams::DerivedParams & dp_): v_tgks(v_tgks_), dp(dp_) {}    
};

Bundle get_bundle(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg);
  

}
}

#endif
