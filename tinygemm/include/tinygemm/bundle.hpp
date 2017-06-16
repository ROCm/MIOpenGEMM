#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <string>

#include <tinygemm/hyperparams.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/geometry.hpp>
#include <tinygemm/kernelstring.hpp>

namespace tinygemm{

  
  

namespace kerngen{



class Bundle{
  public:

    const std::vector<KernelString> v_tgks;
    const std::vector<std::vector<unsigned> > v_wait_indices;
    derivedparams::DerivedParams dp;

    Bundle(std::vector<KernelString> && v_tgks_, std::vector<std::vector<unsigned> > && v_wait_indices_, const derivedparams::DerivedParams & dp_): v_tgks(v_tgks_), v_wait_indices(v_wait_indices_), dp(dp_) {}
};

Bundle get_bundle(const hyperparams::HyperParams & hp, const TinyGemmGeometry & gg, outputwriting::OutputWriter & mowri, bool bundle_verbose);
  

}
}

#endif
