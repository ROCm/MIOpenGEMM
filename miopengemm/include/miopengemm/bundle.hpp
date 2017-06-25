
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef KERNELSTRINGGENERATOR_HPP
#define KERNELSTRINGGENERATOR_HPP

#include <string>

#include <miopengemm/hyperparams.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/kernelstring.hpp>

namespace MIOpenGEMM{

  
  

namespace kerngen{



class Bundle{
  public:

    const std::vector<KernelString> v_tgks;
    const std::vector<std::vector<unsigned> > v_wait_indices;
    
    derivedparams::DerivedParams dp;

    //TODO : when is std::move needed. I know, I really should know...
    Bundle(std::vector<KernelString> && v_tgks_, std::vector<std::vector<unsigned> > && v_wait_indices_, derivedparams::DerivedParams && dp_): v_tgks(std::move(v_tgks_)), v_wait_indices(std::move(v_wait_indices_)), dp(std::move(dp_))  {
      
    }
};

Bundle get_bundle(const hyperparams::HyperParams & hp, const Geometry & gg, outputwriting::OutputWriter & mowri, bool bundle_verbose);
  

}
}

#endif
