/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <string>
#include <miopengemm/error.hpp>
#include <miopengemm/prepgenerator.hpp>
#include <miopengemm/strascopgenerator.hpp>


namespace MIOpenGEMM
{
namespace strasgen
{


                   
//StrasCopGenerator::StrasCopGenerator(Mat::E, const HyPas&, const Geometry&, const DerivedParams&, KType::E){
  //throw miog_error("failed, must implement in strascop");  
//}


KernBlob StrasCopGenerator::get_kernelstring(){
  
  std::cout << "\nin StrasCopGenerator::get_kernelstring, definitely something to do ...\n" << std::endl;
  return {};
}

void StrasCopGenerator::setup_final(){
  std::cout << "\nin StrasCopGenerator::setup_final, probably something to do ...\n" << std::endl;
}

void StrasCopGenerator::set_type(){
  type = "is this necessary ??";
}


KType::E StrasCopGenerator::get_ktype()
{
  throw miog_error("failed in get_ktype of StrasCopGenerator: implement get_ktype");
}


size_t StrasCopGenerator::get_local_work_size(){
  throw miog_error("failed, must implement in strascop: implement get_local_work_size ");
  return 0;
}

size_t StrasCopGenerator::get_n_work_groups(){
  throw miog_error("failed, must implement in strascop: implement get_n_work_groups");
  return 0;
}


StrasCopGenerator::StrasCopGenerator(Mat::E em, const HyPas& hp_, const Geometry& gg_, const DerivedParams& dp_, KType::E kt):prepgen::PrepGenerator(em, hp_, gg_, dp_)
{
  (void)kt;
}

KernBlob get_stras_kernelstring(const HyPas& hp, const Geometry& gg, const DerivedParams& dp, KType::E ktype){

  Mat::E emat;
  switch(ktype){
    case (KType::E::STRAS11A) : emat = Mat::E::A; break;
    case (KType::E::STRAS15A) : emat = Mat::E::A; break;
    case (KType::E::STRAS13B) : emat = Mat::E::B; break;
    case (KType::E::STRAS16B) : emat = Mat::E::B; break;
    case (KType::E::STRAS18C) : emat = Mat::E::C; break;
    case (KType::E::STRAS19C) : emat = Mat::E::C; break;
    default : throw miog_error("invalid KType in get_stras_kernelstring of strascopgenerator");
  }


  StrasCopGenerator strascop(emat, hp, gg, dp, ktype);
  strascop.setup();
  return strascop.get_kernelstring();
  
}


  
}

  

}
