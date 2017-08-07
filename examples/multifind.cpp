/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/geometries.hpp>

int main()
{

  srand(time(NULL));

  using namespace MIOpenGEMM;
  
  std::vector<Constraints> constraints_s{{""}};
 
 
  //std::vector<Geometry> geometries = get_squares(0);
  //for (size_t i = 0; i < geometries.size(); ++i)
  //{
    //std::swap(geometries[i], geometries[i + rand() % (geometries.size() - i)]);
  //}


  std::vector<Geometry> geometries;
  for (auto & key : kernel_cache.get_keys()){
    if (key.gg.m == key.gg.n && key.gg.m == key.gg.k && key.gg.k > 500){
      
      if ((key.gg.tX[Mat::E::A] || key.gg.tX[Mat::E::C]) &&  (key.gg.m == 5717 || key.gg.m == 6300 || key.gg.m == 5100 || key.gg.m == 4096 || key.gg.m == 4097 || key.gg.m == 3300 || key.gg.m == 4817)){
        geometries.push_back(key.gg);
      }
    }
  }


  std::cout << "n geometries : " << geometries.size() << std::endl;
  
  size_t                counter = 0;
  std::vector<Solution> solutions;
  for (auto& gg : geometries)
  {
    for (auto& cons : constraints_s)
    {
      ++counter;
      std::cout << '(' << counter << ')' << std::endl;
      std::string basedir = "/home/james/test44/";
      // WARNING : this call might only work on linux/mac
      std::string syscall =
        "./examples/multifindbase " + basedir + " " + gg.get_string() + " " + cons.get_string();
      // The reason we use a system call here is that if all done directly in loop,
      // compilation gets much slower. No idea why this is the case.
      std::system(syscall.c_str());
    }
  }

  return 0;
}
