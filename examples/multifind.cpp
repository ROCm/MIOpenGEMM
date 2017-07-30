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
  std::vector<Geometry> geometries; //get_deepbench(0);  // zero workspace deepbench geometries
  
  
  std::vector<size_t> dimensions;
  for (size_t k = 5; k < 14; ++k){
    dimensions.push_back(std::pow(2, k));
    for (size_t off = 1; off < 5; ++off){
      dimensions.push_back(std::pow(2, k) + off);
      dimensions.push_back(std::pow(2, k) - off);
    }
  }
  
  for (size_t k = 300; k < 7000; k+= 300){
    dimensions.push_back(k);
  }
  
  for (auto d : dimensions){
    geometries.push_back({1, 0, 0, 0, d, d, d, d, d, d, 0, 'f'});
    geometries.push_back({1, 0, 1, 0, d, d, d, d, d, d, 0, 'f'});
    geometries.push_back({1, 1, 0, 0, d, d, d, d, d, d, 0, 'f'});
  }
  
  std::cout << geometries.size();
  
  
  std::vector<Constraints> constraints_s{{""}};
  for (size_t i = 0; i < geometries.size(); ++i)
  {
    std::swap(geometries[i], geometries[i + rand() % (geometries.size() - i)]);
  }

  size_t                counter = 0;
  std::vector<Solution> solutions;
  for (auto& gg : geometries)
  {
    for (auto& cons : constraints_s)
    {
      ++counter;
      std::cout << '(' << counter << ')' << std::endl;
      std::string basedir = "/home/james/test27/";
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
