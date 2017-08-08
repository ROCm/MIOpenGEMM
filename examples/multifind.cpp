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
  
  std::vector<Constraints> constraints_s{{""}};//A_WOS1__B_WOS1"}};
 
 
  std::vector<Geometry> geometries = get_deepbench(0);
  for (size_t i = 0; i < geometries.size(); ++i)
  {
    std::swap(geometries[i], geometries[i + rand() % (geometries.size() - i)]);
    auto m = geometries[i].m;
    auto n = geometries[i].n;
    auto k = geometries[i].k;
    geometries[i].wSpaceSize = (m + 16)*(k + 16) + (n + 16)*(k + 16);
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
      std::string basedir = "/home/james/test45/";
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
