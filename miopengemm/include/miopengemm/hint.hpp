/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_HINT_H
#define GUARD_MIOPENGEMM_HINT_H

#include <CL/cl.h>
#include <tuple>
#include <miopengemm/outputwriter.hpp>
#include <limits>

namespace MIOpenGEMM
{

class BasicHint{
  private:
  size_t id;
  std::vector<std::string> matches;
  std::string description;
  
  public:  
  bool has_id() const;
  size_t get_id() const;
  BasicHint(std::string); 
  BasicHint(size_t id_, const std::vector<std::string> & matches_, std::string desc_); 
  bool is_match_hit(const std::string & s2) const;  
  void set(size_t & x, const std::vector<std::string> ts) const;
};

class CLHint{
  public:
  // Platform hint
  BasicHint pla;
  // Device hint
  BasicHint dev;
  // Construct with a list of strings, to be matches to platform and device informations.
  // for example {"AMD", "Advanced Micro Devices", "gfx803", "Fiji"} should match
  // an AMD platform with a Fiji GPU.    
  CLHint(const std::vector<std::string> & matches);
  // Construct from Platform ID and Device ID.
  CLHint(size_t pla_id, size_t dev_id);
  // No hint : hope that there's only one device and one platform. 
  CLHint();
  // TODO : A hint which finds the GPU with the most compute units. 
};

}


#endif
