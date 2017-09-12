/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_HINT_H
#define GUARD_MIOPENGEMM_HINT_H

#ifdef __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#include <limits>
#include <tuple>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

class BasicHint
{
  private:
  size_t                   id;
  std::vector<std::string> matches;

  public:
  bool   has_id() const;
  size_t get_id() const;
  BasicHint();
  BasicHint(size_t id_, const std::vector<std::string>& matches_);
  bool is_match_hit(const std::string& s2) const;
  void set(size_t& x, const std::vector<std::string> ts) const;
  virtual std::string get_description() const = 0;
};

class PlatformHint : public BasicHint
{
  public:
  virtual std::string get_description() const override final { return "platforms"; }
  PlatformHint() = default;
  PlatformHint(size_t id_, const std::vector<std::string>& matches_) : BasicHint(id_, matches_) {}
};

class DeviceHint : public BasicHint
{
  public:
  virtual std::string get_description() const override final { return "devices"; }
  DeviceHint() = default;
  DeviceHint(size_t id_, const std::vector<std::string>& matches_) : BasicHint(id_, matches_) {}
};

class CLHint
{
  public:
  // Platform hint
  PlatformHint pla;
  // Device hint
  DeviceHint dev;
  // Construct with a list of strings, to be matches to platform and device informations.
  // for example {"AMD", "Advanced Micro Devices", "gfx803", "Fiji"} should match
  // an AMD platform with a Fiji GPU.
  CLHint(const std::vector<std::string>& matches);
  // Construct from Platform ID and Device ID.
  CLHint(size_t pla_id, size_t dev_id);
  // No hint : hope that there's only one device and one platform.
  CLHint() = default;
};
}

#endif
