/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/

#include <array>
#include <miopengemm/enums.hpp>

namespace MIOpenGEMM {

template <typename T>
T unfilled();

template <>
char unfilled(){
  return '?';
}

template <>
std::string unfilled(){
  return "?";
}

template <typename T>
void confirm(const std::vector<T> & X, std::string enum_name){
  for (auto & x : X){
    if (x == unfilled<T>()){
      throw miog_error("unpopulated element of vector for " + enum_name + ".");
    }
  }
}

template <typename T>
std::unordered_map<T, size_t> get_val(const std::vector<T> & a){
  std::unordered_map<T, size_t> X;
  for (size_t i = 0; i < a.size(); ++i){
    X[a[i]] = i;
  }
  return X;
}


template <typename T>
EnumMapper<T>::EnumMapper(const std::vector<T> & name_) :n(name_.size()), name(name_), val(get_val<T>(name)) {}


template <typename T>
EnumMapper<T> get_enum_mapper(const std::vector<T> & name_, std::string enum_name){
  confirm<T>(name_, enum_name);
  return EnumMapper<T>(name_);
}

namespace BasicKernelType
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::WSA] = "wsa";
    X[E::WSB] = "wsb";
    X[E::BETAC] = "betac";
    X[E::MAIN] = "main";
    return X;
  }
  const EnumMapper<std::string> M = get_enum_mapper<std::string>(get_name(), "BasicKernelType");
  
}

namespace SummStat
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::MEAN] = "mean";
    X[E::MEDIAN] = "median";
    X[E::MAX] = "max";
    return X;
  }
  const EnumMapper<std::string> M = get_enum_mapper<std::string>(get_name(), "SummStat");
}
  
namespace Chi
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::MIC] = "MIC";
    X[E::PAD] = "PAD";
    X[E::PLU] = "PLU";
    X[E::LIW] = "LIW";
    X[E::MIW] = "MIW";
    X[E::WOS] = "WOS";
    return X;
  }
  const EnumMapper<std::string> M = get_enum_mapper<std::string>(get_name(), "Chi");
}

namespace NonChi
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::UNR] = "UNR";
    X[E::GAL] = "GAL";
    X[E::PUN] = "PUN";
    X[E::ICE] = "ICE";
    X[E::NAW] = "NAW";
    X[E::UFO] = "UFO";
    X[E::MAC] = "MAC";
    X[E::SKW] = "SKW";
    return X;
  }
  const EnumMapper<std::string> M = get_enum_mapper<std::string>(get_name(), "NonChi");
}
  
namespace Mat
{
  std::vector<char> get_name() {
    std::vector<char> X(E::N, unfilled<char>());
    X[E::A] = 'A';
    X[E::B] = 'B';
    X[E::C] = 'C';
    return X;
  }  
  const EnumMapper<char> M = get_enum_mapper<char>(get_name(), "Mat");
}


// TODO : make use of this new enum more widely. 
namespace Mem
{
  std::vector<char> get_name() {
    std::vector<char> X(E::N, unfilled<char>());
    X[E::A] = 'A';
    X[E::B] = 'B';
    X[E::C] = 'C';
    X[E::W] = 'W';
    return X;
  }  
  const EnumMapper<char> M = get_enum_mapper<char>(get_name(), "Mem");
}


 
}
