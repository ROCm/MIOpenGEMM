/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/

#include <iostream>
#include <array>
#include <algorithm>
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
T aslower(T X){
  
}

template <>
char aslower(char x){
  char y = std::tolower(static_cast<unsigned char>(x));
  std::cout << "---------- " << x << " : " << y << " ----------------\n";
  

  return y;
}

template <>
std::string aslower(std::string X){
  std::string Y = X;
  std::transform(X.begin(), X.end(), Y.begin(), ::tolower);

  std::cout << "---------- " << X << " : " << Y << " ----------------\n";

  return Y;
}

template <typename T>
EnumMapper<T>::EnumMapper(const std::vector<T> & name_) :N(name_.size()), name(name_), val(get_val<T>(name)) {

  lcase_name.resize(name.size());
  for (size_t i = 0; i < name.size(); ++i){
    lcase_name[i] = aslower<T>(name[i]);
  }
}


template <typename T>
EnumMapper<T> get_enum_mapper(const std::vector<T> & name_, std::string enum_name){
  confirm<T>(name_, enum_name);
  return EnumMapper<T>(name_);
}

namespace BasicKernelType
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::WSA] = "WSA";
    X[E::WSB] = "WSB";
    X[E::BETAC] = "BETAC";
    X[E::MAIN] = "MAIN";
    return X;
  }
  const EnumMapper<std::string> M = get_enum_mapper<std::string>(get_name(), "BasicKernelType");
  
}

namespace SummStat
{
  std::vector<std::string> get_name() {
    std::vector<std::string> X(E::N, unfilled<std::string>());
    X[E::MEAN] = "MEAN";
    X[E::MEDIAN] = "MEDIAN";
    X[E::MAX] = "MAX";
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


  const EnumMapper<std::string> * mat_to_xchi(Mat::E emat){
    switch (emat){
      case Mat::E::A : return &Chi::M;
      case Mat::E::B : return &Chi::M;
      case Mat::E::C : return &NonChi::M;
      default : throw miog_error("unrecognised Mat::E in mat_to_xchi");
    }
  }
  
  // TODO : rather make this an array for lookup
  Mat::E mem_to_mat(Mem::E emat){
    if (emat == Mem::E::A){
      return Mat::E::A;
    }
    
    else if (emat == Mem::E::B){
      return Mat::E::B;
    }
    
    else if (emat == Mem::E::C){
      return Mat::E::C;
    }
    
    else{
      throw miog_error("no mat enum for supposed mem enum provided");
    }
  }
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

  Mem::E mat_to_mem(Mat::E emat){
    if (emat == Mat::E::A){
      return Mem::E::A;
    }
    
    else if (emat == Mat::E::B){
      return Mem::E::B;
    }
    
    else if (emat == Mat::E::C){
      return Mem::E::C;
    }
    
    else{
      throw miog_error("no mem enum for supposed mat enum provided");
    }
  }
}




 
}
