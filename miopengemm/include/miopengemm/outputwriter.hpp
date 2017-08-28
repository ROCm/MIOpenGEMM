/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_OUTPUTWRITER_H
#define GUARD_MIOPENGEMM_OUTPUTWRITER_H

#include <fstream>
#include <iostream>
#include <string>
#include <miopengemm/enums.hpp>

namespace MIOpenGEMM
{
namespace owrite
{

class Flusher
{
};

class Endline
{
};

class BasicWriter
{

  private:
  bool           to_terminal;
  std::ofstream* ptr_file;

  public:
  BasicWriter(bool to_terminal_, std::ofstream* ptr_file_)
    : to_terminal(to_terminal_), ptr_file(ptr_file_)
  {
  }
  BasicWriter() : BasicWriter(false, nullptr) {}

  template <typename T>
  BasicWriter& operator<<(T t)
  {
    if (to_terminal)
    {
      std::cout << t;
    }

    if (ptr_file != nullptr)
    {
      (*ptr_file) << t;
    }
    return *this;
  }
};

template <>
BasicWriter& BasicWriter::operator<<(Flusher f);

template <>
BasicWriter& BasicWriter::operator<<(Endline e);

class Writer
{
  private:
  Ver::E      v;
  std::string filename;
  void        initialise_file();

  public:
  std::ofstream file;
  std::array<BasicWriter, OutPart::N> bw;

  public:
  Writer();
  ~Writer();
  Writer(Ver::E v, std::string filename);

  // a short-cut function
  template <typename T>
  Writer& operator<<(T t)
  {
    bw[OutPart::E::MAI] << t;
    return *this;
  }
};
}

extern owrite::Flusher Flush;
extern owrite::Endline Endl;
}

#endif
