#ifndef OUTPUTWRITER_H
#define OUTPUTWRITER_H

#include <string>
#include <fstream>
#include <iostream>

namespace MIOpenGEMM{
namespace outputwriting{
  
class Flusher{
  public:
    void increment(){};
}; 


class Endline{
  public:
    void increment(){};
}; 


class OutputWriter{

public: 

  bool to_terminal;  
  bool to_file;
  std::ofstream file;
  std::string filename;
  
  OutputWriter();
  ~OutputWriter();
  OutputWriter(bool to_terminal, bool to_file, std::string filename = "");
  void operator() (std::string);
  
    
  template <typename T>
  OutputWriter & operator<< (T t){
    
    if (to_terminal){
      std::cout << t;
    }
    
    if (to_file){
      file << t;
    }
    return *this;
  }
  
};


template <>
OutputWriter & OutputWriter::operator << (Flusher f);

template <>
OutputWriter & OutputWriter::operator << (Endline e);

}

extern outputwriting::Flusher Flush;
extern outputwriting::Endline Endl;

}


#endif
