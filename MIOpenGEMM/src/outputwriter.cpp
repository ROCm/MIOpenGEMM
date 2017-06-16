#include <iostream>
#include <string>
#include <MIOpenGEMM/error.hpp>

#include <MIOpenGEMM/outputwriter.hpp>

namespace MIOpenGEMM{
namespace outputwriting{

OutputWriter::OutputWriter(bool to_terminal_, bool to_file_, std::string filename_):to_terminal(to_terminal_), to_file(to_file_), filename(filename_){


  if (to_file == true){
    if (filename.compare("") == 0){
      throw miog_error("empty filename passed to OutputWrite, with to_file flag true. This is not possible.");
    }
    
    file.open(filename, std::ios::out);
     
    if (file.good() == false){
      std::string errm = "bad filename in constructor of OutputWriter object. The filename provided is `";
      errm += filename;
      errm += "'. The directory of the file must exist, OutputWriters do not create directories. Either create all directories in the path, or change the provided path.  ";
      throw miog_error(errm);
    }
  }
  

}


OutputWriter::~OutputWriter(){

  file.close();
}





template <>
OutputWriter & OutputWriter::operator << (Flusher f){
  f.increment();
  
  if (to_terminal){
    std::cout << std::flush;
  }
  
  if (to_file){
    file.flush();
  }
  return *this;
}


template <>
OutputWriter & OutputWriter::operator << (Endline e){
  e.increment();
  if (to_terminal){
    std::cout << std::endl;
  }
  
  if (to_file){
    file << "\n";
    file.flush();
  }
  return *this;
}


}

outputwriting::Endline Endl;
outputwriting::Flusher Flush;

}
