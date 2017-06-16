#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/stringutilbase.hpp>

#include <iostream>
#include <sstream>

namespace MIOpenGEMM{
  
  std::string tgformat(const std::string& what_arg, std::string prefix, std::string suffix){
    std::stringstream fms;
    fms << "\n\n";
    unsigned l_terminal = 95;
    
    auto frags = stringutil::split(what_arg, "\n");
    
    std::string space;
    space.resize(2*l_terminal, ' ');
    std::string interspace("   ");
    
    unsigned line_current = 1;
    for (auto & x : frags){
      if (x.size() == 0){
        x = " ";
      }
      //else{
      
        
      unsigned p_current = 0;
      while (p_current < x.size()){
        auto l_current = std::min<unsigned>(l_terminal, x.size() - p_current);
        fms <<  prefix << interspace << x.substr(p_current, l_current) <<  space.substr(0, l_terminal - l_current) << interspace << suffix <<  " (" << line_current << ")\n";
        p_current += l_current;
        ++line_current;
    
      //}
      }
    }
    
    fms << "\n";
    
    return fms.str();
  }
  
  miog_error::miog_error(const std::string& what_arg):std::runtime_error(tgformat(what_arg, "MIOpenGEMM",  "ERROR")){}

  void miog_warning(const std::string & warning){
    std::cerr << tgformat(warning, "MIOpenGEMM",  "WARNING") << std::flush;
  }

}



