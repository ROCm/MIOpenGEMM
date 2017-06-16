#ifndef PREPGENERATOR_HPP
#define PREPGENERATOR_HPP

#include <MIOpenGEMM/basegenerator.hpp>

namespace MIOpenGEMM{
namespace prepgen{

class PrepGenerator : public basegen::BaseGenerator{

protected:
  unsigned n_work_items;
  unsigned n_work_groups;

  char matrixchar;
  char MATRIXCHAR;
  nsHP::eMat emat_x;
  
  void set_usage_from_matrixchar();
  void append_basic_what_definitions(std::stringstream & ss);

  virtual size_t get_local_work_size() = 0;
  virtual size_t get_n_work_groups() = 0;

  size_t get_global_work_size(){
    size_t forall_global_work_size = get_n_work_groups() * get_local_work_size();
    return forall_global_work_size;
  }

  void initialise_matrixtype(char matrixchar_in){
    if (matrixchar_in == 'a'){
      matrixchar = 'a';
      MATRIXCHAR = 'A';
      emat_x = nsHP::matA;
    }
    
    else if (matrixchar_in == 'b'){
      matrixchar = 'b';
      MATRIXCHAR = 'B';
      emat_x = nsHP::matB;
    }

    else if (matrixchar_in == 'c'){
      matrixchar = 'c';
      MATRIXCHAR = 'C';
      emat_x = nsHP::matC;
    }
    
    else{
      throw miog_error("in PrepGenerator : unrecognised matrixtype " + std::to_string(matrixchar_in));
    }
  }


public:
  PrepGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, std::string type_);

};

}
}
#endif

