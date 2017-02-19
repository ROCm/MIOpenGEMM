#ifndef PREPGENERATOR_HPP
#define PREPGENERATOR_HPP

#include <tinygemm/basegenerator.hpp>

namespace tinygemm{
namespace prepgen{

class PrepGenerator : public basegen::BaseGenerator{

protected:
  unsigned n_work_items;
  unsigned n_work_groups;
  char matrixchar; 
  void set_usage_from_matrixchar();
  void append_basic_what_definitions(std::stringstream & ss);

  virtual size_t get_local_work_size() = 0;

  virtual size_t get_n_work_groups() = 0;


  size_t get_global_work_size(){
    size_t forall_global_work_size = get_n_work_groups() * get_local_work_size();
    return forall_global_work_size;
  }


public:
  PrepGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string type_);

};

}
}
#endif

