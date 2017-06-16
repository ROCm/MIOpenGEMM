#ifndef FORALLGENERATOR_HPP
#define FORALLGENERATOR_HPP


#include <MIOpenGEMM/prepgenerator.hpp>

namespace MIOpenGEMM{
namespace bylinegen{



class ByLineGenerator : public prepgen::PrepGenerator {

private:
  unsigned n_full_work_items_per_line;
  unsigned n_work_items_per_line;
  unsigned n_full_work_items;
  unsigned start_in_coal_last_work_item;
  unsigned work_for_last_item_in_coal;

  
protected:
  std::string description_string;
  std::string inner_work_string;

  virtual size_t get_work_per_thread()  = 0;
  
  size_t get_n_work_groups() override final;

public:
  ByLineGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, std::string type_);

  KernelString get_kernelstring() final override;
  void setup() final override;

private:

  void append_description_string(std::stringstream & ss);
  void append_how_definitions(std::stringstream & ss);
  void append_copy_preprocessor(std::stringstream & ss);
  void append_derived_definitions(std::stringstream & ss);

  void append_setup_coordinates(std::stringstream & ss);
  void append_positioning_x_string(std::stringstream & ss);
  void append_inner_work(std::stringstream & ss);
  void append_work_string(std::stringstream & ss);
  void append_positioning_w_string(std::stringstream & ss);

protected:
  virtual void setup_additional() = 0;
  virtual void append_derived_definitions_additional(std::stringstream & ss) = 0;

};

}
}







#endif
