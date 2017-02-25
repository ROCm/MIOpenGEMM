#ifndef FORALLGENERATOR_HPP
#define FORALLGENERATOR_HPP


#include <tinygemm/prepgenerator.hpp>

namespace tinygemm{
namespace bylinegen{

///* TODO : these could be made hyper parameters, (1, 2, 4), (64, 128, 256)  */
//const size_t work_per_thread = 2;
//const size_t n_work_items_per_group = 256;


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

  size_t get_work_per_thread(){
    //could be made a hyperparam
    return 2;
  }

  size_t get_local_work_size() override final{
    //could be made a hyperparam
    return 256;
  }


  size_t get_n_work_groups() override final;

public:
  ByLineGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string type_);

  KernelString get_kernelstring() final override;
  void setup() final override;

private:

  void append_description_string(std::stringstream & ss);
  //void append_what_definitions(std::stringstream & ss);
  void append_how_definitions(std::stringstream & ss);
  void append_copy_preprocessor(std::stringstream & ss);
  void append_derived_definitions(std::stringstream & ss);

  void append_function_definition(std::stringstream & ss);
  void append_setup_coordinates(std::stringstream & ss);
  void append_positioning_x_string(std::stringstream & ss);
  void append_inner_work(std::stringstream & ss);
  void append_work_string(std::stringstream & ss);
  void append_positioning_y_string(std::stringstream & ss);

protected:
  virtual void setup_additional() = 0;
  virtual void append_derived_definitions_additional(std::stringstream & ss) = 0;

};

KernelString get_beta_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copya_kernelstring(const tinygemm::hyperparams::HyperParams & hp,  const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

KernelString get_copyb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp);

}
}







#endif
