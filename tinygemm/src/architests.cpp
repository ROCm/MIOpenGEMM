#include <sstream>

#include <tinygemm/architests.hpp>
#include <tinygemm/openclutil.hpp>

namespace tinygemm{
namespace architests{

std::tuple<bool, std::string> architecture_specific_tests(cl_command_queue command_queue, const tinygemm::hyperparams::HyperParams & hp, const tinygemm::derivedparams::DerivedParams & dp){

  //not used currently
  (void)hp;
  
  std::stringstream status_ss;
  
  
  size_t max_work_group_size;
  
  openclutil::get_device_info_from_command_queue(command_queue, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), & max_work_group_size, nullptr, "getting CL_DEVICE_MAX_WORK_GROUP_SIZE");
  
  
  /* check -1 : macro tile not too large */
  if (dp.main_n_work_items_per_workgroup > max_work_group_size){
    status_ss << "n_work_items_per_workgroup > CL_DEVICE_MAX_WORK_GROUP_SIZE, ( " << dp.main_n_work_items_per_workgroup << " > " << max_work_group_size << " ) : cannot compile this kernel to this architecture \n";
  }
  
  
  
  std::string status_string = status_ss.str();
  
  return std::make_tuple(status_string.compare("") == 0, status_string);

}

}
}
