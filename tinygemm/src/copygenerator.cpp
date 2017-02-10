#include <string>
#include <sstream>

#include <tinygemm/copygenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>


namespace tinygemm{
namespace copygen{


static const std::string genericcopyakernelname = "tg_copya";


class CopyGenerator{

private:
  const hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  const derivedparams::DerivedParams & dp;  

public:
  CopyGenerator(const hyperparams::HyperParams & hp_, const tinygemm::TinyGemmGeometry & gg_, const derivedparams::DerivedParams & dp_): hp(hp_), gg(gg_), dp(dp_) {}


  size_t get_local_work_size(){
    return 911;
  }


  size_t get_global_work_size(){
    return 911;
  }


  KernelString get_a_kernelstring(){
    std::stringstream ss;
    ss << "under dev" ;
    return {"copy_a", ss.str(), genericcopyakernelname, get_global_work_size(), get_local_work_size()};
  }

};


KernelString get_copy_a_kernelstring(const hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const derivedparams::DerivedParams & dp){
 CopyGenerator cg(hp, gg, dp);
 return cg.get_a_kernelstring();
}

}
}







