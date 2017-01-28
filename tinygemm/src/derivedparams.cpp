#include <tinygemm/derivedparams.hpp>
#include <tinygemm/mapkeycheck.hpp>



namespace tinygemm{
namespace derivedparams{
  
class DerivedParams{
  public:
  
    
    
    std::map<std::string, std::string> string_params;
    std::map<std::string, unsigned> unsigned_params;

    std::string string_at(const std::string & key){
      if (string_params.count(key) == 0){
        throw tinygemm_error("The key `" + key  + "' does not appear in string_params of DerivedParams"); 
      }
      return string_params.at(key);
    }
      
    unsigned unsigned_at(const std::string & key){
      if (unsigned_params.count(key) == 0){
        throw tinygemm_error("The key `" + key  + "' does not appear in unsigned_params of DerivedParams"); 
      }
      return unsigned_params.at(key);
    }
    
    void check(){
      mapkeycheck::check_map_keys(string_params, all_derived_string_param_names, "DerviedParams, string_params against all_derived_string_param_names");
      mapkeycheck::check_map_keys(unsigned_params, all_derived_unsigned_param_names, "DerviedParams, unsigned_params against all_derived_unsigned_param_names");
    }
  
};


}
}
