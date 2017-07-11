/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP
#define GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <string>
#include <vector>
#include <memory>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{
namespace dev
{



template <typename TFloat>
class Moa
{

  public:
  Moa(Geometry                     gg_,
         Offsets                      toff_,
         const TFloat*                a_,
         const TFloat*                b_,
         const TFloat*                c_,
         outputwriting::OutputWriter& mowri_);         


  Moa(Geometry                     gg_,
         Offsets                      toff_,
         outputwriting::OutputWriter& mowri_);         
                  
  void benchgemm(const std::vector<std::string>& hyperstrings,
                 size_t                          max_number_of_runs,
                 double                          max_time_per_kernel);

  Solution find(const FindParams& find_params, std::string constraints_string);

  void accuracy_test(const std::string& hyperstring, const TFloat* c_true_for_test);

  ///////////////////////////////////

  private:
  Geometry gg;
  Offsets  toff;

  // a, b and c cpu memories.
  std::vector<const TFloat*> cpu_mem;

  //used when no pointer constructor is used.
  std::vector<std::vector<TFloat>> __cpu_mem;
  
  std::vector<TFloat>          c_copy;
  std::vector<TFloat>          c_for_cpu_compute;
  outputwriting::OutputWriter& mowri;

  openclutil::CommandQueueInContext tgcq;

  // a, b, c and workspace, gpu memories.
  std::vector<openclutil::SafeClMem> gpu_safemem;

  // sizes of a, b, c and workspace gpu memories.
  std::vector<size_t> mem_size;

  // read write permissions of gpu data TODO : move to enums.cpp
  std::vector<cl_mem_flags> rw_perms;

  size_t get_mat_memsize(Mem::E emem);

  size_t get_workspace_memsize();

  void opencl_memory_initialise();
  
  void initialise_cpu_mem_from_scratch();
  
  void initialise_cpu_mem(const TFloat * a_, const TFloat * b_, const TFloat * c_);
  
  void initialise_common();
  
  // delegator constructor.
  Moa(Geometry                     gg_,
         Offsets                      toff_,
         outputwriting::OutputWriter& mowri_,
         long);


};


class Goa{
  
  private:
  std::unique_ptr<Moa<double>> d_moa;
  std::unique_ptr<Moa<float>> f_moa;
  char active_type = '?';  
  
  
  template <typename TFloat>
  std::unique_ptr<Moa<TFloat>> & get_up_moa()
  {
    throw miog_error("unrecognised template parameter TFloat in Goa get_up_moa");
  }
  
  
  template <typename TFloat>
  void set_active_type(){
    throw miog_error("unrecognised template parameter TFloat in Goa set_active_type");
  }
  
  public:
  template <typename TFloat>
  Goa(Geometry                     gg_,
         Offsets                      toff_,
         const TFloat*                a_,
         const TFloat*                b_,
         const TFloat*                c_,
         outputwriting::OutputWriter& mowri_){
          get_up_moa<TFloat>().reset(new Moa<TFloat>(gg_, toff_, a_, b_, c_, mowri_));
          set_active_type<TFloat>();           
        }

     Goa(Geometry                     gg_,
         Offsets                      toff_,
         outputwriting::OutputWriter& mowri_){
         
           if (gg_.floattype == 'f'){
              f_moa.reset(new Moa<float>(gg_, toff_, mowri_));
           }
           else if (gg_.floattype == 'd'){
              d_moa.reset(new Moa<double>(gg_, toff_, mowri_));             
           }
          else{
            throw miog_error("unrecognised floattype char in Boa constructor");
          }
          active_type = gg_.floattype;
       }
       
       
       void benchgemm(const std::vector<std::string>& hyperstrings,
                 size_t                          max_number_of_runs,
                 double                          max_time_per_kernel){
          if (active_type == 'f'){
            f_moa->benchgemm(hyperstrings, max_number_of_runs, max_time_per_kernel);
          }
          else if (active_type == 'd'){
            d_moa->benchgemm(hyperstrings, max_number_of_runs, max_time_per_kernel);
          }
          else{
            throw miog_error("unrecognised floattype char in Boa benchgemm");
          }
        }

    Solution find(const FindParams& find_params, std::string constraints_string){
          if (active_type == 'f'){
            f_moa->find(find_params, constraints_string);
          }
          else if (active_type == 'd'){
            d_moa->find(find_params, constraints_string);
          }
          else{
            throw miog_error("unrecognised floattype char in Boa find");
          }
        }
      
      
      
  template <typename TFloat>
  void accuracy_test(const std::string& hyperstring, const TFloat* c_true_for_test){          
          get_up_moa<TFloat>->accuracy_test(hyperstring, c_true_for_test);
        }
};


  template <>
  std::unique_ptr<Moa<float>> & Goa::get_up_moa<float>();

  template <>
  std::unique_ptr<Moa<double>> & Goa::get_up_moa<double>();


  template <>
  void Goa::set_active_type<float>();

  template <>
  void Goa::set_active_type<double>();


//template <typename TFloat>
//void benchgemm(const std::vector<std::string>& hyperstrings,
               //size_t                          max_n_runs,
               //double                          max_time,
               //const Geometry&                 gg,
               //const Offsets&                  toff,
               //const TFloat*                   a,
               //const TFloat*                   b,
               //const TFloat*                   c,
               //outputwriting::OutputWriter&    mowri);

//template <typename TFloat>
//void accuracy_test(const std::string&           hyperstring,
                   //const Geometry&              gg,
                   //const Offsets&               toff,
                   //const TFloat*                a,
                   //const TFloat*                b,
                   //const TFloat*                c,
                   //const TFloat*                c_true_for_test,
                   //outputwriting::OutputWriter& mowri);

//template <typename TFloat>
//Solution find(const FindParams&            find_params,
              //const TFloat*                a,
              //const TFloat*                b,
              //const TFloat*                c,
              //std::string                  constraints_string,
              //const Geometry&              gg,
              //const Offsets&               toff,
              //outputwriting::OutputWriter& mowri);

//Solution basicfind(const FindParams& find_params,
                   //// no a, b, c.
                   //std::string                  constraints_string,
                   //const Geometry&              geometry,
                   //const Offsets&               toff,
                   //outputwriting::OutputWriter& mowri);



}
}

#endif
