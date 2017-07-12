/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_GRAPHHHH_HPP
#define GUARD_MIOPENGEMM_GRAPHHHH_HPP

#include <array>
#include <functional>
#include <map>
#include <vector>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/oclutil.hpp>

namespace MIOpenGEMM
{



std::vector<size_t>
get_constraints(std::string subg_cs, bool subg_csfull, const EnumMapper<std::string> * p_kv, char subg_hash);


class Constraint{
  // TODO

};

using Constraints = std::array<Constraint, Mat::E::N>;


class SuHy
{

  public:
  const EnumMapper<std::string> *  p_enma;
  std::vector<size_t> vs;
  std::string get_string() const; // TODO
  void replace_where_defined(const Constraint & constraint); // TODO
  void checks() const;  // TODO
  SuHy() = default;
  SuHy(Mat::E);
  SuHy(Mat::E, const std::string &);
  
};


class HyPas
{

  public:
  HyPas(const std::array<std::string, Mat::E::N> &);
  std::array<SuHy, Mat::E::N> sus;  
  void replace_where_defined(const Constraints & constraints);
  std::string get_string() const; // TODO
  void checks() const;

};


class SuGr
{
  private:
  // void initialise consists of the following, in order
  virtual void initialise_edges() = 0;
  void initialise_range();
  void initialise_start_range();
  // certain geometries should not start at certain nodes, this function prunes
  virtual void refine_start_range() = 0; 
  void apply_constraints(); // TODO

  
  public:
  

  SuGr(Mat::E emat,
      const Geometry & gg,
      const Constraint & constraint,
      const oclutil::DevInfo& devinfo);

  Mat::E emat;
  const EnumMapper<std::string> *  p_enma;
  const Geometry * ptr_gg;
  const Constraint * ptr_constraint;
  const oclutil::DevInfo* ptr_devinfo;
  
  // all the possible edges from all the possible hyper parameter
  // example : edges[Chi::E::MIC] is a map; edges[Chi::E::MIC][1] --> {2,3,4}
  std::vector<std::map<size_t, std::vector<size_t>>> edges;
  
  // all the possible values of a hyper parameter
  // example : range[Chi::E::MIC] --> {1,2,3,4,5,6,7,8}
  std::vector<std::vector<size_t>> range;
  
  // a subset of range, the possible values returned on a request for a random value
  // example : start_range[Chi::E::MIC] --> {2,8}. It can depend on geometry (from initialisation)
  std::vector<std::vector<size_t>> start_range;

  void initialise();
   
  std::string get_string(size_t hpi); // TODO
  std::string get_edges_string(size_t hpi); // TODO
  std::string get_range_string(size_t hpi); // TODO
  std::string get_start_range_string(size_t hpi); // TODO
  
  bool contains(size_t hpi, size_t val);
  bool contains(const SuHy &);
};

class CSuGr : public SuGr
{
  
  private:
  virtual void initialise_edges() override final; 
  virtual void refine_start_range() override final; 
    
  public:
  CSuGr(const Geometry & gg, const Constraint & constraint, const oclutil::DevInfo& devinfo);   

};

class ChiSuGr : public SuGr
{
  private:
  virtual void initialise_edges() override final; 
  virtual void refine_start_range() override final; 
  void set_start_mic(); 

  public:
  ChiSuGr(Mat::E, const Geometry &, const Constraint &, const oclutil::DevInfo&);  
};

class ASuGr : public ChiSuGr
{
  public:
  ASuGr(const Geometry &, const Constraint &, const oclutil::DevInfo&); 
};

class BSuGr : public ChiSuGr
{
  public:
  BSuGr(const Geometry &, const Constraint &, const oclutil::DevInfo&); 
};


class Graph
{

  private:
  
  ASuGr asubg;
  BSuGr bsubg;
  CSuGr csubg;
  SuGr & at(size_t emat);    

  public:
  Graph(const Geometry &, const oclutil::DevInfo &, const Constraints &);   
  
  
  bool contains(Mat::E, size_t hpi, size_t value); 
  bool contains(const HyPas &); 
  std::vector<HyPas> get_one_aways(const HyPas &); // TODO
  HyPas get_random(); // TODO
  
};




}


#endif
