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
  
// TODO : remove p_enma, go direct from emat. Or some type of inheritance for all "subs" ?


std::vector<size_t> get_hy_v(std::string hy_s, bool hy_s_full, Mat::E emat);
std::string get_str(Mat::E emat, const std::vector<size_t> & vs);

class Constraint{
  public:
  
  Mat::E emat;
  const EnumMapper<std::string> *  p_enma;
  std::vector<size_t> range;
  std::vector<size_t> start_range;
  Constraint() = default;
  Constraint(Mat::E);
  Constraint(Mat::E, const std::string & r);
  Constraint(Mat::E, const std::string & r, const std::string & sr);
  std::string get_r_str() const;
  std::string get_sr_str() const;
};

class Constraints{
  public:
  using str_array = std::array<std::string, Mat::E::N>;
  std::array<Constraint, Mat::E::N> sub;  
  Constraints(const str_array & r);
  Constraints(const std::string & rconcat);
  Constraints(const str_array & r, const str_array & sr);
  // TODO strings ?
};


class SuHy
{

  public:
  // TODO : duplicated using.
  using str_array = std::array<std::string, Mat::E::N>;

  Mat::E emat;
  const EnumMapper<std::string> *  p_enma;
  std::vector<size_t> vs;
  
  std::string get_string() const;
  bool operator==(const SuHy& rhs) const;
  void replace_where_defined(const Constraint & constraint);
  void checks() const;  // TODO
  SuHy() = default;
  SuHy(Mat::E);
  SuHy(Mat::E, const std::string &);
  SuHy(Mat::E, std::vector<size_t> && vs); 
};


class HyPas
{

  public:
  // TODO : duplicated usage. (inheritance...)
  
  using str_array = std::array<std::string, Mat::E::N>;
  
  HyPas(const str_array &);
  HyPas(std::array<SuHy, Mat::E::N> &&);
  HyPas(const HyPas &) = default;  
  std::array<SuHy, Mat::E::N> sus;
  void replace_where_defined(const Constraints & constraints);
  std::string get_string() const; 
  bool operator==(const HyPas& rhs) const;
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
  void apply_constraint(); // TODO
  void add_v_string(std::stringstream & ss, const std::vector<size_t> & values) const;
  void ss_init(size_t, std::stringstream &, std::string) const;
  
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
  
  // all the possible values of where hyper parameter can go
  // example : range[Chi::E::MIC] --> {1,2,3,4,5,6,7,8}
  std::vector<std::vector<size_t>> range;
  
  // a subset of range, the possible values returned on a request for a random value
  // example : start_range[Chi::E::MIC] --> {2,8}. It can depend on geometry (from initialisation)
  std::vector<std::vector<size_t>> start_range;

  void initialise();
   
  std::string get_string(size_t hpi) const;
  std::string get_edges_string(size_t hpi) const;
  std::string get_range_string(size_t hpi) const;
  std::string get_start_range_string(size_t hpi) const;
  bool contains(size_t hpi, size_t val) const;
  bool contains(const SuHy &) const;
  SuHy get_random_start() const;
  void checks() const; // TODO
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
  
  
  std::vector<std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>> p_coupled;

  
  ASuGr asubg;
  BSuGr bsubg;
  CSuGr csubg;
  const SuGr & at(size_t emat) const;    
  // if you need a non-const version, page 23 of Meyers.

  public:
  
  Graph(const Geometry &, const oclutil::DevInfo &, const Constraints &);   
  
  
  bool contains(Mat::E, size_t hpi, size_t value); 
  bool contains(const HyPas &); 
  std::vector<HyPas> get_one_aways(const HyPas &);
  std::vector<HyPas> get_mic_mac_transformed(const HyPas &);
  std::vector<HyPas> get_p_coupled_away(const HyPas &);
  std::vector<HyPas> get_neighbors(const HyPas &);
  HyPas get_random_start();
  void checks() const;
  
};




}


#endif
