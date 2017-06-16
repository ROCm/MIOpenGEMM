#ifndef FINDPARAMS_HPP
#define FINDPARAMS_HPP

#include <vector>
#include <string>
namespace MIOpenGEMM{


enum SummaryStat {Mean=0, Median, Max, nSumStatKeys};

std::string get_sumstatkey(SummaryStat sumstat);

class FindParams{
public:
  float allotted_time;
  unsigned allotted_descents;
  unsigned n_runs_per_kernel;
  SummaryStat sumstat;
  FindParams(float allotted_time, unsigned allotted_descents, unsigned n_runs_per_kernel, SummaryStat sumstat);
  FindParams() = default;
  std::string get_string() const;
};


  
}

#endif
