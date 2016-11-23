#ifndef DEFAULTOUTPATH_HPP
#define DEFAULTOUTPATH_HPP

namespace tinygemm{
namespace defpaths{

const std::string basewritedir = DIR_FOR_WRITING; // DIR_FOR_WRITING is defined in top most cmake file.
const std::string basekernelsdir = basewritedir + "/kernels";
const std::string scratchpadfinddir = basekernelsdir + "/scratchpadfind";
const std::string isacodedir = basewritedir + "/isa_code";
  
}
}


#endif 
