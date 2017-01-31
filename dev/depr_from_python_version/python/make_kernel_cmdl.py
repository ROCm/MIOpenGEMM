#!/usr/bin/python

import make_kernel
import sys

def main(argv):


  import getopt
  import inspect
  
  function_call_spec = inspect.getargspec(make_kernel.make_kernel)
  function_parms = function_call_spec[0]
  default_values = function_call_spec[3]
  parsed_options = getopt.getopt(sys.argv[1::], "", ['%s='%(o,) for o in  function_parms])[0]  
  parsed_options_dict = {}

  for po in parsed_options:
    parsed_options_dict[po[0].replace("--", "")] = po[1]
  
  for fp in function_parms:
    if fp not in parsed_options_dict.keys():
      print "Parameter %s not parsed in preparation for make_kernel call"%(fp,)
      raise RuntimeError("failed to make kernel source")
  
  for k in parsed_options_dict.keys():
    if k not in ["dir_name", "t_float", "kernelname"]:
      parsed_options_dict[k] = int(parsed_options_dict[k])
      
  
  
  #print parsed_options_dict
  
  
  make_kernel.make_kernel(**parsed_options_dict)

def get_parameter_list():
  function_call_spec = inspect.getargspec(make_kernel)
  function_parms = function_call_spec[0]
  return ",  ".join(["\"%s\""%(x,) for x in function_call_spec[0]])


if __name__ == "__main__":
   main(sys.argv[1:])


