import os

import socket
write_base_directory = None

if socket.gethostname() == "duffy":
  write_base_directory = "/home/james/tinygemmout/"

elif socket.gethostname() =='james-All-Series':
  write_base_directory = "/home/james/tinygemmout/"

else:
  raise RuntimeError("Unrecognised hostname, please append write_base_directory info in write_directories")


output_base_directory = os.path.join(write_base_directory, "output")
plot_base_directory = os.path.join(write_base_directory, "plots")
kernels_base_directory = os.path.join(write_base_directory, "kernels")
baidu_base_directory = os.path.join(write_base_directory, "baidu")
