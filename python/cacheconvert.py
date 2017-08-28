"""
convert cache entries to new format. 
"""

olds = open("/home/james/miopengemm/MIOpenGEMM/miopengemm/src/deepbench1.cachetxt").read()
blobs = olds.split("kc.add(")

filly = open("newdeep.txt", "w")

filly.write("// clang-format off \n\n")

for blob in blobs[1::]:
  x = ""
  new_blob = "kc.add(\n" + blob.split("{ //stats")[0].strip()[0:-1] + ");\n\n" 
  new_blob = new_blob.replace("{{{", "{{")
  filly.write(new_blob)


filly.write("// clang-format on \n\n")


filly.close()

