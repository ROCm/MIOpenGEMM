olds = open("../miopengemm/src/olddeepbench.cachetxt").read()

nct = "// clang-format off \n"

blobs = olds.split("kc.add(")


for blob in blobs[1::]:
  x = ""
  lines = blob.split()
  for l in lines:
    x += l
  
  x = x.replace(r'""', "")
  x = x.replace(r'/*', "\n")
  x = x.replace(r'*/', "\n")
  x = x.replace(r'{', "")
  x = x.replace(r'}', "")
  x = x.replace(r'",', r'"')
  x = x.replace(r');', r'')
  #x = x.replace(r'.', "\n")
  
  f = x.split("\n")
  dev = f[0]
  con = f[2]
  gg = f[4]
  soln = f[8]
  ggf = [r'"' + z[2::] + r'"' for z in soln[1:-1].split("__")]
  stats =f[10]
  fp = f[12]

  nct += "kc.add(\n{%s,  // dev\n{%s}, // con\n{%s}}, // gg\n{{{ // hp\n"%(dev, con, gg)
  nct += (ggf[0] + ",\n" + ggf[1] + ",\n"  + ggf[2] + '}},\n{ //stats\n %s, {%s}}};\n'%(stats, fp))
  nct += "\n\n"
  #print X, "\n\n"

nct += "// clang-format on"


filly = open("../miopengemm/src/deepbench.cachetxt", 'w')
filly.write(nct)
filly.close()

#print nct
