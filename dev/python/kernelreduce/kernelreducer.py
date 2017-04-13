fn = "/home/james/akernel_abc_alpha_beta.cl"

filly = open(fn)
lines = filly.readlines()
predefs = {}
for l in lines:
  if "#define" in l:
    frags = l.split()
    if len(frags) != 3:
      print "not 3 frags", frags
    predefs[frags[1]] = frags[2]

keys = predefs.keys()
keys.sort(lambda x,y : 2*(len(x) < len(y)) - 1)
for l in lines:
  if  r"/*" not in l and r"//" not in l and r"*/" not in l and "#define" not in l and l.isspace() == False and l.split()[0][0] != "*":
    
    for k in keys:
      l = l.replace(k, predefs[k])
    
    print l,
