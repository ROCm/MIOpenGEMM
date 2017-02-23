import re



filly = open("hyperparams.cpp")
lines = filly.readlines()
filly.close()


newlines = []

c_keys = ["U", "GA", "PU", "ICE", "NAW", "UFO"]
chiral_keys = ["MAC", "MIC", "PAD", "PLU", "LIW", "MIW", "CTY"]

for l in lines:
  if "_UFO" in l:
    if "ice" in l:
      pass
    else:
      snip = l.split("\"Y")[-1]
      snip = snip.split("_NOF0")[0]
      snip = "Y" + snip + "_NOF0"
      
      oldparms = snip.split("_")
      d_oldparms = {}
      for x in oldparms:
        #if re.search("\d", x):
          ##print x, " yip"
        #else:
          ##print x, " nope"
        start = re.search("\d", x).start()
        d_oldparms[x[0:start]] = x[start::]
      #print d_oldparms
      
  
      vals = {'A':{}, 'B':{}, 'C':{}}

      vals['A']["MAC"] = d_oldparms['Y']
      vals['A']["MIC"] = d_oldparms['y']
      vals['B']["MAC"] = d_oldparms['X']
      vals['B']["MIC"] = d_oldparms['x']
              
      for X in ['A', 'B']:

        vals[X]["PAD"] = d_oldparms["P" + X]
        vals[X]["PLU"] = d_oldparms[X + "PLU"]
        vals[X]["LIW"] = d_oldparms["LIW"]
        vals[X]["MIW"] = d_oldparms["MIW"]
        vals[X]["CTY"] = d_oldparms[X + "CW"]
  
      for key in c_keys:
        vals['C'][key] = d_oldparms[key]
      
      new_string = ""
      for X in ["A", "B"]:
        if X == "B":
          new_string += "_"
        new_string += X 
        new_string += "_"
        for k in chiral_keys:
          new_string += k 
          new_string += vals[X][k]
          new_string += "_"
      for k in c_keys:
        new_string += "_"
        new_string += k 
        new_string += vals['C'][k]
       
      print new_string
          
      #print "\n"
      #print d_oldparms
      #print vals
    
  else:
    newlines.append(l)


  #Y32_X64_y2_x4_U32_PA1_PB1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ICE24_NAW64_UFO0_ACW0_BCW0_NOF0
  #to
  #A_()__B()__()





