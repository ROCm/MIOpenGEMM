

import numpy as np

C = [1,2,3,4,5,6,8]

micro_tile_step  = {}
micro_tile_step[1] = [1,2]
micro_tile_step[2] = [1,2,3,4]
micro_tile_step[3] = [2,3,4,6]
micro_tile_step[4] = [2,3,4,5,6]
micro_tile_step[5] = [4,5,6,8]
micro_tile_step[6] = [4,5,6,8]
micro_tile_step[8] = [4,6,8]

goto = {}

for x in C:
  goto[x] = {}
  for y in C:
    goto[x][y] = []
    for nx in micro_tile_step[x]:
      for ny in micro_tile_step[y]:

	#eliminate type 1 skinny micro-tiles
	condition1 = np.abs(nx - ny) <= 4
	
	#eliminate too dramatic changes is skinniness
	delta_ratio = (float(x)/float(y)) / (float(nx)/float(ny))
	skininess_change_good = delta_ratio <= 2 and delta_ratio >= 0.5
	
	#eliminate too dramatic changes in volume unless going to an `even hub' 
	delta_volume = (float(x)*float(y)) / (float(nx)*float(ny))
	volumn_change_good = (nx%2 == 0 and ny%2 == 0) or (delta_volume <= 2 and delta_volume >= 0.5)
	
	#you can only go to 5,8 from 4,8
	condition_not_58 = (x == 4 and y == 8) or (x == 8 and y == 4) or  (not (nx == 5 and ny == 8) and not (ny == 8 and ny !== 5))
	
	if condition1 and skininess_change_good and volumn_change_good and condition_not_58: #condition4 and condition5 : #eliminate too dramatic changes in skininess 
	  goto[x][y].append([nx, ny])
    
  
for x in C:
  for y in C:
    #if np.abs(x - y) <= 4:
    inners = ", ".join(["{%d, %d}"%(xn, yn) for xn, yn in goto[x][y]])
    print "micro_tile_edges [ {%d,%d} ] = { %s };"%(x,y,inners) #{1,4}, {2,4} };
    x, y, goto[x][y]
  



