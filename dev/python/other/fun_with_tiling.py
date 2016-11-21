
def get_tile_1(H, W, h, w):
  """
  This is a useless function
  """
  i = 0
  while H + W > 2:
    if H > W:
      H_over_2 = H/2
      
      if h >= H_over_2:
        i += H_over_2*W
        h -= H_over_2
        H -= H_over_2
        
      else:
        #h = (h - H - H/2)%H
        H = H_over_2
      #h = (h - H)%H
      
    else:
      W_over_2 = W/2
      if w >= W_over_2:
        i += W_over_2*H
        w -= W_over_2 #(w - W/2)%W
        W -= W_over_2
      else:
        #w = (w - W - W/2)%W
        W = W_over_2

      #w = (w - W)%W
      
  return i

def get_tile_from_n(H, W, n):
  h,w = n%H, n/H
  return get_tile_1(H, W, h, W)

#def get_tile_2(H, W, h, w):
  #i = 0
  #while H + W > 2:
    #if H > W:
      #if h >= H/2:
        #i += (H/2)*W
        #H -= H/2
      #else:
        #H /= 2
      #h %= H
      
    #else:
      #if w >= W/2:
        #i += (W/2)*H
        #W -= W/2
      #else:
        #W /= 2
      #w %= W
      
  #return i  
  

H = 55
W = 55

all_values = []
for h in range(H):
  values = []
  for w in range(W):
    print get_tile_1(H, W, h, w), " ",
    values.append(get_tile_1(H, W, h, w))

  all_values.append(values)
  
  print " ."


pl.matshow(np.array(all_values))
