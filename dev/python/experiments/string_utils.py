def padded(s, length):
  sin = s
  len_stw = len(s)
  if len_stw < length:
    s += "".join([" "]*(length - len_stw))
  s += "\t"
  return s
