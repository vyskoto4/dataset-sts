import sys
from collections import defaultdict
import random

if __name__ == "__main__":
  fname = sys.argv[1]
  cls_s = defaultdict(list)
  for line in open(fname):
      s,cls = line[:-1].split(": ")
      cls_s[cls].append(s)

  trainf = open('mintent_train.txt','wb')
  valf = open('mintent_valf.txt','wb')
  num_val = 2
  for cls, s_list in cls_s.iteritems():
      random.shuffle(s_list)
      print(s_list)
      for i in range(num_val):
          valf.write("%s: %s \n"%(s_list[i],cls))
      for i in range(num_val, len(s_list)):
          trainf.write("%s: %s \n"%(s_list[i],cls))
  trainf.close()
  valf.close()         
