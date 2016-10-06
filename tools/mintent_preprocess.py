import sys
from collections import defaultdict
import random
import re

def prepare(fname,trainf,valf):
  cls_s = defaultdict(list)
  for line in open(fname):
      s,cls = line[:-1].split(": ")
      s=re.sub( r"([0-9]+)", "!CENA!", s)
      cls_s[cls].append(s)

  trainf = open(trainf,'wb')
  valf = open(valf,'wb')
  num_val = 2
  for cls, s_list in cls_s.iteritems():
      random.shuffle(s_list)
      for i in range(num_val):
          valf.write("%s: %s\n"%(s_list[i],cls))
      for i in range(num_val, len(s_list)):
          trainf.write("%s: %s\n"%(s_list[i],cls))
  trainf.close()
  valf.close()         


if __name__ == "__main__":
  fname, trainf,valf = sys.argv[1:3]
  prepare(fname,trainf,valf)
