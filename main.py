import sys

from week1 import run_w1 as w1
from week2 import run_w2_1 as w2_1
from week2 import run_w2 as w2

if __name__=="__main__":

    if "--week1" in sys.argv:
        w1.run()

    if "--week2" in sys.argv:
        if len(sys.argv)>2:
            rows = cols = int(sys.argv[2])
            w2_1.run(rows=rows, cols=cols)
        else:
            w2.run()
