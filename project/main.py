import sys

from week1 import run_w1 as w1

if __name__=="__main__":

    if "--week1" in sys.argv:
        w1.run()
