import json
import sys

if __name__ == "__main__":

  filename = "C:\\Users\\HLS501\\Desktop\\pyCGM2.info" #sys.argv[1]
  inputs = json.loads(open(str(filename)).read())
