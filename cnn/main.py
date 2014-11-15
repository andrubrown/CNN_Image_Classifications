import cnnFunc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_img", help = "the input file name")
args = parser.parse_args()
input_img = str(args.input_img)

label = cnnFunc.classify(input_img)
print label
