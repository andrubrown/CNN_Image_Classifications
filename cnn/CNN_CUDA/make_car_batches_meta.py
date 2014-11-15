'make cuda-convnet batches from images in the input dir; start numbering batches from 7'
# To use this script, we should modify the image size, batch size, labels. Then, write our own data provider.
# Set labels according to the file names
# 5 fold split
import os
import sys
import numpy as np
import cPickle as pickle
import random
from natsort import natsorted
from PIL import Image
from PIL import ImageOps

def main():
	output_dir = sys.argv[1]
	filename = "batches.meta"
	path = os.path.join( output_dir, filename )
	label_names = ['Car','Pickup','MiniVan','SUV']
	batch = {'label_names':label_names}
	pickle.dump( batch, open( path, "wb" ))

if __name__ == '__main__':
	main()
