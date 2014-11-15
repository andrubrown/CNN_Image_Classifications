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

def process( image ):
	image = np.array( image )           # 224 x 224 x 3
	image = np.rollaxis( image, 2 )     # 3 x 224 x 224
	image = image.reshape( -1 )         # 150528
	return image
	
def get_batch_path( output_dir, number ):
	filename = "data_batch_{}".format( number )
	return os.path.join( output_dir, filename )

# change here
def get_empty_batch():	
	return np.zeros(( 150528, 0 ), dtype = np.uint8 )
	
def write_batch( path, batch , labels):
	print "writing {}...\n".format( path )
	d = { 'labels': labels, 'data': batch }
	pickle.dump( d, open( path, "wb" ))
	
# change here
def get_label(file_name):
    tag = file_name.strip().split('.')[0].split('_')[5]
    if tag == 'C':
	    return 0
    elif tag == 'P':
        return 1
    elif tag == 'MV':
	    return 2
    elif tag == 'S':
	    return 3

def main():
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	try:
		batch_counter = int( sys.argv[3] )
	except IndexError:
		batch_counter = 1
# change here	
	batch_size = 800
	my_lable = []
	print len(my_lable)
	print "reading file names..."
	names = [ d for d in os.listdir( input_dir ) if d.endswith( '.png') ]
	names = natsorted( names )

	if batch_counter > 1:
		omit_batches = batch_counter - 1
		omit_images = omit_batches * batch_size
		names = names[omit_images:]
		print "omiting {} images".format( omit_images )

	current_batch = get_empty_batch()
	count = 0
	for n in names:
	
		image = Image.open( os.path.join( input_dir, n ))
		try:
			image = process( image )
		except ValueError:
			print "problem with image {}".format( n )
			sys.exit( 1 )
        # set label to this image
		my_lable.append(get_label(n))
		count += 1
	
		image = image.reshape( -1, 1 )
		current_batch = np.hstack(( current_batch, image ))
		if count % 100 == 0:
			print get_label(n)
			print n
			print len(my_lable)
			print current_batch.shape[1]
		if current_batch.shape[1] == batch_size:
			batch_path = get_batch_path( output_dir, batch_counter )
			write_batch( batch_path, current_batch , my_lable)
			
			batch_counter += 1
			current_batch = get_empty_batch()
	print len(my_lable)
		

	

if __name__ == '__main__':
	main()
