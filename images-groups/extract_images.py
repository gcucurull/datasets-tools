'''
	This file extracts the train and test images for the age and gender
	Images of Groups dataset (http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html).
	Needed:
		- eventrain.mat
		- eventest.mat
'''

import scipy.io
from scipy import ndimage
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import getAngleBetweenPoints
import math
from scipy.misc import imsave

def get_filename(path):
	return path.split('\\')[-1]

def get_age_label(age):
	if age == 1:
		return 0
	elif age == 5:
		return 1
	elif age == 10:
		return 2
	elif age == 16:
		return 3
	elif age == 28:
		return 4
	elif age == 51:
		return 5
	elif age == 75:
		return 6
	else:
		assert False # this point should never be reached

def extract_images(stage, images):
	'''
		stage: 'train' or 'test'
	'''
	
	assert (stage == 'train' or stage == 'test')

	if stage == 'train':
		key = 'trcoll'
	else:
		key = 'tecoll'

	# load data
	evendata = scipy.io.loadmat('even'+stage+'.mat')
	data_filenames = evendata[key]['name'][0][0][0]
	data_facepos = evendata[key]['facePosSize'][0][0]
	data_age = evendata[key]['ageClass'][0][0]
	data_gen = evendata[key]['genClass'][0][0]
	n_images = data_filenames.shape[0]

	out_path = 'crop_dataset/images/'+stage+'/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	listfile_age = []
	listfile_gender = []
	for i in range(n_images):
		image_path = data_filenames[i][0] # get the path for the image
		image_pos = data_facepos[i] # get the position of the face
		filename = get_filename(image_path)
		age = data_age[i][0]
		age = get_age_label(age) # get the label for an ange value
		gender = data_gen[i][0]-1

		print(filename)
		assert filename in images # image exists in the downloads
		im = mpimg.imread(data_path+filename)

		center_x = image_pos[4]
		center_y = image_pos[5]
		eyes_dist = image_pos[6]
		face = im[max(0,center_y-eyes_dist*2):min(im.shape[0], center_y+eyes_dist*2), max(0,center_x-eyes_dist*2):min(im.shape[1], center_x+eyes_dist*2), :]
		print(face.shape)

		origin = (center_y-eyes_dist*2, center_x-eyes_dist*2)

		left_eye = [image_pos[1]-origin[0], image_pos[0]-origin[1]]
		right_eye = [image_pos[3]-origin[0], image_pos[2]-origin[1]]

		# plt.plot(left_eye[1], left_eye[0], 'ro')
		# plt.plot(right_eye[1], right_eye[0], 'ro')

		radians = getAngleBetweenPoints(left_eye[1], left_eye[0], right_eye[1], right_eye[0])
		degrees = math.degrees(radians)-360

		rotated_im = ndimage.rotate(face, degrees, reshape=False)
		# plt.imshow(rotated_im)

		imsave(out_path+str(i)+'_face_'+filename, rotated_im)
		listfile_age.append(str(i)+'_face_'+filename+' '+str(age))
		listfile_gender.append(str(i)+'_face_'+filename+' '+str(gender))

		# plt.show()

	# create listfiles
	list_path = 'crop_dataset/listfiles/'
	if not os.path.exists(list_path):
		os.makedirs(list_path)

	with open(list_path+stage+'_age.txt', 'w') as f:
		for line in listfile_age:
			f.write(line+'\n')

	with open(list_path+stage+'_gender.txt', 'w') as f:
		for line in listfile_gender:
			f.write(line+'\n')

# Load the filenames of the downlaoded images
data_path = "./raw_dataset/images/"
images = []
for image in os.listdir(data_path):
	images.append(image)

extract_images('train', images)
extract_images('test', images)