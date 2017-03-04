from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import pickle
import hashlib

train_folders=['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
test_folders=['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']

import random
#def disp_samples(train_folders,sample_size):
# 	for folder in train_folders:
# 		print(folder)
# 		image_file=os.listdir(folder)
# 		image_sample=random.sample(image_file,sample_size)
# 		for image in image_sample:
# 			image_file=os.path.join(folder,image)
# 			i=Image(filename=image_file)
# 			display(i)
#
#disp_samples(train_folders, 1)

image_size=28
pixel_depth=255.0

def load_letter(folder,min_num_images):
	imgFiles=os.listdir(folder)
	dataset=np.ndarray(shape=(len (imgFiles),image_size,image_size),dtype=np.float32)
	print ('name is ',folder)
	num_images=0
	for image in imgFiles:
		image_file=os.path.join(folder,image)
		try:
			image_data=(ndimage.imread(image_file).astype(float)-pixel_depth/2)/pixel_depth
			if image_data.shape!=(image_size,image_size):
				raise Exception('image not in shape ',str(image_data.shape))
			dataset[num_images,:,:]=image_data
			num_images += 1
		except IOError as e:
			print('Could not read image ',image_file)
	dataset=dataset[0:num_images,:,:]
	if num_images<min_num_images:
		raise Exception('Fewer images than expected.')

	print('Full dataset: ',dataset.shape)
	print('Mean: ',np.mean(dataset))
	print('Standard Deviation: ',np.std(dataset))
	return dataset

def maybe_pickle(dataFolder,min_num_img_per_class,force=False):
	dataset_names=[]
	# print ('datafolder ',dataFolder)
	for folder in dataFolder:
		# print ('folder ',folder)
		set_filename=folder+'.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			print ('File already exists: ', set_filename)
		else:
			print ('Pickling ',set_filename)
			dataset=load_letter(folder,min_num_img_per_class)
			try:
				f=open(set_filename,'wb')
				pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print ('Unable to save data')
	return dataset_names

trainDatasets=maybe_pickle(train_folders,45000)
testDatasets=maybe_pickle(test_folders,1800)
print ('trainDatasets ',trainDatasets)
print ('testDatasets ',testDatasets)
# print (matplotlib.backends.backend)

def disp_sample_pickles(train_folders):
	folder=random.sample(train_folders,1)
	pickle_filename=''.join(folder)+'.pickle'
	try:
		f=open(pickle_filename,'rb')
		dataset=pickle.load(f,encoding='latin1')
	except Exception as e:
		print ('Unable to openfile ',f,':',e)
		return
	plt.suptitle(''.join(folder)[-1])
	for i, img in enumerate(random.sample(list(dataset), 5)):
	    plt.subplot(2, 4, i+1)
	    plt.axis('off')
	    plt.imshow(img)

disp_sample_pickles(train_folders)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None

  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  print ('v size ',valid_size)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  print ('v labels ',valid_labels)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f,encoding='latin1')
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(trainDatasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(testDatasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape[0])
print('Validation:', valid_dataset.shape, valid_labels.shape[0])
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset,labels):
	perm=np.random.permutation(labels.shape[0])
	return dataset[perm,:,:],labels[perm]

train_dataset,train_labels=randomize(train_dataset,train_labels)
valid_dataset,valid_labels=randomize(valid_dataset,valid_labels)
test_dataset,test_labels=randomize(test_dataset,test_labels)

#pickle_file='notMNIST.pickle'
#try:
#	f=open(pickle_file,'wb')
#	save={'train_dataset': train_dataset,'train_labels': train_labels,'valid_dataset': valid_dataset,'valid_labels': valid_labels,
#		'test_dataset': test_dataset,
#	    'test_labels': test_labels}
#	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
#	f.close()
#except Exception as e:
#	print ('Unable to save data')
#
#MAX_MANHATTAN_DIST=10
#def displayOverlap(overlap,src_dataset,dest_dataset):
#	item=random.choice(overlap.keys())
#	images=np.concatenate(([src_dataset[item]],dest_dataset[overlap[item][0:5]]))
#	plt.suptitle(item)
#	for i,img in enumerate(images):
#		plt.subplot(2,4,i+1)
#		plt.axis('off')
#		plt.imshow(img)

#def extract_overlap(dataset1,dataset2,labels_1):
#	overlap={}
#	for i,img1 in enumerate(dataset1):
#		diff=dataset2-img1
#		norm=np.sum(np.abs(diff),axis=1)
#		duplicates=np.where(norm<MAX_MANHATTAN_DIST)
#		if len(duplicates[0]):
#			overlap[i]=duplicates[0]
#
#     print ('Done')
#     # return overlap
#	return np.delete(dataset1,overlap,0),np.delete(labels_1,overlap,None)


def remove_overlap(dataset_1, dataset_2, labels_1):
  dataset_hash_1 = np.array([hashlib.md5(img).hexdigest() for img in dataset_1])
  dataset_hash_2 = np.array([hashlib.md5(img).hexdigest() for img in dataset_2])
  overlap = []
  for i, hash1 in enumerate(dataset_hash_1):
    duplicates = np.where(dataset_hash_2 == hash1)
    if len(duplicates[0]):
      overlap.append(i) 
  return np.delete(dataset_1, overlap, 0), np.delete(labels_1, overlap, None)

#test_flat = test_dataset.reshape(test_dataset.shape[0],28*28)
#train_flat = train_dataset.reshape(train_dataset.shape[0],28*28)
#valid_flat = valid_dataset.reshape(valid_dataset.shape[0],28*28)
#test_flat,test_labels=extract_overlap(test_flat[:200],train_flat,test_labels)
test_dataset_good, test_labels_good = remove_overlap(test_dataset, train_dataset, test_labels)
print('Overlapping images removed from test: ', len(test_dataset) - len(test_dataset_good))

valid_dataset_good, valid_labels_good=remove_overlap(valid_dataset,train_dataset,valid_labels)
print ('Overlapping images removed from valid: ',len(valid_dataset)-len(valid_dataset_good))
# print ('Number of overlaps ',len(overlap_test_train.keys()))
# displayOverlap(overlap_test_train,test_dataset[:200],train_dataset)

#good_pickle_file='notMNIST_good.pickle'
#try:
#	f=open(good_pickle_file,'wb')
#	save={'train_dataset': train_dataset,'train_labels': train_labels,'valid_dataset': valid_dataset_good,'valid_labels': valid_labels_good,
#		'test_dataset': test_dataset_good,
#	    'test_labels': test_labels_good}
#	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
#	f.close()
#except Exception as e:
#	print ('Unable to save data')
 
pretty_labels={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}
def disp_labels(dataset,labels):
    items=random.sample(range(len(labels)),5)
    for i,item in enumerate(items):
        plt.subplot(2,4,i+1)
        plt.axis('off')
        plt.title(pretty_labels[labels[item]])
        plt.imshow(dataset[item])
     
 
sample_size=5000
clf=LogisticRegression()
X_test=test_dataset_good.reshape(test_dataset_good.shape[0],28*28)
y_test=test_labels_good
X_train=train_dataset[:sample_size].reshape(sample_size,784)
y_train=train_labels[:sample_size]
clf.fit(X_train,y_train)
print ('Accuracy is: ',clf.score(X_test,y_test))
pred_labels=clf.predict(X_test)
disp_labels(test_dataset_good,pred_labels)