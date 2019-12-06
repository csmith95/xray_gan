import pickle
from pathlib import Path

# fetch current labels
labels = None
with open('./labels.data', 'rb') as f:
	labels = pickle.load(f)
	assert labels != None


print(labels)

# add generated data labels
#filenames = []
#for subdr, label in zip(['healthy', 'unhealthy'], [1, 0]):
#	paths = Path('./generated_images/{}/'.format(subdr)).glob('*.png')
#	for path in paths:
#		filename = str(path).split('/')[-1]
#		filenames.append(filename)
#		labels[filename] = label

#print(filenames)

#with open('./generated_image_ids.data', 'wb') as f:
 #   pickle.dump(filenames, f)

# overwrite labels with legit+generated labels
#with open('./labels.data', 'wb') as f:
    #pickle.dump(labels, f)

#print(labels)
