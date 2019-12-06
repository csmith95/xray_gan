import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import hyperparams
import os
import pickle
from PIL import Image

def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, list_IDs, input_transform, labels, data_dir):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.data_dir = data_dir
		self.input_transform = input_transform
		# assert len(labels) == len(list_IDs)
		assert data_dir != None and os.path.exists(data_dir)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		ID = self.list_IDs[index]

		# Load data and get label
		data_dir = self.data_dir
		if (ID.startswith('fake_')):
			data_dir = '../data/generated_images/'
		X = image_loader('{}/{}'.format(data_dir, ID))
		y = self.labels[ID]

		if self.input_transform is not None:
		    X = self.input_transform(X)

		return X, y


print("Initializing Datasets and Dataloaders...")

path_prefix = 'sample_' if hyperparams.use_sample_data else ''

# Create train and validation Datasets
# first load list of image IDs (image filenames)
image_IDs = {}
for split_name in ['train', 'val', 'test']:
	with open('../{}data/{}_image_ids.data'.format(path_prefix, split_name), 'rb') as filehandle:
		image_IDs[split_name] = pickle.load(filehandle)

labels = None
with open('../{}data/{}labels.data'.format(path_prefix, path_prefix), 'rb') as filehandle:
	labels = pickle.load(filehandle)
	assert labels != None

# construct Dataset objects
# resize, convert to tensor, and normalize images. 
# these normalization values are from pretrained models
data_transforms = transforms.Compose([
	transforms.Resize(hyperparams.input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # normalize grayscale RGB to range [-1, 1]
])
torchvision.set_image_backend('accimage')

if hyperparams.use_generated_data:
	# add fake data to train data IDs
	with open('../data/generated_image_ids.data', 'rb') as f:
    		filenames = pickle.load(f)
    		image_IDs['train'] += filenames
    		print('added {} generated images to training set'.format(len(filenames)))

print('Num train images: ', len(image_IDs['train']))
print('Num val images: ', len(image_IDs['val']))
print('Num test images: ', len(image_IDs['test']))

data_dir = '../{}data/{}images/'.format(path_prefix, path_prefix)
image_datasets = { split_name : Dataset(image_IDs[split_name], data_transforms, labels, data_dir)\
	for split_name in ['train', 'val', 'test'] }


# Create training and validation Dataloaders
dataloader_params = {
	'batch_size': hyperparams.batch_size,
	'num_workers': hyperparams.num_workers,
	'shuffle': True
}

dataloaders = { split_name : data.DataLoader(image_datasets[split_name], **dataloader_params) for split_name in ['train', 'val', 'test']}

