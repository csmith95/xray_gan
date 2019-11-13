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
        return img.convert('RGB')


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
		X = image_loader('{}/{}'.format(self.data_dir, ID))
		y = self.labels[ID]

		if self.input_transform is not None:
		    X = self.input_transform(X)

		return X, y


print("Initializing Datasets and Dataloaders...")

# Create train and validation Datasets
# first load list of image IDs (image filenames)
image_IDs = {}
for split_name in ['train', 'val']:
	with open('../sample_data/{}_image_ids.data'.format(split_name), 'rb') as filehandle:
		image_IDs[split_name] = pickle.load(filehandle)

labels = None
with open('../sample_data/sample_labels.data', 'rb') as filehandle:
	labels = pickle.load(filehandle)
	assert labels != None

# construct Dataset objects
# resize, convert to tensor, and normalize images. 
# these normalization values are from pretrained models
data_transforms = transforms.Compose([
	transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
torchvision.set_image_backend('accimage')
data_dir = '../sample_data/sample_images'	
image_datasets = { split_name : Dataset(image_IDs[split_name], data_transforms, labels, data_dir)\
	for split_name in ['train', 'val'] }

# Create training and validation Dataloaders
dataloader_params = {
	'batch_size': hyperparams.batch_size,
	'num_workers': hyperparams.num_workers,
	'shuffle': True
}

dataloaders = { split_name : data.DataLoader(image_datasets[split_name], **dataloader_params) for split_name in ['train', 'val']}

