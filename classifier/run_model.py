import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import Adam
from sklearn.metrics import classification_report
from my_dataloaders import dataloaders
from models import ResNet18
import config
import time
import copy
import shutil
import os


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=config.num_epochs):
    since = time.time()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print('cuda available: ', use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    cudnn.benchmark = True      # allows optimization if inputs are of same size
    model = model.to(device)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    # set up tensorboard visualizationss
    path_prefix = 'sample_' if config.use_sample_data else ''
    log_dir = './{}logs_img_sz_{}/'.format(path_prefix, config.input_size)
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    n_iters = { 'train': 0, 'val': 0 }
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                should_log = (n_iters[phase] % 10) == 0 # log every 10 batches
                n_iters[phase] += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if should_log:
                        # record minibatch accuracies
                        acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                        writer.add_scalar('minibatch_loss/{}'.format(phase), loss, n_iters[phase])
                        writer.add_scalar('minibatch_accuracy/{}'.format(phase), acc, n_iters[phase])

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model with best accuracy on val set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, './best_model_wts_img_sz_{}.pt'.format(config.input_size))



            # record epoch accuracy
            writer.add_scalar('epoch_accuracy/{}'.format(phase), epoch_acc, epoch)


        print()


    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))


def test_model(model, dataloader):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print('cuda available: ', use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    cudnn.benchmark = True      # allows optimization if inputs are of same size
    model = model.to(device)

    model.load_state_dict(torch.load('./best_model_wts_img_sz_{}.pt'.format(config.input_size)))
    model.eval()

    # Iterate over data.
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds)
        all_labels.append(labels)
	break

    print(all_labels)
    print(all_preds)
    print(classification_report(all_labels, all_preds))

    # print('F1: {}'.format(f1_score(all_labels, all_preds, average="samples")))
    # print('Precision: {}'.format(precision_score(all_labels, all_preds, average="samples")))
    # print('Recall: {}'.format(recall_score(all_labels, all_preds, average="samples")))
    # print('Accuracy: {}'.format(accuracy_score(all_labels, all_preds, average="samples")))


# helper fn to set up resnet18 classifier
def init_model():

    # train all params in model
    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = True

    model = ResNet18() # use custom implementation of resnet18 for grayscale inputs & 2 outputs
    set_parameter_requires_grad(model)

    return model


############################################ Program Starts Here ###############################################


### set up model
model = init_model()

### set up optimizer (using Adam here)
# note that this needs to be updated if we decide to do fine tuning (train all params)
# instead of feature extraction (train last fc layer)
print('Params to learn: ')
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print('\t', name)
print('\n\n')

optimizer = Adam(params_to_update, lr=config.lr, weight_decay=config.weight_decay)

### set up loss function
loss_fn = nn.CrossEntropyLoss()

### finally, train model
if config.mode == 'train':
    train_model(model, dataloaders, loss_fn, optimizer)
else:
    test_model(model, dataloaders['train'])


