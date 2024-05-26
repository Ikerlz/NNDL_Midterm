

import os

import numpy as np

import torch
import torch.nn as nn


from torch.utils.data import DataLoader
from torch.autograd import Variable
from conf import settings
from dataset.dataset import CUB_200_2011_Train, CUB_200_2011_Test
from torchvision.models import (
    vgg11, vgg13, vgg16, vgg19,
    resnet18, resnet34, resnet50, resnet101, resnet152,
    VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)


def get_network(args):
    model_dict = {
        'vgg11': (vgg11, VGG11_Weights),
        'vgg13': (vgg13, VGG13_Weights),
        'vgg16': (vgg16, VGG16_Weights),
        'vgg19': (vgg19, VGG19_Weights),
        'resnet18': (resnet18, ResNet18_Weights),
        'resnet34': (resnet34, ResNet34_Weights),
        'resnet50': (resnet50, ResNet50_Weights),
        'resnet101': (resnet101, ResNet101_Weights),
        'resnet152': (resnet152, ResNet152_Weights),
    }

    if args.net not in model_dict:
        raise ValueError("We only implement the vggs and resnets.")

    model_class, weights_enum = model_dict[args.net]

    if args.pretrained:
        weights = weights_enum.DEFAULT
    else:
        weights = None

    net = model_class(weights=weights)
    return net

def get_train_dataloader(path, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    train_dataset = CUB_200_2011_Train(
        path, 
        transform=transforms,
        target_transform=target_transforms
    )
    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_dataloader

def get_test_dataloader(path, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    test_dataset = CUB_200_2011_Test(
        path, 
        transform=transforms,
        target_transform=target_transforms
    )

    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return test_dataloader

def get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur
    
    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para
        
    return last_layer_weights, last_layer_bias

def visualize_network(writer, net):
    """visualize network architecture"""
    input_tensor = torch.Tensor(3, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE) 
    input_tensor = input_tensor.to(next(net.parameters()).device)
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

def visualize_lastlayer(writer, net, n_iter):
    """visualize last layer grads"""
    weights, bias = get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_train_loss(writer, loss, n_iter):
    """visualize training loss"""
    writer.add_scalar('Train/loss', loss, n_iter)

def visualize_param_hist(writer, net, epoch):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def visualize_test_loss(writer, loss, epoch):
    """visualize test loss"""
    writer.add_scalar('Test/loss', loss, epoch)

def visualize_test_acc(writer, acc, epoch):
    """visualize test acc"""
    writer.add_scalar('Test/Accuracy', acc, epoch)

def visualize_learning_rate(writer, lr, epoch):
    """visualize learning rate"""
    writer.add_scalar('Train/LearningRate', lr, epoch)

def init_weights(net, pretrained=False):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    try:
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, settings.CLASSES)
    except:
        fc_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(fc_features, settings.CLASSES)
    if pretrained:
        try:
            nn.init.xavier_uniform_(net.fc.weight)
            if net.fc.bias is not None:
                nn.init.constant_(net.fc.bias, 0)
        except:
            nn.init.xavier_uniform_(net.classifier[-1].weight)
            if net.classifier[-1].bias is not None:
                nn.init.constant_(net.classifier[-1].bias, 0)
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def mixup_data(x, y, alpha=0.2):

    """Returns mixed up inputs pairs of targets and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    index = index.to(x.device)

    lam = max(lam, 1 - lam)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = y
    y_b = y[index, :]

    return mixed_x, y_a, y_b, lam
