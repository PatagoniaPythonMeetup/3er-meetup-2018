import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import skimage.transform as trf
import os
import imageio
import torch.utils.data as utils
import torch
import gzip


def to_tensor(data, b_size=20):
    """ Return a 'dataset' and a 'Dataloader'(respectively) ready to be used for pytorch nets """
    tensor_x = torch.stack( [torch.from_numpy(i).unsqueeze(0) for i in data] ) # transform to torch tensors
    x_train_data = utils.TensorDataset( tensor_x ) # create your datset
    x_train_loader = utils.DataLoader( x_train_data, batch_size=b_size ) # create your dataloader    
    return x_train_data, x_train_loader

def to_numpy(data):
    new_data = []
    for d in data:
        tmp = d[0].squeeze(1).numpy()
        new_data.append( tmp )
    return np.array( new_data )


def plot_n_faces( data, target, n=5, predicted=None, h_size=3 ):
    """ Plot 'n' correspondig items from 'data', 'target' and 'predicted' (if not None).
        'h_size' is the heigth of heach figure. """
    if predicted is None:
        n_ax = 2
    else:
        n_ax = 3
    for i in range(n):
        k = 1
        j = np.random.randint( data.shape[0] )
        f = plt.figure(figsize=((h_size*n_ax)+1, h_size))
        ax = f.add_subplot(1, n_ax, 1)
        ax.set_title("data")
        ax.imshow( data[j] , cmap=plt.cm.gray)

        ax = f.add_subplot(1, n_ax, 2) 
        ax.set_title("target")
        ax.imshow( target[j] , cmap=plt.cm.gray)
        
        if n_ax==3:
            ax = f.add_subplot(1, n_ax, 3) 
            ax.set_title("predicted")
            ax.imshow( predicted[j] , cmap=plt.cm.gray)
        plt.show()
    return
    
    
def data_loader(path, shape=(56, 56)):
    """ Load all the images contained in the folder specified for 'path', and reshape them to 'shape' size """
    images = []
    folders = os.listdir(path)
    for f in folders:
        if not os.path.isdir( "/".join( [path, f] ) ):
            continue
        imgs = os.listdir("/".join( [path, f] ) )
        for i in imgs:
            tmp = imageio.imread("/".join( [path, f, i] ))
            tmp = trf.resize( tmp, shape, mode="constant" )
            images.append( tmp )
    images = np.array( images )
    return images

def plot_learning( history, f_size=(12, 6) ):
    n_epochs = history['epochs']
    loss = history['loss']
    val_loss = history["val_loss"]
    
    f = plt.figure(fig_size=f_size)
    plt.plot(range(n_ephochs), loss, c="r", s="o-", label="loss")
    plt.plot(range(n_ephochs), val_loss, c="b", s="o-", label="loss")
    plt.legend()
    plt.show()
    return


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
