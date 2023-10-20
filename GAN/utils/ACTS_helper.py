# %%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import glob

# %%
class ACTS(Dataset):
    def __init__(self, input_fn, transform=None, sample_size = None, min_fraction=0.001):
        data_list = []
        for input_file_name in glob.iglob(input_fn):
            print('Processing {}'.format(input_file_name))
            data = torch.from_numpy(np.load(input_file_name))
            data_list.append(data)

        self.pixel_hit = np.concatenate(data_list)
        self.pixel_hit = self.pixel_hit.transpose([0,2,3,1])
        self.pixel_hit = self.pixel_hit.astype('float32')

        ### Preprocessing of the dataset
        # remove events with zero energy deposition
        self.total_hits = self.pixel_hit.sum(axis = (1,2,3))
        self.total_hits = self.total_hits.reshape(-1, 1)
        index_to_remove = np.where(self.total_hits == 0)
        index_to_remove = index_to_remove[0]
        
        self.pixel_hit = np.delete(self.pixel_hit, index_to_remove, axis = 0)
        self.total_hits = np.delete(self.total_hits, index_to_remove, axis = 0)
   
        # keep pixels with average hits larger than min_fraction
        sum_hits_x = self.pixel_hit.sum(axis=(2,3))
        mean_x = np.mean(sum_hits_x, axis=0)
        index_to_keep = np.where(mean_x > min_fraction)[0]
        self.pixel_hit = self.pixel_hit[:,index_to_keep,:,:]

        sum_hits_y = self.pixel_hit.sum(axis=(1,3))
        mean_y = np.mean(sum_hits_y, axis=0)
        index_to_keep = np.where(mean_y > min_fraction)[0]
        self.pixel_hit = self.pixel_hit[:,:,index_to_keep,:]

        # Only keep #events = sample_size
        if sample_size:
            self.pixel_hit = self.pixel_hit[:sample_size]

        print('Initializing ACTS 1D dataset ...')
        print('Shape of the dataset is: ', self.pixel_hit.shape)

    
    def __len__(self) -> int:
        return self.pixel_hit.shape[0]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.pixel_hit[index]



def show_sum_of_hits(data, axis=(1,2,3)):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(data.sum(axis=axis), density=True)
    ax.set_xlabel('total hits')
    ax.set_ylabel('density')


def compare_sum_of_hits(gen_data, target_data):
    fig = plt.figure()
    ax = fig.add_subplot()

    for data, label in zip([gen_data, target_data], ['Generated data', 'Real data']):
        ax.hist(data.sum(axis=(1,2,3)), bins=range(0,55), alpha=0.5, density=True, label = label)
    ax.set_xlabel('total hits')
    ax.set_ylabel('density')
    
    plt.legend(loc="upper right")
    plt.show()

    return fig


def cal_mean_std(data: np.array):
    '''Calculate the mean and standard variation of the input data with shape of (batch_size, image_size)'''    
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    
    return mean, std

def show_mean_std(data, axis=None, xlabel ='', log=False):
    data = data.sum(axis=axis)

    mean, std = cal_mean_std(data)
    x = np.indices(mean.shape).squeeze()+1
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(x, mean, yerr = std, fmt = 'o')
    ax.set_ylabel('number of hits')
    ax.set_xlabel(xlabel)

    if log:
        plt.yscale('log')
        ax.set_ylim(0.0001, (mean.max()+std.max())*1.4)
    else:
        ax.set_ylim(0, (mean.max()+std.max())*1.4)

    plt.grid(True)
    plt.show()

    return fig

def compare_mean_std(gen_data, target_data, axis=None, xlabel='', log=False):
    fig = plt.figure()
    ax = fig.add_subplot()

    mean_max = 0
    std_max = 0
            
    for data, label, shift in zip([gen_data, target_data], ['Generated data', 'Real data'], [-0.1, 0.1]):
        data = data.sum(axis=axis)
        mean, std = cal_mean_std(data)
        x = np.indices(mean.shape).squeeze()+1
        ax.errorbar(x+shift, mean, yerr=std, fmt='o', label=label)
    
        if mean.max() > mean_max:
            mean_max = mean.max()
        if std.max() > std_max:
            std_max = std.max()

    ax.set_ylabel('number of hits')
    ax.set_xlabel(xlabel)

    if log:
        plt.yscale('log')
        ax.set_ylim(0.0001, (mean_max+std_max)*1.4)
    else:
        ax.set_ylim(0, (mean_max+std_max)*1.4)

    plt.legend(loc='upper right')
    plt.show()
    return fig

def compare_xy_mean_std(gen_data, target_data, axis=None, xlabel='', log=False):
    fig = plt.figure()
    ax = fig.add_subplot()

    mean_max = 0
    std_max = 0
            
    for data, label, shift in zip([gen_data, target_data], ['Generated data', 'Real data'], [-0.2, 0.2]):
        data = data.sum(axis=axis)
        mean, std = cal_mean_std(data)
        x = np.indices(mean.shape).squeeze()+1
        # ax.errorbar(x+shift, mean, yerr=std, fmt='o', label=label)
        ax.bar(x+shift, mean, width=0.4, yerr=std, capsize=5, label=label)
        # ax.hist(mean, label = label)
        # ax.bar(x+shift, mean)
    
        if mean.max() > mean_max:
            mean_max = mean.max()
        if std.max() > std_max:
            std_max = std.max()

    ax.set_ylabel('number of hits')
    ax.set_xlabel(xlabel)

    if log:
        plt.yscale('log')
        ax.set_ylim(0.0001, (mean_max+std_max)*1.4)
    else:
        ax.set_ylim(0, (mean_max+std_max)*1.4)

    plt.legend(loc='upper right')
    plt.show()
    return fig
    
def show_image_3d(image):
    image = image.squeeze()
    image = image.transpose(0, 2, 1)
    index = np.indices(image.shape)
    value = image.flatten()

    size = value.copy()
    size[size < 1e-3] = 0
    size[size >= 1e-3] = 1
    
    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')

    color_map = plt.get_cmap('jet')

    print(value.min())
    scatter_plot = ax3D.scatter(index[0], index[1], index[2], 
                                s = size*3, c = value, 
                                cmap = color_map, norm=colors.LogNorm(0.01, 0.35) )                                                                          
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Z')
    ax3D.set_zlabel('Y')
    ax3D.set_xlim(0, image.shape[0])
    ax3D.set_ylim(0, image.shape[1])
    ax3D.set_zlim(0, image.shape[2])

#    plt.colorbar(scatter_plot, shrink = 0.8, 
#                 label = 'Energy [GeV]', anchor=(0.5, 0.5))
    plt.show()

    return fig

# %%
if __name__ == '__main__':
    batch_size = 512
    # Load the dataset
    input_fn = '../data/ACTS/*.npy'
    data = ACTS(input_fn)

    show_sum_of_hits(data.pixel_hit)
    compare_sum_of_hits(data.pixel_hit, data.pixel_hit*1.1)
    show_mean_std(data.pixel_hit, axis=(1,2), xlabel='chip', log=False)
    compare_mean_std(data.pixel_hit, data.pixel_hit, axis=(1,2), xlabel='chip')
    show_mean_std(data.pixel_hit, axis=(1,3), xlabel='y', log=True)
    show_mean_std(data.pixel_hit, axis=(2,3), xlabel='x', log=True)
    show_image_3d(data.pixel_hit[0])


