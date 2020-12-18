
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
import torchvision
import numpy
import scipy
import pickle

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path2real', type=str, help=('Path to the real images'))
parser.add_argument('--path2fake', type=str, help=('Path to generated images'))


def get_activations(files, model, batch_size=1, dims=64):
    model.eval()
    if len(files) % batch_size != 0:
        if batch_size > len(files):
            batch_size = len(files)
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
        images = images[:,:,:,0:3]
        images = images.transpose((0, 3, 1, 2))
        images /= 255
        batch = torch.from_numpy(images).type(torch.FloatTensor).cuda()
        pred = model(batch)[0]
        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=64):

    act = get_activations(files, model, batch_size, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def _compute_statistics_of_path(files, model, batch_size, dims):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg'))+ list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims)

    return m, s


def calculate_sifid_given_paths(path1, path2, batch_size, dims):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.cuda()
    path1 = pathlib.Path(path1)
    files1 = list(path1.glob('*.%s' %'jpg'))
    path2 = pathlib.Path(path2)
    files2 = list(path2.glob('*.%s' %'jpg'))

    fid_values = []
    for i in range(len(files2)):
        m1, s1 = calculate_activation_statistics([files1[i]], model, batch_size, dims)
        m2, s2 = calculate_activation_statistics([files2[i]], model, batch_size, dims)
        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
        file_num1 = files1[i].name
        file_num2 = files2[i].name

    return fid_values


if __name__ == '__main__':
    args = parser.parse_args()
    path1 = args.path2real
    path2 = args.path2fake
    sifid_values = calculate_sifid_given_paths(path1,path2,1,64)
    sifid_values = np.asarray(sifid_values,dtype=np.float32)
    print('SIFID: ', sifid_values.mean())
