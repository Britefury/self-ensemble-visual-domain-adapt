import numpy as np

from skimage.transform import downscale_local_mean, resize

from batchup.datasets import mnist, fashion_mnist, cifar10, svhn, stl, usps
import domain_datasets


def rgb2grey_tensor(X):
    return (X[:, 0:1, :, :] * 0.2125) + (X[:, 1:2, :, :] * 0.7154) + (X[:, 2:3, :, :] * 0.0721)

# Dataset loading functions
def load_svhn(zero_centre=False, greyscale=False, val=False, extra=False):
    #
    #
    # Load SVHN
    #
    #

    print('Loading SVHN...')
    if val:
        d_svhn = svhn.SVHN(n_val=10000)
    else:
        d_svhn = svhn.SVHN(n_val=0)

    if extra:
        d_extra = svhn.SVHNExtra()
    else:
        d_extra = None

    d_svhn.train_X = d_svhn.train_X[:]
    d_svhn.val_X = d_svhn.val_X[:]
    d_svhn.test_X = d_svhn.test_X[:]
    d_svhn.train_y = d_svhn.train_y[:]
    d_svhn.val_y = d_svhn.val_y[:]
    d_svhn.test_y = d_svhn.test_y[:]

    if extra:
        d_svhn.train_X = np.append(d_svhn.train_X, d_extra.X[:], axis=0)
        d_svhn.train_y = np.append(d_svhn.train_y, d_extra.y[:], axis=0)

    if greyscale:
        d_svhn.train_X = rgb2grey_tensor(d_svhn.train_X)
        d_svhn.val_X = rgb2grey_tensor(d_svhn.val_X)
        d_svhn.test_X = rgb2grey_tensor(d_svhn.test_X)

    if zero_centre:
        d_svhn.train_X = d_svhn.train_X * 2.0 - 1.0
        d_svhn.val_X = d_svhn.val_X * 2.0 - 1.0
        d_svhn.test_X = d_svhn.test_X * 2.0 - 1.0

    print('SVHN: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_svhn.train_X.shape, d_svhn.train_y.shape, d_svhn.val_X.shape, d_svhn.val_y.shape, d_svhn.test_X.shape,
        d_svhn.test_y.shape))

    print('SVHN: train: X.min={}, X.max={}'.format(
        d_svhn.train_X.min(), d_svhn.train_X.max()))

    d_svhn.n_classes = 10

    return d_svhn


def load_mnist(invert=False, zero_centre=False, intensity_scale=1.0, val=False, pad32=False, downscale_x=1,
               rgb=False):
    #
    #
    # Load MNIST
    #
    #

    print('Loading MNIST...')

    if val:
        d_mnist = mnist.MNIST(n_val=10000)
    else:
        d_mnist = mnist.MNIST(n_val=0)

    d_mnist.train_X = d_mnist.train_X[:]
    d_mnist.val_X = d_mnist.val_X[:]
    d_mnist.test_X = d_mnist.test_X[:]
    d_mnist.train_y = d_mnist.train_y[:]
    d_mnist.val_y = d_mnist.val_y[:]
    d_mnist.test_y = d_mnist.test_y[:]

    if downscale_x != 1:
        d_mnist.train_X = downscale_local_mean(d_mnist.train_X, (1, 1, 1, downscale_x))
        d_mnist.val_X = downscale_local_mean(d_mnist.val_X, (1, 1, 1, downscale_x))
        d_mnist.test_X = downscale_local_mean(d_mnist.test_X, (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_mnist.train_X.shape[2]) // 2
        px = (32 - d_mnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_mnist.train_X = np.pad(d_mnist.train_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')
        d_mnist.val_X = np.pad(d_mnist.val_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')
        d_mnist.test_X = np.pad(d_mnist.test_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')

    if invert:
        # Invert
        d_mnist.train_X = 1.0 - d_mnist.train_X
        d_mnist.val_X = 1.0 - d_mnist.val_X
        d_mnist.test_X = 1.0 - d_mnist.test_X

    if intensity_scale != 1.0:
        d_mnist.train_X = (d_mnist.train_X - 0.5) * intensity_scale + 0.5
        d_mnist.val_X = (d_mnist.val_X - 0.5) * intensity_scale + 0.5
        d_mnist.test_X = (d_mnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_mnist.train_X = d_mnist.train_X * 2.0 - 1.0
        d_mnist.test_X = d_mnist.test_X * 2.0 - 1.0

    if rgb:
        d_mnist.train_X = np.concatenate([d_mnist.train_X] * 3, axis=1)
        d_mnist.val_X = np.concatenate([d_mnist.val_X] * 3, axis=1)
        d_mnist.test_X = np.concatenate([d_mnist.test_X] * 3, axis=1)

    print('MNIST: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_mnist.train_X.shape, d_mnist.train_y.shape,
        d_mnist.val_X.shape, d_mnist.val_y.shape,
        d_mnist.test_X.shape, d_mnist.test_y.shape))

    print('MNIST: train: X.min={}, X.max={}'.format(
        d_mnist.train_X.min(), d_mnist.train_X.max()))

    d_mnist.n_classes = 10

    return d_mnist


def load_fashion_mnist(invert=False, zero_centre=False, intensity_scale=1.0, val=False, pad32=False, downscale_x=1):
    #
    #
    # Load MNIST
    #
    #

    print('Loading Fashion MNIST...')

    if val:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=10000)
    else:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=0)

    d_fmnist.train_X = d_fmnist.train_X[:]
    d_fmnist.val_X = d_fmnist.val_X[:]
    d_fmnist.test_X = d_fmnist.test_X[:]
    d_fmnist.train_y = d_fmnist.train_y[:]
    d_fmnist.val_y = d_fmnist.val_y[:]
    d_fmnist.test_y = d_fmnist.test_y[:]

    if downscale_x != 1:
        d_fmnist.train_X = downscale_local_mean(d_fmnist.train_X, (1, 1, 1, downscale_x))
        d_fmnist.val_X = downscale_local_mean(d_fmnist.val_X, (1, 1, 1, downscale_x))
        d_fmnist.test_X = downscale_local_mean(d_fmnist.test_X, (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_fmnist.train_X.shape[2]) // 2
        px = (32 - d_fmnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_fmnist.train_X = np.pad(d_fmnist.train_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')
        d_fmnist.val_X = np.pad(d_fmnist.val_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')
        d_fmnist.test_X = np.pad(d_fmnist.test_X, [(0 ,0), (0 ,0), (py ,py), (px ,px)], mode='constant')

    if invert:
        # Invert
        d_fmnist.train_X = 1.0 - d_fmnist.train_X
        d_fmnist.val_X = 1.0 - d_fmnist.val_X
        d_fmnist.test_X = 1.0 - d_fmnist.test_X

    if intensity_scale != 1.0:
        d_fmnist.train_X = (d_fmnist.train_X - 0.5) * intensity_scale + 0.5
        d_fmnist.val_X = (d_fmnist.val_X - 0.5) * intensity_scale + 0.5
        d_fmnist.test_X = (d_fmnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_fmnist.train_X = d_fmnist.train_X * 2.0 - 1.0
        d_fmnist.test_X = d_fmnist.test_X * 2.0 - 1.0

    print('Fashion MNIST: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
          'test: X.shape={}, y.shape={}'.format(
        d_fmnist.train_X.shape, d_fmnist.train_y.shape,
        d_fmnist.val_X.shape, d_fmnist.val_y.shape,
        d_fmnist.test_X.shape, d_fmnist.test_y.shape))

    print('Fashion MNIST: train: X.min={}, X.max={}'.format(
        d_fmnist.train_X.min(), d_fmnist.train_X.max()))

    d_fmnist.n_classes = 10

    return d_fmnist


def load_usps(invert=False, zero_centre=False, val=False, scale28=False):
    #
    #
    # Load USPS
    #
    #

    print('Loading USPS...')

    if val:
        d_usps = usps.USPS()
    else:
        d_usps = usps.USPS(n_val=None)

    d_usps.train_X = d_usps.train_X[:]
    d_usps.val_X = d_usps.val_X[:]
    d_usps.test_X = d_usps.test_X[:]
    d_usps.train_y = d_usps.train_y[:]
    d_usps.val_y = d_usps.val_y[:]
    d_usps.test_y = d_usps.test_y[:]

    if scale28:
        def _resize_tensor(X):
            X_prime = np.zeros((X.shape[0], 1, 28, 28), dtype=np.float32)
            for i in range(X.shape[0]):
                X_prime[i, 0, :, :] = resize(X[i, 0, :, :], (28, 28), mode='constant')
            return X_prime
        # Scale 16x16 to 28x28
        d_usps.train_X = _resize_tensor(d_usps.train_X)
        d_usps.val_X = _resize_tensor(d_usps.val_X)
        d_usps.test_X = _resize_tensor(d_usps.test_X)

    if invert:
        # Invert
        d_usps.train_X = 1.0 - d_usps.train_X
        d_usps.val_X = 1.0 - d_usps.val_X
        d_usps.test_X = 1.0 - d_usps.test_X

    if zero_centre:
        d_usps.train_X = d_usps.train_X * 2.0 - 1.0
        d_usps.test_X = d_usps.test_X * 2.0 - 1.0

    print('USPS: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_usps.train_X.shape, d_usps.train_y.shape,
        d_usps.val_X.shape, d_usps.val_y.shape,
        d_usps.test_X.shape, d_usps.test_y.shape))

    print('USPS: train: X.min={}, X.max={}'.format(
        d_usps.train_X.min(), d_usps.train_X.max()))

    d_usps.n_classes = 10

    return d_usps


def load_cifar10(range_01=False, val=False):
    #
    #
    # Load CIFAR-10 for adaptation with STL
    #
    #

    print('Loading CIFAR-10...')
    if val:
        d_cifar = cifar10.CIFAR10(n_val=5000)
    else:
        d_cifar = cifar10.CIFAR10(n_val=0)

    d_cifar.train_X = d_cifar.train_X[:]
    d_cifar.val_X = d_cifar.val_X[:]
    d_cifar.test_X = d_cifar.test_X[:]
    d_cifar.train_y = d_cifar.train_y[:]
    d_cifar.val_y = d_cifar.val_y[:]
    d_cifar.test_y = d_cifar.test_y[:]

    # Remap class indices so that the frog class (6) has an index of -1 as it does not appear int the STL dataset
    cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
    d_cifar.train_y = cls_mapping[d_cifar.train_y]
    d_cifar.val_y = cls_mapping[d_cifar.val_y]
    d_cifar.test_y = cls_mapping[d_cifar.test_y]

    # Remove all samples from skipped classes
    train_mask = d_cifar.train_y != -1
    val_mask = d_cifar.val_y != -1
    test_mask = d_cifar.test_y != -1

    d_cifar.train_X = d_cifar.train_X[train_mask]
    d_cifar.train_y = d_cifar.train_y[train_mask]
    d_cifar.val_X = d_cifar.val_X[val_mask]
    d_cifar.val_y = d_cifar.val_y[val_mask]
    d_cifar.test_X = d_cifar.test_X[test_mask]
    d_cifar.test_y = d_cifar.test_y[test_mask]

    if range_01:
        d_cifar.train_X = d_cifar.train_X * 2.0 - 1.0
        d_cifar.val_X = d_cifar.val_X * 2.0 - 1.0
        d_cifar.test_X = d_cifar.test_X * 2.0 - 1.0

    print('CIFAR-10: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_cifar.train_X.shape, d_cifar.train_y.shape, d_cifar.val_X.shape, d_cifar.val_y.shape, d_cifar.test_X.shape,
        d_cifar.test_y.shape))

    print('CIFAR-10: train: X.min={}, X.max={}'.format(
        d_cifar.train_X.min(), d_cifar.train_X.max()))

    d_cifar.n_classes = 9

    return d_cifar


def load_stl(zero_centre=False, val=False):
    #
    #
    # Load STL for adaptation with CIFAR-10
    #
    #

    print('Loading STL...')
    if val:
        d_stl = stl.STL()
    else:
        d_stl = stl.STL(n_val_folds=0)

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]
    d_stl.train_y = d_stl.train_y[:]
    d_stl.val_y = d_stl.val_y[:]
    d_stl.test_y = d_stl.test_y[:]

    # Remap class indices to match CIFAR-10:
    cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
    d_stl.train_y = cls_mapping[d_stl.train_y]
    d_stl.val_y = cls_mapping[d_stl.val_y]
    d_stl.test_y = cls_mapping[d_stl.test_y]

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]

    # Remove all samples from class -1 (monkey) as it does not appear int the CIFAR-10 dataset
    train_mask = d_stl.train_y != -1
    val_mask = d_stl.val_y != -1
    test_mask = d_stl.test_y != -1

    d_stl.train_X = d_stl.train_X[train_mask]
    d_stl.train_y = d_stl.train_y[train_mask]
    d_stl.val_X = d_stl.val_X[val_mask]
    d_stl.val_y = d_stl.val_y[val_mask]
    d_stl.test_X = d_stl.test_X[test_mask]
    d_stl.test_y = d_stl.test_y[test_mask]

    # Downsample images from 96x96 to 32x32
    d_stl.train_X = downscale_local_mean(d_stl.train_X, (1, 1, 3, 3))
    d_stl.val_X = downscale_local_mean(d_stl.val_X, (1, 1, 3, 3))
    d_stl.test_X = downscale_local_mean(d_stl.test_X, (1, 1, 3, 3))

    if zero_centre:
        d_stl.train_X = d_stl.train_X * 2.0 - 1.0
        d_stl.val_X = d_stl.val_X * 2.0 - 1.0
        d_stl.test_X = d_stl.test_X * 2.0 - 1.0

    print('STL: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_stl.train_X.shape, d_stl.train_y.shape, d_stl.val_X.shape, d_stl.val_y.shape, d_stl.test_X.shape,
        d_stl.test_y.shape))

    print('STL: train: X.min={}, X.max={}'.format(
        d_stl.train_X.min(), d_stl.train_X.max()))

    d_stl.n_classes = 9

    return d_stl


def load_syn_digits(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading Syn-digits...')
    if val:
        d_synd = domain_datasets.SynDigits(n_val=10000)
    else:
        d_synd = domain_datasets.SynDigits(n_val=0)

    d_synd.train_X = d_synd.train_X[:]
    d_synd.val_X = d_synd.val_X[:]
    d_synd.test_X = d_synd.test_X[:]
    d_synd.train_y = d_synd.train_y[:]
    d_synd.val_y = d_synd.val_y[:]
    d_synd.test_y = d_synd.test_y[:]

    if greyscale:
        d_synd.train_X = rgb2grey_tensor(d_synd.train_X)
        d_synd.val_X = rgb2grey_tensor(d_synd.val_X)
        d_synd.test_X = rgb2grey_tensor(d_synd.test_X)

    if zero_centre:
        d_synd.train_X = d_synd.train_X * 2.0 - 1.0
        d_synd.val_X = d_synd.val_X * 2.0 - 1.0
        d_synd.test_X = d_synd.test_X * 2.0 - 1.0

    print('SynDigits: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_synd.train_X.shape, d_synd.train_y.shape, d_synd.val_X.shape, d_synd.val_y.shape, d_synd.test_X.shape,
        d_synd.test_y.shape))

    print('SynDigits: train: X.min={}, X.max={}'.format(
        d_synd.train_X.min(), d_synd.train_X.max()))

    d_synd.n_classes = 10

    return d_synd


def load_syn_signs(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading Syn-signs...')
    if val:
        d_syns = domain_datasets.SynSigns(n_val=10000, n_test=10000)
    else:
        d_syns = domain_datasets.SynSigns(n_val=0, n_test=10000)

    d_syns.train_X = d_syns.train_X[:]
    d_syns.val_X = d_syns.val_X[:]
    d_syns.test_X = d_syns.test_X[:]
    d_syns.train_y = d_syns.train_y[:]
    d_syns.val_y = d_syns.val_y[:]
    d_syns.test_y = d_syns.test_y[:]

    if greyscale:
        d_syns.train_X = rgb2grey_tensor(d_syns.train_X)
        d_syns.val_X = rgb2grey_tensor(d_syns.val_X)
        d_syns.test_X = rgb2grey_tensor(d_syns.test_X)

    if zero_centre:
        d_syns.train_X = d_syns.train_X * 2.0 - 1.0
        d_syns.val_X = d_syns.val_X * 2.0 - 1.0
        d_syns.test_X = d_syns.test_X * 2.0 - 1.0

    print('SynSigns: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
          'test: X.shape={}, y.shape={}'.format(
        d_syns.train_X.shape, d_syns.train_y.shape, d_syns.val_X.shape, d_syns.val_y.shape, d_syns.test_X.shape,
        d_syns.test_y.shape))

    print('SynSigns: train: X.min={}, X.max={}'.format(
        d_syns.train_X.min(), d_syns.train_X.max()))

    d_syns.n_classes = 43

    return d_syns


def load_gtsrb(zero_centre=False, greyscale=False, val=False):
    #
    #
    # Load syn digits
    #
    #

    print('Loading GTSRB...')
    if val:
        d_gts = domain_datasets.GTSRB(n_val=10000)
    else:
        d_gts = domain_datasets.GTSRB(n_val=0)

    d_gts.train_X = d_gts.train_X[:]
    d_gts.val_X = d_gts.val_X[:]
    d_gts.test_X = d_gts.test_X[:]
    d_gts.train_y = d_gts.train_y[:]
    d_gts.val_y = d_gts.val_y[:]
    d_gts.test_y = d_gts.test_y[:]

    if greyscale:
        d_gts.train_X = rgb2grey_tensor(d_gts.train_X)
        d_gts.val_X = rgb2grey_tensor(d_gts.val_X)
        d_gts.test_X = rgb2grey_tensor(d_gts.test_X)

    if zero_centre:
        d_gts.train_X = d_gts.train_X * 2.0 - 1.0
        d_gts.val_X = d_gts.val_X * 2.0 - 1.0
        d_gts.test_X = d_gts.test_X * 2.0 - 1.0

    print('GTSRB: train: X.shape={}, y.shape={}, val: X.shape={}, y.shape={}, '
          'test: X.shape={}, y.shape={}'.format(
        d_gts.train_X.shape, d_gts.train_y.shape, d_gts.val_X.shape, d_gts.val_y.shape, d_gts.test_X.shape,
        d_gts.test_y.shape))

    print('GTSRB: train: X.min={}, X.max={}'.format(
        d_gts.train_X.min(), d_gts.train_X.max()))

    d_gts.n_classes = 43

    return d_gts

