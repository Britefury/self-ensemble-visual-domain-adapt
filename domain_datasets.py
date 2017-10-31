import sys, os
if sys.version_info[0] == 2:
    from ConfigParser import RawConfigParser
else:
    from configparser import RawConfigParser

import numpy as np
import tables
from batchup.datasets import dataset, svhn
from scipy.io import loadmat
from batchup.image.utils import ImageArrayUInt8ToFloat32
from sklearn.model_selection import StratifiedShuffleSplit


_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        if os.path.exists('domain_datasets.cfg'):
            _CONFIG = RawConfigParser()
            _CONFIG.read('domain_datasets.cfg')
        else:
            raise ValueError('Could not find configuration file domain_datasets.cfg')
    return _CONFIG


def get_data_dir(name):
    config = get_config()
    path = config.get('paths', name)
    if path is not None and path != '':
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise ValueError('Configuration file entry for paths:{} does not exist'.format(name))
        return path
    else:
        raise ValueError('Configuration file did not have entry for paths:{}'.format(name))



def _syndigits_train_path():
    return os.path.join(get_data_dir('syn_digits'), 'synth_train_32x32.mat')

def _syndigits_test_path():
    return os.path.join(get_data_dir('syn_digits'), 'synth_test_32x32.mat')

def _syndigits_h5_path():
    return os.path.abspath(os.path.join(get_data_dir('syn_digits'), 'syn_digits.h5'))

_TRAIN_SRC = dataset.ExistingSourceFile(_syndigits_train_path, None)
_TEST_SRC = dataset.ExistingSourceFile(_syndigits_test_path, None)


@dataset.fetch_and_convert_dataset(
        [_TRAIN_SRC, _TEST_SRC], _syndigits_h5_path)
def fetch_syn_digits(source_paths, target_path):
    train_path, test_path = source_paths

    f_out = tables.open_file(target_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'syn_digits', 'Syn-Digits data')

    # Load in the training data Matlab file
    print('Converting {} to HDF5...'.format(train_path))
    train_X_u8, train_y = svhn._read_svhn_matlab(train_path)
    f_out.create_array(g_out, 'train_X_u8', train_X_u8)
    f_out.create_array(g_out, 'train_y', train_y)
    del train_X_u8
    del train_y

    # Load in the test data Matlab file
    print('Converting {} to HDF5...'.format(test_path))
    test_X_u8, test_y = svhn._read_svhn_matlab(test_path)
    f_out.create_array(g_out, 'test_X_u8', test_X_u8)
    f_out.create_array(g_out, 'test_y', test_y)
    del test_X_u8
    del test_y

    f_out.close()

    return target_path


def delete_cache():  # pragma: no cover
    dataset.delete_dataset_cache(_syndigits_h5_path())


class SynDigits (object):
    def __init__(self, n_val=10000, val_lower=0.0, val_upper=1.0):
        h5_path = fetch_syn_digits()
        if h5_path is not None:
            f = tables.open_file(h5_path, mode='r')

            train_X_u8 = f.root.syn_digits.train_X_u8
            train_y = f.root.syn_digits.train_y
            self.test_X_u8 = f.root.syn_digits.test_X_u8
            self.test_y = f.root.syn_digits.test_y

            if n_val == 0 or n_val is None:
                self.train_X_u8 = train_X_u8
                self.train_y = train_y
                self.val_X_u8 = np.zeros((0, 3, 32, 32), dtype=np.uint8)
                self.val_y = np.zeros((0,), dtype=np.int32)
            else:
                self.train_X_u8 = train_X_u8[:-n_val]
                self.train_y = train_y[:-n_val]
                self.val_X_u8 = train_X_u8[-n_val:]
                self.val_y = train_y[-n_val:]
        else:
            raise RuntimeError('Could not load Syn-Digits dataset')

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)



class GTSRB (object):
    def __init__(self, n_val=2935, shuffle_seed=12345, val_lower=0.0, val_upper=1.0):
        h5_path = os.path.join(get_data_dir('gtsrb'), 'gtsrb.h5')
        if not os.path.exists(h5_path):
            raise RuntimeError('Could not load GTSRB from {}; please run '
                               '\'prepare_gtsrb.py\' to create it'.format(h5_path))

        f = tables.open_file(h5_path, mode='r')

        train_X_u8 = f.root.gtsrb.train_X_u8
        train_y = f.root.gtsrb.train_y
        test_X_u8 = f.root.gtsrb.test_X_u8
        test_y = f.root.gtsrb.test_y

        shuffle_rng = np.random.RandomState(shuffle_seed)
        train_ndx = shuffle_rng.permutation(len(train_X_u8))
        test_ndx = shuffle_rng.permutation(len(test_X_u8))
        train_X_u8 = train_X_u8[:][train_ndx]
        train_y = train_y[:][train_ndx]
        test_X_u8 = test_X_u8[:][test_ndx]
        test_y = test_y[:][test_ndx]
        if n_val == 0 or n_val is None:
            self.train_X_u8, self.train_y = train_X_u8, train_y
            self.val_X_u8 = np.zeros((0, 3, 40, 40), dtype=np.uint8)
            self.val_y = np.zeros((0,), dtype=np.int32)
        else:
            self.train_X_u8, self.val_X_u8 = train_X_u8[:-n_val], train_X_u8[-n_val:]
            self.train_y, self.val_y = train_y[:-n_val], train_y[-n_val:]
        self.test_X_u8 = test_X_u8
        self.test_y = test_y

        self.n_classes = 43

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)



class SynSigns (object):
    def __init__(self, n_val=10000, n_test=10000, shuffle_seed=12345, val_lower=0.0, val_upper=1.0):
        h5_path = os.path.join(get_data_dir('syn_signs'), 'syn_signs.h5')
        if not os.path.exists(h5_path):
            raise RuntimeError('Could not load Syn-Signs from {}; please run '
                               '\'prepare_synsigns.py\' to create it'.format(h5_path))

        f = tables.open_file(h5_path, mode='r')

        X_u8 = f.root.syn_signs.X_u8
        y = f.root.syn_signs.y

        shuffle_rng = np.random.RandomState(shuffle_seed)
        ndx = shuffle_rng.permutation(len(X_u8))
        X_u8 = X_u8[:][ndx]
        y = y[:][ndx]
        n_vt = n_val + n_test
        self.train_X_u8 = X_u8[:-n_vt]
        self.train_y = y[:-n_vt]
        valtest_X_u8 = X_u8[-n_vt:]
        valtest_y = y[-n_vt:]
        self.val_X_u8 = valtest_X_u8[:n_val]
        self.val_y = valtest_y[:n_val]
        self.test_X_u8 = valtest_X_u8[n_val:]
        self.test_y = valtest_y[n_val:]

        self.n_classes = 43

        self.train_X = ImageArrayUInt8ToFloat32(self.train_X_u8, val_lower,
                                                val_upper)
        self.val_X = ImageArrayUInt8ToFloat32(self.val_X_u8, val_lower,
                                              val_upper)
        self.test_X = ImageArrayUInt8ToFloat32(self.test_X_u8, val_lower,
                                               val_upper)
