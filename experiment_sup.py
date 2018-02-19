import click


@click.command()
@click.option('--exp', type=click.Choice(['svhn_mnist', 'mnist_svhn',
                                          'svhn_mnist_rgb', 'mnist_svhn_rgb',
                                          'cifar_stl', 'stl_cifar',
                                          'mnist_usps', 'usps_mnist',
                                          'syndigits_svhn', 'svhn_syndigits',
                                          'synsigns_gtsrb', 'gtsrb_synsigns'
                                          ]), default='svhn_mnist',
              help='experiment to run')
@click.option('--arch', type=click.Choice([
    '',
    'mnist-bn-32-64-256',
    'grey-32-64-128-gp', 'grey-32-64-128-gp-wn', 'grey-32-64-128-gp-nonorm',
    'rgb-128-256-down-gp', 'resnet18-32',
    'rgb40-48-96-192-384-gp', 'rgb40-96-192-384-gp',
]), default='', help='network architecture')
@click.option('--learning_rate', type=float, default=0.001, help='learning rate (Adam)')
@click.option('--standardise_samples', default=False, is_flag=True, help='standardise samples (0 mean unit var)')
@click.option('--affine_std', type=float, default=0.1, help='aug xform: random affine transform std-dev')
@click.option('--xlat_range', type=float, default=2.0, help='aug xform: translation range')
@click.option('--hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--intens_flip', is_flag=True, default=False, help='aug colour; intensity flip')
@click.option('--intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--intens_offset_range', type=str, default='',
              help='aug colour; intensity offset range `low:high` (-0.5:0.5 for mnist-svhn)')
@click.option('--gaussian_noise_std', type=float, default=0.1,
              help='aug: standard deviation of Gaussian noise to add to samples')
@click.option('--num_epochs', type=int, default=200, help='number of epochs')
@click.option('--batch_size', type=int, default=64, help='mini-batch size')
@click.option('--seed', type=int, default=0, help='random seed (0 for time-based)')
@click.option('--log_file', type=str, default='', help='log file path (none to disable)')
@click.option('--device', type=int, default=0, help='Device')
def experiment(exp, arch, learning_rate,
               standardise_samples, affine_std, xlat_range, hflip,
               intens_flip, intens_scale_range, intens_offset_range, gaussian_noise_std,
               num_epochs, batch_size, seed,
               log_file, device):
    import os
    import sys
    import cmdline_helpers

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None

    if log_file is not None:
        if os.path.exists(log_file):
            print('Output log file {} already exists'.format(log_file))
            return

    intens_scale_range_lower, intens_scale_range_upper, intens_offset_range_lower, intens_offset_range_upper = \
        cmdline_helpers.intens_aug_options(intens_scale_range, intens_offset_range)


    import time
    import math
    import numpy as np
    from batchup import data_source, work_pool
    import data_loaders
    import standardisation
    import network_architectures
    import augmentation
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F

    with torch.cuda.device(device):
        pool = work_pool.WorkerThreadPool(2)


        n_chn = 0


        if exp == 'svhn_mnist':
            d_source = data_loaders.load_svhn(zero_centre=False, greyscale=True)
            d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False)
        elif exp == 'mnist_svhn':
            d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True)
            d_target = data_loaders.load_svhn(zero_centre=False, greyscale=True, val=False)
        elif exp == 'svhn_mnist_rgb':
            d_source = data_loaders.load_svhn(zero_centre=False, greyscale=False)
            d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False, rgb=True)
        elif exp == 'mnist_svhn_rgb':
            d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, rgb=True)
            d_target = data_loaders.load_svhn(zero_centre=False, greyscale=False, val=False)
        elif exp == 'cifar_stl':
            d_source = data_loaders.load_cifar10(range_01=False)
            d_target = data_loaders.load_stl(zero_centre=False, val=False)
        elif exp == 'stl_cifar':
            d_source = data_loaders.load_stl(zero_centre=False)
            d_target = data_loaders.load_cifar10(range_01=False, val=False)
        elif exp == 'mnist_usps':
            d_source = data_loaders.load_mnist(zero_centre=False)
            d_target = data_loaders.load_usps(zero_centre=False, scale28=True, val=False)
        elif exp == 'usps_mnist':
            d_source = data_loaders.load_usps(zero_centre=False, scale28=True)
            d_target = data_loaders.load_mnist(zero_centre=False, val=False)
        elif exp == 'syndigits_svhn':
            d_source = data_loaders.load_syn_digits(zero_centre=False)
            d_target = data_loaders.load_svhn(zero_centre=False, val=False)
        elif exp == 'svhn_syndigits':
            d_source = data_loaders.load_svhn(zero_centre=False, val=False)
            d_target = data_loaders.load_syn_digits(zero_centre=False)
        elif exp == 'synsigns_gtsrb':
            d_source = data_loaders.load_syn_signs(zero_centre=False)
            d_target = data_loaders.load_gtsrb(zero_centre=False, val=False)
        elif exp == 'gtsrb_synsigns':
            d_source = data_loaders.load_gtsrb(zero_centre=False, val=False)
            d_target = data_loaders.load_syn_signs(zero_centre=False)
        else:
            print('Unknown experiment type \'{}\''.format(exp))
            return

        # Delete the training ground truths as we should not be using them
        del d_target.train_y

        if standardise_samples:
            standardisation.standardise_dataset(d_source)
            standardisation.standardise_dataset(d_target)

        n_classes = d_source.n_classes

        print('Loaded data')



        if arch == '':
            if exp in {'mnist_usps', 'usps_mnist'}:
                arch = 'mnist-bn-32-64-256'
            if exp in {'svhn_mnist', 'mnist_svhn'}:
                arch = 'grey-32-64-128-gp'
            if exp in {'cifar_stl', 'stl_cifar', 'syndigits_svhn', 'svhn_syndigits', 'svhn_mnist_rgb', 'mnist_svhn_rgb'}:
                arch = 'rgb-48-96-192-gp'
            if exp in {'synsigns_gtsrb', 'gtsrb_synsigns'}:
                arch = 'rgb40-48-96-192-384-gp'


        net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)

        if expected_shape != d_source.train_X.shape[1:]:
            print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
                  'data has samples of shape {}'.format(arch, exp, expected_shape, d_source.train_X.shape[1:]))
            return


        net = net_class(n_classes).cuda()
        params = list(net.parameters())

        optimizer = torch.optim.Adam(params, lr=learning_rate)
        classification_criterion = nn.CrossEntropyLoss()

        print('Built network')

        aug = augmentation.ImageAugmentation(
            hflip, xlat_range, affine_std,
            intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
            intens_offset_range_lower=intens_offset_range_lower, intens_offset_range_upper=intens_offset_range_upper,
            intens_flip=intens_flip, gaussian_noise_std=gaussian_noise_std)

        def augment(X_sup, y_sup):
            X_sup = aug.augment(X_sup)
            return [X_sup, y_sup]



        def f_train(X_sup, y_sup):
            X_sup = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            y_sup = torch.autograd.Variable(torch.from_numpy(y_sup).long().cuda())

            optimizer.zero_grad()
            net.train(mode=True)

            sup_logits_out = net(X_sup)

            # Supervised classification loss
            clf_loss = classification_criterion(sup_logits_out, y_sup)

            loss_expr = clf_loss

            loss_expr.backward()
            optimizer.step()

            n_samples = X_sup.size()[0]

            return float(clf_loss.data.cpu().numpy()) * n_samples


        print('Compiled training function')

        def f_pred_src(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            net.train(mode=False)
            return F.softmax(net(X_var)).data.cpu().numpy()

        def f_pred_tgt(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            net.train(mode=False)
            return F.softmax(net(X_var)).data.cpu().numpy()

        def f_eval_src(X_sup, y_sup):
            y_pred_prob = f_pred_src(X_sup)
            y_pred = np.argmax(y_pred_prob, axis=1)
            return float((y_pred != y_sup).sum())

        def f_eval_tgt(X_sup, y_sup):
            y_pred_prob = f_pred_tgt(X_sup)
            y_pred = np.argmax(y_pred_prob, axis=1)
            return float((y_pred != y_sup).sum())

        print('Compiled evaluation function')


        # Setup output
        def log(text):
            print(text)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(text + '\n')
                    f.flush()
                    f.close()
        cmdline_helpers.ensure_containing_dir_exists(log_file)

        # Report setttings
        log('sys.argv={}'.format(sys.argv))

        # Report dataset size
        log('Dataset:')
        log('SOURCE Train: X.shape={}, y.shape={}'.format(d_source.train_X.shape, d_source.train_y.shape))
        log('SOURCE Test: X.shape={}, y.shape={}'.format(d_source.test_X.shape, d_source.test_y.shape))
        log('TARGET Train: X.shape={}'.format(d_target.train_X.shape))
        log('TARGET Test: X.shape={}, y.shape={}'.format(d_target.test_X.shape, d_target.test_y.shape))

        print('Training...')
        train_ds = data_source.ArrayDataSource([d_source.train_X, d_source.train_y]).map(augment)

        source_test_ds = data_source.ArrayDataSource([d_source.test_X, d_source.test_y])
        target_test_ds = data_source.ArrayDataSource([d_target.test_X, d_target.test_y])

        if seed != 0:
            shuffle_rng = np.random.RandomState(seed)
        else:
            shuffle_rng = np.random

        best_src_test_err = 1.0
        for epoch in range(num_epochs):
            t1 = time.time()

            train_res = train_ds.batch_map_mean(
                f_train, batch_size=batch_size, shuffle=shuffle_rng)

            train_clf_loss = train_res[0]
            src_test_err, = source_test_ds.batch_map_mean(f_eval_src, batch_size=batch_size * 4)
            tgt_test_err, = target_test_ds.batch_map_mean(f_eval_tgt, batch_size=batch_size * 4)

            t2 = time.time()

            if src_test_err < best_src_test_err:
                log('*** Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
                    'SRC TEST ERR={:.3%}, TGT TEST err={:.3%}'.format(
                    epoch, t2 - t1, train_clf_loss, src_test_err, tgt_test_err))
                best_src_test_err = src_test_err
            else:
                log('Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
                    'SRC TEST ERR={:.3%}, TGT TEST err={:.3%}'.format(
                    epoch, t2 - t1, train_clf_loss, src_test_err, tgt_test_err))

if __name__ == '__main__':
    experiment()