import click


@click.command()
@click.argument('plot_path', type=str)
@click.argument('ds_name', type=click.Choice(['mnist', 'usps', 'svhn', 'svhn_grey',
                                          'cifar', 'stl', 'syndigits', 'synsigns', 'gtsrb'
                                          ]))
@click.option('--no_aug', is_flag=True, default=False)
@click.option('--affine_std', type=float, default=0.1, help='aug xform: random affine transform std-dev')
@click.option('--scale_u_range', type=str, default='',
              help='aug xform: scale uniform range; lower:upper')
@click.option('--scale_x_range', type=str, default='',
              help='aug xform: scale x range; lower:upper')
@click.option('--scale_y_range', type=str, default='',
              help='aug xform: scale y range; lower:upper')
@click.option('--xlat_range', type=float, default=2.0, help='aug xform: translation range')
@click.option('--hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--intens_flip', is_flag=True, default=False,
              help='aug colour; enable intensity flip')
@click.option('--intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--intens_offset_range', type=str, default='',
              help='aug colour; intensity offset range `low:high` (-0.5:0.5 for mnist-svhn)')
@click.option('--grid_h', type=int, default=1, help='sample grid height')
@click.option('--grid_w', type=int, default=5, help='sample grid width')
@click.option('--seed', type=int, default=12345, help='random seed (0 for time-based)')
def experiment(plot_path, ds_name, no_aug, affine_std, scale_u_range, scale_x_range, scale_y_range, xlat_range, hflip,
               intens_flip, intens_scale_range, intens_offset_range,
               grid_h, grid_w, seed):
    settings = locals().copy()

    import os
    import sys
    import cmdline_helpers

    intens_scale_range_lower, intens_scale_range_upper = cmdline_helpers.colon_separated_range(intens_scale_range)
    intens_offset_range_lower, intens_offset_range_upper = cmdline_helpers.colon_separated_range(intens_offset_range)
    scale_u_range = cmdline_helpers.colon_separated_range(scale_u_range)
    scale_x_range = cmdline_helpers.colon_separated_range(scale_x_range)
    scale_y_range = cmdline_helpers.colon_separated_range(scale_y_range)


    import numpy as np
    from skimage.util.montage import montage2d
    from PIL import Image
    from batchup import data_source
    import data_loaders
    import augmentation

    n_chn = 0


    if ds_name == 'mnist':
        d_source = data_loaders.load_mnist(zero_centre=False)
    elif ds_name == 'usps':
        d_source = data_loaders.load_usps(zero_centre=False, scale28=True)
    elif ds_name == 'svhn_grey':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=True)
    elif ds_name == 'svhn':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=False)
    elif ds_name == 'cifar':
        d_source = data_loaders.load_cifar10()
    elif ds_name == 'stl':
        d_source = data_loaders.load_stl()
    elif ds_name == 'syndigits':
        d_source = data_loaders.load_syn_digits(zero_centre=False, greyscale=False)
    elif ds_name == 'synsigns':
        d_source = data_loaders.load_syn_signs(zero_centre=False, greyscale=False)
    elif ds_name == 'gtsrb':
        d_source = data_loaders.load_gtsrb(zero_centre=False, greyscale=False)
    else:
        print('Unknown dataset \'{}\''.format(ds_name))
        return

    # Delete the training ground truths as we should not be using them
    del d_source.train_y

    n_classes = d_source.n_classes

    print('Loaded data')



    src_aug = augmentation.ImageAugmentation(
        hflip, xlat_range, affine_std,
        intens_flip=intens_flip,
        intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
        intens_offset_range_lower=intens_offset_range_lower, intens_offset_range_upper=intens_offset_range_upper,
        scale_u_range=scale_u_range, scale_x_range=scale_x_range, scale_y_range=scale_y_range)

    def augment(X):
        if not no_aug:
            X = src_aug.augment(X)
        return X,


    rampup_weight_in_list = [0]

    print('Rendering...')
    train_ds = data_source.ArrayDataSource([d_source.train_X], repeats=-1).map(augment)
    n_samples = len(d_source.train_X)


    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random


    batch_size = grid_h * grid_w
    display_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

    best_src_test_err = 1.0

    x_batch, = next(display_batch_iter)

    montage = []
    for chn_i in range(x_batch.shape[1]):
        m = montage2d(x_batch[:, chn_i, :, :], grid_shape=(grid_h, grid_w))
        montage.append(m[:, :, None])
    montage = np.concatenate(montage, axis=2)

    if montage.shape[2] == 1:
        montage = montage[:, :, 0]

    lower = min(0.0, montage.min())
    upper = max(1.0, montage.max())
    montage = (montage - lower) / (upper - lower)
    montage = (np.clip(montage, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(montage).save(plot_path)



if __name__ == '__main__':
    experiment()