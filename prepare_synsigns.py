import click


@click.command()
def prepare():
    import os
    import sys
    import numpy as np
    import tables
    import tqdm
    import domain_datasets
    import cv2

    synsigns_path = domain_datasets.get_data_dir('syn_signs')
    data_path = os.path.join(synsigns_path, 'synthetic_data')

    labels_path = os.path.join(data_path, 'train_labelling.txt')

    if not os.path.exists(labels_path):
        print('Labels path {} does not exist'.format(labels_path))
        sys.exit(0)

    # Open the file that lists the image files along with their ground truth class
    lines = [line.strip() for line in open(labels_path, 'r').readlines()]
    lines = [line for line in lines if line != '']

    output_path = os.path.join(synsigns_path, 'syn_signs.h5')
    print('Creating {}...'.format(output_path))
    f_out = tables.open_file(output_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'syn_signs', 'Syn-Signs data')
    filters = tables.Filters(complevel=9, complib='blosc')
    X_u8_arr = f_out.create_earray(
        g_out, 'X_u8', tables.UInt8Atom(), (0, 3, 40, 40), expectedrows=len(lines),
        filters=filters)

    y = []
    for line in tqdm.tqdm(lines):
        image_filename, gt, _ = line.split()
        image_path = os.path.join(data_path, image_filename)

        if not os.path.exists(image_path):
            print('Could not find image file {} mentioned in annotations'.format(image_path))
            return
        image_data = cv2.imread(image_path)[:, :, ::-1]

        X_u8_arr.append(image_data.transpose(2, 0, 1)[None, ...])
        y.append(int(gt))

    y = np.array(y, dtype=np.int32)
    f_out.create_array(g_out, 'y', y)

    print('X.shape={}'.format(X_u8_arr.shape))
    print('y.shape={}'.format(y.shape))

    f_out.close()


if __name__ == '__main__':
    prepare()