import click


@click.command()
@click.option('--width', type=int, default=40)
@click.option('--height', type=int, default=40)
@click.option('--ignore_roi', is_flag=True, default=False)
def prepare(width, height, ignore_roi):
    import os
    import sys
    import numpy as np
    import tables
    import pandas as pd
    import tqdm
    import domain_datasets
    import cv2

    path = domain_datasets.get_data_dir('gtsrb')

    output_path = os.path.join(path, 'gtsrb.h5')
    print('Creating {}...'.format(output_path))
    f_out = tables.open_file(output_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'gtsrb', 'GTSRB data')
    filters = tables.Filters(complevel=9, complib='blosc')
    train_X_arr = f_out.create_earray(
        g_out, 'train_X_u8', tables.UInt8Atom(), (0, 3, height, width), filters=filters)
    train_y_arr = f_out.create_earray(
        g_out, 'train_y', tables.Int32Atom(), (0,))
    test_X_arr = f_out.create_earray(
        g_out, 'test_X_u8', tables.UInt8Atom(), (0, 3, height, width), filters=filters)
    test_y_arr = f_out.create_earray(
        g_out, 'test_y', tables.Int32Atom(), (0,))

    train_path = os.path.join(path, 'Final_Training', 'Images')
    test_path = os.path.join(path, 'Final_Test', 'Images')

    if not os.path.exists(train_path):
        print('ERROR!!! Training images path {} does not exist'.format(train_path))
        return

    if not os.path.exists(test_path):
        print('ERROR!!! Test images path {} does not exist'.format(test_path))
        return


    def load_image_dir(X_arr, y_arr, dir_path, anno_path):
        if not os.path.exists(anno_path):
            print('ERROR!!! Could not find annotations file {}'.format(anno_path))
            return False

        annotations = pd.read_csv(anno_path, sep=';')

        for index, row in tqdm.tqdm(annotations.iterrows(), desc='Images', total=len(annotations.index)):
            image_filename = row['Filename']
            image_path = os.path.join(dir_path, image_filename)
            if not os.path.exists(image_path):
                print('ERROR!!!  Could not find image file {} mentioned in annotations'.format(image_path))
                return False
            image_data = cv2.imread(image_path)[:, :, ::-1]

            if not ignore_roi:
                # Crop out the region of interest
                roi_x1 = int(row['Roi.X1'])
                roi_x2 = int(row['Roi.X2'])
                roi_y1 = int(row['Roi.Y1'])
                roi_y2 = int(row['Roi.Y2'])
                image_data = image_data[roi_y1:roi_y2, roi_x1:roi_x2, :]

            image_data = cv2.resize(image_data, (width, height), interpolation=cv2.INTER_AREA)

            class_id = int(row['ClassId'])

            X_arr.append(image_data.transpose(2, 0, 1)[None, ...])
            y_arr.append(np.array([class_id], dtype=np.int32))

        return True

    print('Processing training data...')
    for clf_dir_name in tqdm.tqdm(os.listdir(train_path), desc='Class'):
        clf_ndx = int(clf_dir_name)
        clf_path = os.path.join(train_path, clf_dir_name)
        anno_path = os.path.join(clf_path, 'GT-{:05d}.csv'.format(clf_ndx))
        success = load_image_dir(train_X_arr, train_y_arr, clf_path, anno_path)
        if not success:
            f_out.close()
            os.remove(output_path)
            return

    print('train_X.shape={}'.format(f_out.root.gtsrb.train_X_u8.shape))
    print('train_y.shape={}'.format(f_out.root.gtsrb.train_y.shape))


    print('Processing test data...')
    test_anno_path = os.path.join(path, 'GT-final_test.csv')
    success = load_image_dir(test_X_arr, test_y_arr, test_path, test_anno_path)
    if not success:
        f_out.close()
        os.remove(output_path)
        return
    print('test_X.shape={}'.format(f_out.root.gtsrb.test_X_u8.shape))
    print('test_y.shape={}'.format(f_out.root.gtsrb.test_y.shape))

    f_out.close()

if __name__ == '__main__':
    prepare()