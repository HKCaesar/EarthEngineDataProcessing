import rasterio
import numpy as np
import pandas as pd
import os
import shelve
import datetime
import bisect
from multiprocessing import Pool


def create_folders(shelve_dir):
    if not os.path.exists(shelve_dir):
        os.mkdir(shelve_dir)
    if not os.path.exists(shelve_dir + 'old/'):
        os.mkdir(shelve_dir + 'old/')
    if not os.path.exists(shelve_dir + 'train/'):
        os.mkdir(shelve_dir + 'train/')
    if not os.path.exists(shelve_dir + 'old/maps'):
        os.mkdir(shelve_dir + 'old/maps')
    if not os.path.exists(shelve_dir + 'old/maps_interpolated'):
        os.mkdir(shelve_dir + 'old/maps_interpolated')
    if not os.path.exists(shelve_dir + 'new/'):
        os.mkdir(shelve_dir + 'new/')
    if not os.path.exists(shelve_dir + 'new/maps'):
        os.mkdir(shelve_dir + 'new/maps')
    if not os.path.exists(shelve_dir + 'new/maps_interpolated'):
        os.mkdir(shelve_dir + 'new/maps_interpolated')


# define multiprocessing method for reading / combining raw images and masks
class ImageReader:
    def __init__(self, time_start, image_dir, mask_dir, shelve_dir):
        self.time_start = time_start
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.shelve_dir = shelve_dir

    def __call__(self, fn):
        arr_img = rasterio.open(self.image_dir + fn).read()
        arr_msk = rasterio.open(self.mask_dir + fn).read()
        ts = self.time_start[self.time_start['system:index'] == fn.split('.')[0]]['system:time_start'].iloc[0]
        combo = np.concatenate((arr_img, arr_msk), axis=0)
        return str(ts), combo


def read_image_data(image_dir='2014/images/',
                    mask_dir='2014/masks/',
                    table_dir='2014/tables/LC8_SR.csv', 
                    shelve_dir=None,
                    processes=1):
    """
    reads data from raw images and store combined band and mask data in a shelve indexed by the image timestamp
    :param image_dir: directory of the images (GeoTiff) containing band info
    :param mask_dir: directory of the masks GeoTiff
    :param table_dir: path to the table with image metadata
    :param shelve_dir: directory for storing the shelf file containing processed data
    :param processes: number of processes to use
    :return: a Shelf View of the processed data, or a dict if shelve_dir is not specified
    """
    # combined = {}
    if shelve_dir is None:
        combined = {}  # write to file instead
    else:
        try:
            os.remove(shelve_dir + 'combined.*')
        except FileNotFoundError:
            pass
        combined = shelve.open(shelve_dir + 'combined')

    table = pd.read_csv(table_dir)
    time_start = table[['system:index', 'system:time_start']]

    p = Pool(processes)
    for ts, combo in p.imap(ImageReader(time_start, image_dir, mask_dir, shelve_dir), os.listdir(image_dir)):
        # combined[ts] = img
        fp = np.memmap(shelve_dir + 'maps/' + str(ts), dtype='int16', mode='w+', shape=combo.shape)
        fp[:] = combo[:]
        combined[ts] = combo.shape
        del fp, combo
    p.close()

    # for fn in os.listdir(image_dir):
    #     arr_img = rasterio.open(image_dir + fn).read()
    #     arr_msk = rasterio.open(mask_dir + fn).read()
    #     ts = time_start[time_start['system:index'] == fn.split('.')[0]]['system:time_start'].iloc[0]
    #     combined[str(ts)] = np.concatenate((arr_img, arr_msk), axis=0)

    return combined


def get_boolean_mask(image, level=1):
    """
    Generate a 1/0 cloud mask for a given image
    :param image: input image
    :param level: minimal confidence level
    :return: a mask matrix
    """
    # cfmask = image[3, :, :]
    # cfmask_conf = image[4, :, :]
    valid = (image[0, :, :] > 0)
    valid = valid & ((image[3, :, :] == 0) | (image[3, :, :] == 1))
    return valid & (image[4, :, :] <= level)


def zigzag_integer_pairs(max_x, max_y):
    """
    Generator
    Generate pairs of integers like (0,0), (0,1), (1,0), (0,2), (1,1), (2,0), ...
    Used for selecting images used for interpolation operations
    :param max_x: maximum number for the first element
    :param max_y: maximum number for the second element
    """
    total = 0
    x = 0
    while total <= max_x + max_y:
        if total - x <= max_y:
            yield (x, total - x)
        if x <= min(max_x - 1, total - 1):
            x += 1
        else:
            total += 1
            x = 0
            

def interpolate(timestamp, maps, max_days_apart=None, shelve_dir=None):
    # assuming dict keys are strings
    """
    Calculate the interpolated image at a given timestamp
    :param timestamp: timestamp for interpolation (as int)
    :param maps: a dict or shelf view for image data, assuming timestamps are strings
    :param max_days_apart: maximum days allowed between two interpolated value and a known value to be used as input
    in interpolation before the algorithm reports a missing value
    :return: the interpolated image array
    """
    keys = list(maps.keys())
    times = [int(k) for k in keys]
    times.sort()
    pos = bisect.bisect(times, timestamp)
    # n_times = len(times)
    # dims = dataset[str(times[0])].shape
    dims = maps[next(iter(maps.keys()))]
    interpolated = np.ones((3, dims[1], dims[2]), dtype='int16') * (-9999)
    times_before = times[:pos]
    times_before.reverse()
    times_after = times[pos:]
    unfilled = np.ones(dims[1:], dtype=bool)
    for pair in zigzag_integer_pairs(len(times_before) - 1, len(times_after) - 1):
        before = times_before[pair[0]]
        after = times_after[pair[1]]
        delta = datetime.datetime.fromtimestamp(after/1000) - datetime.datetime.fromtimestamp(before/1000)
        if max_days_apart is None or delta.days < max_days_apart:
            alpha = 1.0 * (timestamp - before) / (after - before)
            image_before = np.memmap(shelve_dir + 'maps/' + str(before), dtype='int16', mode='r', shape=maps[str(before)])
            image_after = np.memmap(shelve_dir + 'maps/' + str(after), dtype='int16', mode='r', shape=maps[str(after)])
            mask_before = get_boolean_mask(image_before)
            mask_after = get_boolean_mask(image_after)
            common_unmasked = mask_before & mask_after
            del mask_before, mask_after
            valid = common_unmasked & unfilled
            del common_unmasked
            #         fitted = dataset[before][:3, :, :] * alpha + dataset[after][:3, :, :] * (1 - alpha)
            fitted = np.zeros((3, dims[1], dims[2]))
            fitted[:, valid] = image_before[:3, valid] * alpha + image_after[:3, valid] * (1 - alpha)
            unfilled = unfilled ^ valid
            interpolated[:, valid] = fitted[:, valid]
            del image_before, image_after, valid, fitted
    times.sort(key=lambda t: abs(t - timestamp))
    for ts in times:
        delta = datetime.datetime.fromtimestamp(ts / 1000) - datetime.datetime.fromtimestamp(timestamp / 1000)
        if max_days_apart is None or abs(delta.days) < max_days_apart:
            img_nearest = np.memmap(shelve_dir + 'maps/' + str(ts), dtype='int16', mode='r', shape=maps[str(ts)])
            mask = get_boolean_mask(img_nearest)
            valid = mask & unfilled
            unfilled = unfilled ^ valid
            interpolated[:, valid] = img_nearest[:3, valid]
            del img_nearest, mask, valid
    return interpolated


class Interpolater(object):

    def __init__(self, maps, max_days_apart=None, shelve_dir=None):
        self.maps = maps
        self.max_days_apart = max_days_apart
        self.shelve_dir = shelve_dir

    def __call__(self, ts):
        return str(ts), interpolate(ts, self.maps, self.max_days_apart, self.shelve_dir)


def interpolate_images(timestamps, maps, max_days_apart=None, processes=1, shelve_dir=None):
    if processes == 1:
        return {ts: interpolate(ts, maps, max_days_apart) for ts in timestamps}
    else:
        try:
            os.remove(shelve_dir + 'interpolated.*')
        except FileNotFoundError:
            pass
        imgs = shelve.open(shelve_dir + 'interpolated')
        p = Pool(processes)
        for ts, img in p.imap(Interpolater(maps, max_days_apart, shelve_dir), timestamps):
            fp = np.memmap(shelve_dir + 'maps_interpolated/' + str(ts), dtype='int16', mode='w+', shape=img.shape)
            fp[:] = img[:]
            imgs[ts] = img.shape
            del fp, img
        p.close()
        return imgs


def open_img(img_id, img_dir, res=None):
    return np.memmap(img_dir + img_id, dtype='int16', mode='r', shape=res)


def generate_coordinate_columns(x, y):
    res = np.zeros((x * y, 2), dtype=int)
    res[:, 0] = np.array([i for i in range(x) for j in range(y)])
    res[:, 1] = np.array([i for i in range(y)] * x)
    return res


def extract_label_column(label_array):
    df = pd.DataFrame(label_array)
    df['x'] = df.index
    label_set = pd.melt(df, id_vars='x')
    return label_set.sort_values(by=['x', 'variable'])['value'].as_matrix()


def extract_partial_set(index_range, imgs, img_dir, labels=None):
    start, end = index_range
    items = list(imgs.items())
    items.sort()
    df = pd.DataFrame()
    for i in range(len(items)):
        ts, res = items[i]
        img = open_img(ts, img_dir, (res[0], res[1] * res[2]))
        for band in range(3):
            df[ts + '_' + str(band)] = img[band, start: end]
    if labels is not None:
        df['label'] = extract_label_column(labels)[start: end]
    return df


def partial_set_iterator(step, imgs, img_dir, labels=None):
    ts0, res0 = (next(iter(imgs.items())))
    length = res0[1] * res0[2]
    if length < step:
        ranges = [(0, length)]
    else:
        stops = list(range(0, length, step))
        ranges = [(stops[i], stops[i+1]) for i in range(len(stops) - 1)] + [(stops[-1], length)]
    for index_range in ranges:
        yield index_range, extract_partial_set(index_range, imgs, img_dir, labels)


def store_set(step, imgs, img_dir, shelve_dir, name='trains', labels=None):
    try:
        os.remove(shelve_dir + name + '.*')
    except FileNotFoundError:
        pass
    trains = shelve.open(shelve_dir + name)
    n = 0
    for train in partial_set_iterator(step, imgs, img_dir, labels):
        trains[str(n)] = train
        n += 1
    return trains


def old_data_preprocess_workflow(image_dir, mask_dir, table_dir, shelve_root_dir, labels, new_table_dir=None,
                                 max_days_apart=60, processes=1, step=250000, timestamps=None):
    """
    preprocess training data, interpolating it using the next year's available data timestamps
    :param image_dir: directory of the image files
    :param mask_dir: directory of the mask files
    :param table_dir: path to the metadata table
    :param new_table_dir: path to the next year's metadata table (not used if timestamps are given)
    :param shelve_root_dir: root directory for shelf and memmap files to be created
    :param labels: class label map in array form
    :param max_days_apart: maximum days between interpolated value and an known value before reporting missing value
    :param processes: number of processes for multiprocessing
    :param step: batch size for training set generation
    :param timestamps: timestamps for interpolation (not necessary if new_table_dir is given)
    :return:
    """
    print("reading data...")
    maps = read_image_data(image_dir, mask_dir, table_dir, shelve_root_dir + 'old/', processes)
    print("reading new timestamps...")
    # res = ds[list(ds.keys())[0]].shape[1:]
    new_table = pd.read_csv(new_table_dir)
    new_times = list(new_table['system:time_start'])
    if timestamps is None:
        times_to_fit = []
        for t in new_times:
            dt = datetime.datetime.fromtimestamp(t / 1000)
            dt = dt.replace(year=dt.year - 1)
            times_to_fit += [int(dt.timestamp() * 1000)]
    else:
        times_to_fit = timestamps
    times_to_fit.sort()
    print("interpolating images...")
    imgs = interpolate_images(times_to_fit, maps, max_days_apart, processes, shelve_root_dir + 'old/')
    print("generating sets...")
    trains = store_set(step, imgs, shelve_root_dir + 'old/maps_interpolated/', shelve_root_dir, 'trains', labels)
    return trains


def new_data_preprocess_workflow(image_dir, mask_dir, table_dir, shelve_root_dir,
                                 max_days_apart=60, processes=1, step=250000):
    """
    preprocess training data, interpolating it using the next year's available data timestamps
    :param image_dir: directory of the image files
    :param mask_dir: directory of the mask files
    :param table_dir: path to the metadata table
    :param shelve_root_dir: root directory for shelf and memmap files to be created
    :param max_days_apart: maximum days between interpolated value and an known value before reporting missing value
    :param processes: number of processes for multiprocessing
    :param step: batch size for training set generation
    :return:
    """
    print("reading data...")
    maps = read_image_data(image_dir, mask_dir, table_dir, shelve_root_dir + 'new/', processes)
    print("reading new timestamps...")
    # res = ds[list(ds.keys())[0]].shape[1:]
    new_table = pd.read_csv(table_dir)
    times_to_fit = list(new_table['system:time_start'])
    times_to_fit.sort()
    print("interpolating images...")
    imgs = interpolate_images(times_to_fit, maps, max_days_apart, processes, shelve_root_dir + 'new/')
    print("generating sets...")
    trains = store_set(step, imgs, shelve_root_dir + 'new/maps_interpolated/', shelve_root_dir, 'to_predict')
    return trains


# def convert_to_dataframe(key_image):
#     frame = pd.Panel(key_image[1]).to_frame()
#     return key_image[0], frame


# def make_set(images):
#     # times = list(images.keys())
#     # times.sort()
#     res = pd.concat([convert_to_dataframe(i)[1] for i in images.items()], axis=1, keys=images.keys())
# #     res = pd.concat(list(map(convert_to_dataframe, images.values())), axis=0)
#     return res.reset_index()
#
#
# def store_set(images, processes=1, shelve_dir=None):
#     try:
#         os.remove(shelve_dir + 'pixels.*')
#     except FileNotFoundError:
#         pass
#     pix = shelve.open(shelve_dir + 'pixels')
#     p = Pool(processes)
#     for ts, frame in p.imap(convert_to_dataframe, images.items()):
#         pix[ts] = frame.as_matrix()
#     p.close()
#     return pix



# class PixelRowReader(object):
#
#     def __init__(self, pixels, limits):
#         self.pixels = pixels
#         self.limits = limits
#
#     def __call__(self, ts):
#         return self.pixels[ts][self.limits[0]: self.limits[1], :]
#
#
# def combine_set(pixels, shelve_dir=None, res=None, step=250000, processes=1, labels=None):
#     times = list(pixels.keys())
#     times.sort()
#     try:
#         os.remove(shelve_dir + 'set.*')
#     except FileNotFoundError:
#         pass
#     pix_set = shelve.open(shelve_dir + 'set')
#     length = res[0] * res[1]
#     if length < step:
#         ranges = [(0, length)]
#     else:
#         stops = list(range(0, length, step))
#         ranges = [(stops[i], stops[i+1]) for i in range(len(stops) - 1)] + [(stops[-1], length)]
#     n = 0
#     if labels is not None:
#         label_column = extract_label_column(labels)
#     else:
#         label_column = None
#     for limits in ranges:
#         p = Pool(processes)
#         pieces = p.map(PixelRowReader(pixels, limits), times)
#         p.close()
#         if label_column is not None:
#             pieces += [label_column[limits[0]: limits[1]].reshape(limits[1] - limits[0], 1)]
#         res = np.concatenate(pieces, axis=1)
#         pix_set[str(n)] = res
#         n += 1
#     return pix_set


# def old_data_preprocess_workflow(image_dir, mask_dir, table_dir, shelve_root_dir, labels, new_table_dir=None,
#                                  max_days_apart=60, processes=1, step=250000, timestamps=None):
#     """
#     preprocess training data, interpolating it using the next year's available data timestamps
#     :param image_dir: directory of the image files
#     :param mask_dir: directory of the mask files
#     :param table_dir: path to the metadata table
#     :param new_table_dir: path to the next year's metadata table (not used if timestamps are given)
#     :param shelve_root_dir: root directory for shelf files to be created
#     :param labels: class label map in array form
#     :param max_days_apart: maximum days between interpolated value and an known value before reporting missing value
#     :param processes: number of processes for multiprocessing
#     :param step: batch size for training set generation
#     :param timestamps: timestamps for interpolation (not necessary if new_table_dir is given)
#     :return:
#     """
#     print("reading data...")
#     ds = read_image_data(image_dir, mask_dir, table_dir, shelve_root_dir + 'old/', processes)
#     print("reading new timestamps...")
#     res = ds[list(ds.keys())[0]].shape[1:]
#     new_table = pd.read_csv(new_table_dir)
#     new_times = list(new_table['system:time_start'])
#     if timestamps is None:
#         times_to_fit = []
#         for t in new_times:
#             dt = datetime.datetime.fromtimestamp(t / 1000)
#             dt = dt.replace(year=dt.year - 1)
#             times_to_fit += [int(dt.timestamp() * 1000)]
#     else:
#         times_to_fit = timestamps
#     times_to_fit.sort()
#     print("interpolating images...")
#     imgs = interpolate_images(times_to_fit, ds, max_days_apart, processes, shelve_root_dir)
#     ds.close()
#     print("storing sets...")
#     pix = store_set(imgs, processes, shelve_root_dir + 'old/')
#     imgs.close()
#     print("combining sets...")
#     train = combine_set(pix, shelve_root_dir + 'old/', res, step, processes, labels)
#     return train


# # old
# def generate_interpolated_training_set(ds, ds_new=None, label_img_dir='labels/labels.png',
#                                        labels=None, max_days_apart=None):
#     # assuming dict keys are strings
#     keys = list(ds_new.keys())
#     times = [int(k) for k in keys]
#     times.sort()
#     times_to_fit = []
#     for t in times:
#         dt = datetime.datetime.fromtimestamp(t / 1000)
#         dt = dt.replace(year=dt.year - 1)
#         times_to_fit += [int(dt.timestamp() * 1000)]
#
#     fitted_images = interpolate_images(times_to_fit, ds, max_days_apart)
#     train = make_set(fitted_images)
#
#     if ds_new is not None:
#         new_no_cloud = interpolate_images(dataset=ds_new, timestamps=list(ds_new.keys()),
#                                           max_days_apart=max_days_apart)
#         to_predict = make_set(new_no_cloud)
#         to_predict.columns = [c for c in to_predict.columns]
#         to_predict = to_predict.rename(columns={('major', ''): 'x', ('minor', ''): 'y'})
#     else:
#         to_predict = None
#
#     if labels is None:
#         label_image = Image.open(label_img_dir)
#         labels = np.array(label_image)
#         pix_array = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
#         label_colours = np.vstack({tuple(row) for row in pix_array})
#         label_map = {tuple(row): i for row, i in zip(label_colours, range(label_colours.shape[0]))}
#         label_converted = np.apply_along_axis(lambda x: label_map[tuple(x)], 2, labels)
#     else:
#         label_converted = labels
#
#     df = pd.DataFrame(label_converted)
#     df['x'] = df.index
#     label_set = pd.melt(df, id_vars='x')
#     train_labelled = pd.merge(train, label_set, left_on=['major', 'minor'], right_on=['x', 'variable'], copy=False)
#     train_labelled = train_labelled.drop(['x', 'variable'], axis=1).rename(
#         columns={('major', ''): 'x', ('minor', ''): 'y', 'value': 'label'})
#     return train_labelled, to_predict
#
#
# def generate_interpolated_set_from_timestamps(ds, times, labels=None, on_self=False, max_days_apart=None):
#     # times need to be int already
#     times.sort()
#     times_to_fit = []
#     # TODO: separate timestamp lagging and fitting
#     if not on_self:
#         for t in times:
#             dt = datetime.datetime.fromtimestamp(t / 1000)
#             dt = dt.replace(year=dt.year - 1)
#             times_to_fit += [int(dt.timestamp() * 1000)]
#     else:
#         times_to_fit = times
#
#     fitted_images = interpolate_images(times_to_fit, ds, max_days_apart)
#     train = make_set(fitted_images)
#
#     if labels is not None:
#         df = pd.DataFrame(labels)
#         df['x'] = df.index
#         label_set = pd.melt(df, id_vars='x')
#         train_labelled = pd.merge(train, label_set, left_on=['major', 'minor'], right_on=['x', 'variable'], copy=False)
#         train_labelled = train_labelled.drop(['x', 'variable'], axis=1).rename(
#             columns={('major', ''): 'x', ('minor', ''): 'y', 'value': 'label'})
#         return train_labelled
#     else:
#         train.columns = [c for c in train.columns]
#         train = train.rename(columns={('major', ''): 'x', ('minor', ''): 'y'})
#         return train
