import numpy as np
import gdal
import os
import pandas as pd
import datetime
import bisect
from PIL import Image


def read_image_data(image_dir='2014/images/',
                    mask_dir='2014/masks/',
                    table_dir='2014/tables/LC8_SR.csv'):
    combined = {}
    for fn in os.listdir(image_dir):
        raw_img = gdal.Open(image_dir + fn)
        arr_img = raw_img.ReadAsArray()
        raw_msk = gdal.Open(mask_dir + fn)
        arr_msk = raw_msk.ReadAsArray()
        combined[fn.split('.')[0]] = np.concatenate((arr_img, arr_msk), axis=0)
    table = pd.read_csv(table_dir)
    time_start = table[['system:index', 'system:time_start']]
    ds = {}
    for k, v in combined.items():
        ts = time_start[time_start['system:index'] == k]['system:time_start'].iloc[0]
        ds[ts] = v
    return ds


def get_boolean_mask(image, level=1):
    cfmask = image[3, :, :]
    cfmask_conf = image[4, :, :]
    return ((cfmask == 0) | (cfmask == 1)) & (cfmask_conf <= level)


def zigzag_integer_pairs(max_x, max_y):
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


def interpolate(timestamp, dataset, max_days_apart=None):
    times = list(dataset.keys())
    times.sort()
    pos = bisect.bisect(times, timestamp)
    # n_times = len(times)
    dims = dataset[times[0]].shape
    interpolated = np.ones((3, dims[1], dims[2])) * (-999)
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
            mask_before = get_boolean_mask(dataset[before])
            mask_after = get_boolean_mask(dataset[after])
            common_unmasked = mask_before & mask_after
            valid = common_unmasked & unfilled
            #         fitted = dataset[before][:3, :, :] * alpha + dataset[after][:3, :, :] * (1 - alpha)
            fitted = np.zeros((3, dims[1], dims[2]))
            fitted[:, valid] = dataset[before][:3, valid] * alpha + dataset[after][:3, valid] * (1 - alpha)
            unfilled = unfilled ^ valid
            interpolated[:, valid] = fitted[:, valid]
    times.sort(key=lambda t: abs(t - timestamp))
    for ts in times:
        delta = datetime.datetime.fromtimestamp(ts / 1000) - datetime.datetime.fromtimestamp(timestamp / 1000)
        if max_days_apart is None or abs(delta.days) < max_days_apart:
            mask = get_boolean_mask(dataset[ts])
            valid = mask & unfilled
            unfilled = unfilled ^ valid
            interpolated[:, valid] = dataset[ts][:3, valid]
    return interpolated


def interpolate_images(timestamps, dataset, max_days_apart=None):
    return {ts: interpolate(ts, dataset, max_days_apart) for ts in timestamps}


def convert_to_dataframe(image):
    frame = pd.Panel(image).to_frame()
    return frame


def make_set(images):
    times = list(images.keys())
    times.sort()
    res = pd.concat([convert_to_dataframe(i) for i in images.values()], axis=1, keys=images.keys())
#     res = pd.concat(list(map(convert_to_dataframe, images.values())), axis=0)
    return res.reset_index()


def generate_interpolated_training_set(ds, ds_new=None, label_img_dir='labels/labels.png',
                                       labels=None, max_days_apart=None):
    times = list(ds_new.keys())
    times.sort()
    times_to_fit = []
    for t in times:
        dt = datetime.datetime.fromtimestamp(t / 1000)
        dt = dt.replace(year=dt.year - 1)
        times_to_fit += [int(dt.timestamp() * 1000)]

    fitted_images = interpolate_images(times_to_fit, ds, max_days_apart)
    train = make_set(fitted_images)

    if ds_new is not None:
        new_no_cloud = interpolate_images(dataset=ds_new, timestamps=list(ds_new.keys()),
                                          max_days_apart=max_days_apart)
        to_predict = make_set(new_no_cloud)
        to_predict.columns = [c for c in to_predict.columns]
        to_predict = to_predict.rename(columns={('major', ''): 'x', ('minor', ''): 'y'})
    else:
        to_predict = None

    if labels is None:
        label_image = Image.open(label_img_dir)
        labels = np.array(label_image)
        pix_array = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
        label_colours = np.vstack({tuple(row) for row in pix_array})
        label_map = {tuple(row): i for row, i in zip(label_colours, range(label_colours.shape[0]))}
        label_converted = np.apply_along_axis(lambda x: label_map[tuple(x)], 2, labels)
    else:
        label_converted = labels

    df = pd.DataFrame(label_converted)
    df['x'] = df.index
    label_set = pd.melt(df, id_vars='x')
    train_labelled = pd.merge(train, label_set, left_on=['major', 'minor'], right_on=['x', 'variable'], copy=False)
    train_labelled = train_labelled.drop(['x', 'variable'], axis=1).rename(
        columns={('major', ''): 'x', ('minor', ''): 'y', 'value': 'label'})
    return train_labelled, to_predict


def generate_interpolated_set_from_timestamps(ds, times, labels=None, on_self=False, max_days_apart=None):
    times.sort()
    times_to_fit = []
    # TODO: separate timestamp lagging and fitting
    if not on_self:
        for t in times:
            dt = datetime.datetime.fromtimestamp(t / 1000)
            dt = dt.replace(year=dt.year - 1)
            times_to_fit += [int(dt.timestamp() * 1000)]
    else:
        times_to_fit = times

    fitted_images = interpolate_images(times_to_fit, ds, max_days_apart)
    train = make_set(fitted_images)

    if labels is not None:
        df = pd.DataFrame(labels)
        df['x'] = df.index
        label_set = pd.melt(df, id_vars='x')
        train_labelled = pd.merge(train, label_set, left_on=['major', 'minor'], right_on=['x', 'variable'], copy=False)
        train_labelled = train_labelled.drop(['x', 'variable'], axis=1).rename(
            columns={('major', ''): 'x', ('minor', ''): 'y', 'value': 'label'})
        return train_labelled
    else:
        train.columns = [c for c in train.columns]
        train = train.rename(columns={('major', ''): 'x', ('minor', ''): 'y'})
        return train
