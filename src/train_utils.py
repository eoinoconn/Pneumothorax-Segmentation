import glob
import random

from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils.data_utils import Sequence
import mlflow
import cv2 as cv
import numpy as np
import pydicom
import cv2

from src.mask_functions import rle2mask


def get_available_gpus():
    """
    Gets available GPUs
    Returns:
        (list): list of visible GPUs
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def load_numpy_data_from_json(data_json,
                              resize_shape,
                              data_store_location,
                              classes=8):
    """
    Loads images and labelsfrom a json into a numpy array.
    Args:
        data_json (dict): dictionary with json structure of image locations
        resize_shape (tuple of ints): Desired output dimensions of images
        data_store_location (string): root location of data
    Returns:
        (tuple of numpy arrays): two numpy arrays, one is data, the other is labels
    """
    x_data = []
    y_data = []
    for inst in data_json["data"]:
        relative_url = inst["url"]
        img = cv.imread(data_store_location + relative_url)
        resized_img = cv.resize(img, resize_shape, interpolation=cv.INTER_AREA)
        x_data.append(resized_img)

        label = inst["classes"]["alloy"]
        one_hot_label = np.zeros(classes)
        np.put(one_hot_label, label, 1)
        y_data.append(one_hot_label)

    x_data = np.stack(x_data)
    y_data = np.stack(y_data)

    return (x_data, y_data)


def log_history_obj(history):
    """Parses Keras history object for metrics

    Args:
        history (obj): Keras history obj.
    """

    for metric, values in history.history.items():
        for i, value in enumerate(values):
            mlflow.log_metric(metric, value, step=i)


class data_gen(Sequence):

    def __init__(self,
                 df, files,
                 batch_size=16,
                 img_width=1024,
                 img_height=1024,
                 img_channel=1,
                 shuffle=True):

        self.df = df
        self.files = files
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.image_ids = sorted(self.df["ImageId"].unique())
        self.shuffle = shuffle


    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))


    def  __getitem__(self, idx):

        while True:

            x_data = np.zeros((self.batch_size, self.img_width, self.img_height, self.img_channel), dtype="float32")
            y_data = np.zeros((self.batch_size, self.img_width, self.img_height, self.img_channel), dtype="float32")
            
            train_files_batch = self.files[self.batch_size * idx:self.batch_size * (idx+1)]

            for n, inst in enumerate(train_files_batch):
                image_id = inst.split("/")[-1][:-4]
                if image_id in self.image_ids:
                    img = pydicom.dcmread(inst).pixel_array
                    img = cv2.resize(img, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_CUBIC)
                    img = img.astype("float32") / 255
                    x_data[n] = np.expand_dims(img, axis=2)

                    rle_list = self.df[self.df["ImageId"] == image_id].EncodedPixels.values
                    for rle in rle_list:
                        if rle == "-1":
                            continue
                        else:
                            mask = rle2mask(rle, 1024, 1024).astype("float32")
                            mask = cv2.resize(mask, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_CUBIC)
                            y_data[n] = np.expand_dims(mask, axis=2)
            return (x_data, y_data)
            
    def __call__(self):
        idx = 0
        while True:

            x_data = np.zeros((self.batch_size, self.img_width, self.img_height, self.img_channel), dtype="uint8")
            y_data = np.zeros((self.batch_size, self.img_width, self.img_height, self.img_channel), dtype="uint8")
            
            train_files_batch = self.files[self.batch_size * idx:self.batch_size * (idx+1)]
            idx += 1
            for n, inst in enumerate(train_files_batch):
                image_id = inst.split("/")[-1][:-4]
                if image_id in self.image_ids:
                    img = pydicom.dcmread(inst).pixel_array
                    img = cv2.resize(img, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_CUBIC)
                    x_data[n] = np.expand_dims(img, axis=2)

                    rle_list = self.df[self.df["ImageId"] == image_id].EncodedPixels.values
                    for rle in rle_list:
                        if rle == "-1":
                            continue
                        else:
                            mask = rle2mask(rle, 1024, 1024).astype("uint8")
                            mask = cv2.resize(mask, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_CUBIC)
                            y_data[n] = np.expand_dims(mask, axis=2)
            yield (x_data, y_data)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.files)
