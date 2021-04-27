import os
import glob
import numpy as np
import random

import tensorflow as tf
import tensorflow_addons as tfa


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, masks_dir, output_resolution, batch_size=32, shuffle=True, seed=42):
        self.image_files = glob.glob(image_dir + '/*/*', recursive=True)
        self.masks_files = glob.glob(masks_dir + '/*/*', recursive=True)
        self.indexes = np.arange(len(self.image_files))
        self.classes = os.listdir(image_dir)
        self.class_dict = {self.classes[i]: i for i in range(len(self.classes))}
        self.output_resolution = output_resolution
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_filenames = [self.image_files[k] for k in indexes]
        masks_filenames = [self.masks_files[k] for k in indexes]
        images = self.read_images(image_filenames)
        masks = self.read_masks(masks_filenames)
        images, masks = self.aug(images, masks)
        return images, masks

    def aug(self, images, masks):
        rotation_angle = random.uniform(-0.174533, 0.174533)
        vertical_flip = random.uniform(0, 1) > 0.5
        horizontal_flip = random.uniform(0, 1) < 0.5
        images = tfa.image.rotate(images, rotation_angle)
        masks = tfa.image.rotate(masks, rotation_angle)
        if horizontal_flip:
            images = tf.image.flip_left_right(images)
            masks = tf.image.flip_left_right(masks)
        if vertical_flip:
            images = tf.image.flip_up_down(images)
            masks = tf.image.flip_up_down(masks)
        return images, masks

    def read_images(self, image_filenames):
        image_tensors = []
        for name in image_filenames:
            image = tf.keras.preprocessing.image.load_img(name, target_size=self.output_resolution)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image_tensors.append(image / 255)
        return tf.stack(image_tensors)

    def read_masks(self, masks_filenames):
        masks_tensors = []
        for name in masks_filenames:
            class_name = name.split('/')[-2]
            image = tf.keras.preprocessing.image.load_img(name, color_mode='grayscale',
                                                          target_size=self.output_resolution)
            image = tf.keras.preprocessing.image.img_to_array(image)
            class_index = self.class_dict[class_name]
            image_tensor = np.zeros(shape=(self.output_resolution[0], self.output_resolution[1], len(self.class_dict)))
            image_tensor[..., class_index] = tf.squeeze(image)
            masks_tensors.append(image_tensor)
        return tf.stack(masks_tensors)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            c = list(zip(self.image_files, self.masks_files))
            random.shuffle(c)
            self.image_files, self.masks_files = zip(*c)
