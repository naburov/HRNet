from HRnet import HRNet
import tensorflow as tf
import numpy as np


input_image = np.zeros(shape=(1, 384, 384, 3))
model = HRNet(out_classes=4)
# model.build((None, 384, 384, 3))
# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.RMSprop())
# out = model(input_image)
with tf.device("/cpu:0"):
    out = model(input_image)
    print(out.shape)

# tf.keras.utils.plot_model(model, 'model.png', show_shapes=False)
