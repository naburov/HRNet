import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization


class StemModule(tf.keras.Model):
    def __init__(self, name):
        super(StemModule, self).__init__(name=name)
        self.conv_1 = Conv2D(64, 3, strides=2, padding='same')
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(64, 3, strides=2, padding='same')
        self.bn_2 = BatchNormalization()

        self.conv_3 = Conv2D(32, 3, strides=1, padding='same')
        self.bn_3 = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs, training=training)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x, training=training)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_3(x, training=training)
        x = self.bn_3(x, training=training)
        x = tf.nn.relu(x)

        return x
