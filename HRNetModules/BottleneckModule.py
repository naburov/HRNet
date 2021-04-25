import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization


class BottleNeckBlock(tf.keras.Model):
    def __init__(self, name):
        super(BottleNeckBlock, self).__init__(name=name)
        self.conv_1 = Conv2D(64, 1, strides=(1, 1,), padding='same')
        self.conv_2 = Conv2D(64, 3, strides=(1, 1,), padding='same')
        self.conv_3 = Conv2D(256, 1, strides=(1, 1), padding='same')

        self.bn_1 = BatchNormalization()
        self.bn_2 = BatchNormalization()
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


class Bottleneck(tf.keras.Model):
    def __init__(self, name):
        super(Bottleneck, self).__init__(name=name)
        self.bottleneck_1 = BottleNeckBlock('bottleneck_1')
        self.bottleneck_2 = BottleNeckBlock('bottleneck_2')
        self.bottleneck_3 = BottleNeckBlock('bottleneck_3')

    def call(self, inputs, training=None, mask=None):
        x = self.bottleneck_1(inputs, training=training)
        x = self.bottleneck_2(x, training=training)
        x = self.bottleneck_3(x, training=training)
        return x
