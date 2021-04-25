import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization


class BasicBlockBlock(tf.keras.Model):
    def __init__(self, filters, name):
        super(BasicBlockBlock, self).__init__(name=name)
        self.conv_1 = Conv2D(filters, 3, strides=(1, 1), padding='same')
        self.conv_2 = Conv2D(filters, 3, strides=(1, 1), padding='same')

        self.bn_1 = BatchNormalization()
        self.bn_2 = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs, training=training)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x, training=training)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x += inputs
        return x


class BasicBlock(tf.keras.Model):
    def __init__(self, filters, name):
        super(BasicBlock, self).__init__(name=name)
        self.bblock_1 = BasicBlockBlock(filters, name='bblock-1')
        self.bblock_2 = BasicBlockBlock(filters, name='bblock-2')
        self.bblock_3 = BasicBlockBlock(filters, name='bblock-3')
        self.bblock_4 = BasicBlockBlock(filters, name='bblock-4')

    def call(self, inputs, training=None, mask=None):
        x = self.bblock_1(inputs, training=training)
        x = self.bblock_2(x, training=training)
        x = self.bblock_3(x, training=training)
        x = self.bblock_4(x, training=training)
        return x
