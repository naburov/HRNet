import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D


class DownscaleFusionBranch(tf.keras.Model):
    def __init__(self, input_filters, output_filters, name):
        super(DownscaleFusionBranch, self).__init__(self, name=name)
        assert output_filters > input_filters
        temp_out_filters = output_filters
        self.conv_layers = []
        while temp_out_filters > input_filters * 2:
            self.conv_layers.append(Conv2D(32, 3, 2, padding='same'))
            self.conv_layers.append(BatchNormalization(name='batch_norm_{0}'.format(output_filters)))
            temp_out_filters /= 2
        self.conv_layers.append(Conv2D(output_filters, 3, 2, padding='same'))
        self.conv_layers.append(BatchNormalization(name='batch_norm_{0}'.format(output_filters)))

    def call(self, inputs, training=None, mask=None):
        x = self.conv_layers[0](inputs, training=training)
        for layer in self.conv_layers[1:]:
            if not 'batch_norm' in layer.name:
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class UpScaleFusionBranch(tf.keras.Model):
    def __init__(self, input_filters, output_filters, name):
        assert input_filters > output_filters
        super(UpScaleFusionBranch, self).__init__(self, name=name)
        upscale_factor = int(input_filters / output_filters)
        self.conv = Conv2D(output_filters, 1, strides=1, padding='same')
        self.bn = BatchNormalization()
        self.upsample_layer = UpSampling2D((upscale_factor, upscale_factor), interpolation='bilinear')

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs, training=training)
        x = self.bn(x)
        x = self.upsample_layer(x)
        return x


class FusionBlock_Stage_1(tf.keras.Model):
    def __init__(self, name):
        super(FusionBlock_Stage_1, self).__init__(name=name)
        self.branch_1_2 = DownscaleFusionBranch(32, 64, name='downscale-1-2')

    def call(self, inputs, training=None, mask=None):
        branch_2 = self.branch_1_2(inputs)
        return inputs, branch_2


class FusionBlock_Stage_2(tf.keras.Model):
    def __init__(self, name):
        super(FusionBlock_Stage_2, self).__init__(name=name)
        self.branch_1_2 = DownscaleFusionBranch(32, 64, name='downscale-1-2')
        self.branch_1_3 = DownscaleFusionBranch(32, 128, name='downscale-1-2')
        self.branch_2_3 = DownscaleFusionBranch(64, 128, name='downscale-2-3')
        self.branch_2_1 = UpScaleFusionBranch(64, 32, name='upscale-2-1')

    def call(self, inputs, training=None, mask=None):
        input_branch_1 = inputs[0]
        input_branch_2 = inputs[1]

        branch_1 = self.branch_2_1(input_branch_2) + input_branch_1
        branch_2 = self.branch_1_2(input_branch_1) + input_branch_2
        branch_3 = self.branch_1_3(input_branch_1) + self.branch_2_3(input_branch_2)
        return branch_1, branch_2, branch_3


class FusionBlock_Stage_3(tf.keras.Model):
    def __init__(self, name):
        super(FusionBlock_Stage_3, self).__init__(name=name)
        self.branch_1_2 = DownscaleFusionBranch(32, 64, name='downscale-1-2')
        self.branch_1_3 = DownscaleFusionBranch(32, 128, name='downscale-1-3')

        self.branch_2_1 = UpScaleFusionBranch(64, 32, name='upscale-2-1')
        self.branch_2_3 = DownscaleFusionBranch(64, 128, name='downscale-2-3')

        self.branch_3_1 = UpScaleFusionBranch(128, 32, name='upscale-3-1')
        self.branch_3_2 = UpScaleFusionBranch(128, 64, name='upscale-3-2')

        self.branch_1_4 = DownscaleFusionBranch(32, 256, name='downscale-1-4')
        self.branch_2_4 = DownscaleFusionBranch(64, 256, name='downscale-2-4')
        self.branch_3_4 = DownscaleFusionBranch(128, 256, name='downscale-3-4')

    def call(self, inputs, training=None, mask=None):
        input_branch_1 = inputs[0]
        input_branch_2 = inputs[1]
        input_branch_3 = inputs[2]

        branch_1_2 = self.branch_1_2(input_branch_1)
        branch_3_2 = self.branch_3_2(input_branch_3)

        branch_2_1 = self.branch_2_1(input_branch_2)
        branch_3_1 = self.branch_3_1(input_branch_3)

        branch_1_3 = self.branch_1_3(input_branch_1)
        branch_2_3 = self.branch_2_3(input_branch_2)

        branch_1_4 = self.branch_1_4(input_branch_1)
        branch_2_4 = self.branch_2_4(input_branch_2)
        branch_3_4 = self.branch_3_4(input_branch_3)

        branch_1 = input_branch_1 + branch_2_1 + branch_3_1
        branch_2 = input_branch_2 + branch_1_2 + branch_3_2
        branch_3 = input_branch_3 + branch_1_3 + branch_2_3
        branch_4 = branch_1_4 + branch_2_4 + branch_3_4
        return branch_1, branch_2, branch_3, branch_4


class FusionBlock_Stage_4(tf.keras.Model):
    def __init__(self, name):
        super(FusionBlock_Stage_4, self).__init__(name=name)
        self.branch_1_2 = DownscaleFusionBranch(32, 64, name='downscale-1-2')
        self.branch_1_3 = DownscaleFusionBranch(32, 128, name='downscale-1-3')
        self.branch_1_4 = DownscaleFusionBranch(32, 256, name='downscale-1-4')

        self.branch_2_1 = UpScaleFusionBranch(64, 32, name='upscale-2-1')
        self.branch_2_3 = DownscaleFusionBranch(64, 128, name='downscale-2-3')
        self.branch_2_4 = DownscaleFusionBranch(64, 256, name='downscale-2-4')

        self.branch_3_1 = UpScaleFusionBranch(128, 32, name='upscale-3-1')
        self.branch_3_2 = UpScaleFusionBranch(128, 64, name='upscale-3-2')
        self.branch_3_4 = DownscaleFusionBranch(128, 256, name='downscale-3-4')

        self.branch_4_1 = UpScaleFusionBranch(256, 32, name='upscale-4-1')
        self.branch_4_2 = UpScaleFusionBranch(256, 64, name='upscale-4-2')
        self.branch_4_3 = UpScaleFusionBranch(256, 128, name='upscale-4-3')

    def call(self, inputs, training=None, mask=None):
        input_branch_1 = inputs[0]
        input_branch_2 = inputs[1]
        input_branch_3 = inputs[2]
        input_branch_4 = inputs[3]

        branch_1_2 = self.branch_1_2(input_branch_1)
        branch_3_2 = self.branch_3_2(input_branch_3)
        branch_4_2 = self.branch_4_2(input_branch_4)

        branch_2_1 = self.branch_2_1(input_branch_2)
        branch_3_1 = self.branch_3_1(input_branch_3)
        branch_4_1 = self.branch_4_1(input_branch_4)

        branch_1_3 = self.branch_1_3(input_branch_1)
        branch_2_3 = self.branch_2_3(input_branch_2)
        branch_4_3 = self.branch_4_3(input_branch_4)

        branch_1_4 = self.branch_1_4(input_branch_1)
        branch_2_4 = self.branch_2_4(input_branch_2)
        branch_3_4 = self.branch_3_4(input_branch_3)

        branch_1 = input_branch_1 + branch_2_1 + branch_3_1 + branch_4_1
        branch_2 = input_branch_2 + branch_1_2 + branch_3_2 + branch_4_2
        branch_3 = input_branch_3 + branch_1_3 + branch_2_3 + branch_4_3
        branch_4 = branch_1_4 + branch_2_4 + branch_3_4 + input_branch_4
        return branch_1, branch_2, branch_3, branch_4
