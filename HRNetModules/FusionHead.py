import tensorflow as tf
from .FusionBlock import UpScaleFusionBranch
from tensorflow.keras.layers import Concatenate


class FusionHead(tf.keras.Model):
    def __init__(self, name):
        super(FusionHead, self).__init__(name=name)
        self.branch_4_1 = UpScaleFusionBranch(256, 32, name='upscale-4-1')
        self.branch_3_1 = UpScaleFusionBranch(128, 32, name='upscale-3-1')
        self.branch_2_1 = UpScaleFusionBranch(64, 32, name='upscale-2-1')
        self.concat = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        input_branch_1 = inputs[0]
        input_branch_2 = inputs[1]
        input_branch_3 = inputs[2]
        input_branch_4 = inputs[3]

        branch_4_1 = self.branch_4_1(input_branch_4)
        branch_3_1 = self.branch_3_1(input_branch_3)
        branch_2_1 = self.branch_2_1(input_branch_2)

        final = self.concat([branch_4_1, branch_3_1, branch_2_1, input_branch_1])
        return final
