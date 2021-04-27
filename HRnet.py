import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from HRNetModules.Stem import StemModule
from HRNetModules.FusionBlock import FusionBlock_Stage_1, FusionBlock_Stage_2, FusionBlock_Stage_3, FusionBlock_Stage_4
from HRNetModules.BasicBlock import BasicBlock
from HRNetModules.FusionHead import FusionHead
from HRNetModules.BottleneckModule import Bottleneck


class HRNet(tf.keras.Model):
    def __init__(self, out_classes):
        super(HRNet, self).__init__(name='hrnet')
        self.stem = StemModule(name='stem_module')

        self.stage_1_branch_1 = BasicBlock(32, name='stage_1_branch_1')
        self.stage_2_branch_1 = BasicBlock(32, name='stage_2_branch_1')
        self.stage_3_branch_1 = BasicBlock(32, name='stage_3_branch_1')
        self.stage_4_branch_1 = BasicBlock(32, name='stage_4_branch_1')

        self.stage_2_branch_2 = BasicBlock(64, name='stage_2_branch_2')
        self.stage_3_branch_2 = BasicBlock(64, name='stage_3_branch_2')
        self.stage_4_branch_2 = BasicBlock(64, name='stage_4_branch_2')

        self.stage_3_branch_3 = BasicBlock(128, name='stage_3_branch_3')
        self.stage_4_branch_3 = BasicBlock(128, name='stage_4_branch_3')

        self.stage_4_branch_4 = BasicBlock(256, name='stage_4_branch_4')

        self.fusion_1 = FusionBlock_Stage_1(name='fusion_1')
        self.fusion_2 = FusionBlock_Stage_2(name='fusion_2')
        self.fusion_3 = FusionBlock_Stage_3(name='fusion_3')
        self.fusion_4 = FusionBlock_Stage_4(name='fusion_4')

        self.fusion_head = FusionHead(name='head')
        self.out = Conv2D(out_classes, 3, 1, padding='same', activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        b1 = self.stem(inputs, training=training)
        b1 = self.stage_1_branch_1(b1, training=training)

        b1, b2 = self.fusion_1(b1, training=training)
        b1 = self.stage_2_branch_1(b1, training=training)
        b2 = self.stage_2_branch_2(b2, training=training)

        b1, b2, b3 = self.fusion_2([b1, b2], training=training)
        b1 = self.stage_3_branch_1(b1, training=training)
        b2 = self.stage_3_branch_2(b2, training=training)
        b3 = self.stage_3_branch_3(b3, training=training)

        b1, b2, b3, b4 = self.fusion_3([b1, b2, b3], training=training)
        b1 = self.stage_4_branch_1(b1, training=training)
        b2 = self.stage_4_branch_2(b2, training=training)
        b3 = self.stage_4_branch_3(b3, training=training)
        b4 = self.stage_4_branch_4(b4, training=training)

        b1, b2, b3, b4 = self.fusion_4([b1, b2, b3, b4], training=training)
        final = self.fusion_head([b1, b2, b3, b4], training=training)
        out = self.out(final, training=training)
        out = tf.image.resize(out, [inputs.shape[0], inputs.shape[1]])
        return out
