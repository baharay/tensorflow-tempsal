
from collections import OrderedDict
from PNAS_TF.PNASnet_tf import NetworkImageNet, AuxiliaryHeadImageNet, Cell
from PNAS_TF.genotypes_tf import PNASNet

from tensorflow.keras import layers, Sequential, models
import tensorflow as tf
import torch
import torch.onnx


import sys


class PNASModel(tf.keras.Model):

    def __init__(self, num_chanels=3, train_enc=False, load_weight=1):
        super(PNASModel, self).__init__()
        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)

        if load_weight:
            self.pnas.load_weights(self.path)

        for param in self.pnas.variables:
            param.trainable = train_enc
        
        self.padding = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        self.drop_path_prob = 0

        self.linear_upsampling = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.deconv_layer0 = Sequential(
            [
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(16, 16, 4320)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer1 = Sequential(
            [
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(32, 32, 512+2160)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer2 = Sequential(
            [
                layers.Conv2D(
                    filters=270,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(64, 64, 1080+256)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer3 = Sequential(
            [
                layers.Conv2D(
                    filters=96,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(128, 128, 540)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.decpnv_layer4 = Sequential(
            [
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 192)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer5 = Sequential(
            [
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 128)
                ),
                layers.ReLU(),
                layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 128)
                ),
                layers.Activation('sigmoid'),
            ]
        )
    
    def call(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==1:
                out5 = s1
        
        out5 = self.deconv_layer0(out5)

        x = tf.concat([out5, out4], axis=-1)
        x = self.deconv_layer1(x)

        x = tf.concat([x, out3], axis=-1)
        x = self.deconv_layer2(x)

        x = tf.concat([x, out2], axis=-1)
        x = self.deconv_layer3(x)

        x = tf.concat([x, out1], axis=-1)

        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)
        x = tf.squeeze(x, axis=-1)

        return x


class PNASVolModellast(tf.keras.Model):

    def __init__(self, time_slices, num_channels=3, train_enc=False, load_weight=1):
        super(PNASVolModellast, self).__init__()

        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)

        # if load_weight:
        #     state_dict = models.load_model(self.path)
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         if 'module' in k:
        #             k = 'module.pnas.' + k
        #         else:
        #             k = k.replace('pnas.', '')
        #         new_state_dict[k] = v
        #     self.pnas.load_weights(new_state_dict)
        self.create_model_weights(self.pnas, train_enc)
        self.pnas = tf.saved_model.load("PnasVolModelslast_weights/model_weights.onnx")

        self.padding = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        self.drop_path_prob = 0

        self.linear_upsampling = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.deconv_layer0 = Sequential(
            [
                layers.Conv2D(
                    filters=96, #the dimension of the output space (the number of filters in the convolution).
                    kernel_size=(3, 3), # specifying the size of the convolution window.
                    padding='same',
                    use_bias=True,
                    input_shape=(16, 16, 4320)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer1 = Sequential(
            [
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(32, 32, 512+2160)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer2 = Sequential(
            [
                layers.Conv2D(
                    filters=270,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(64, 64, 1080+256)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer3 = Sequential(
            [
                layers.Conv2D(
                    filters=96,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(128, 128, 540)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer4 = Sequential(
            [
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 192)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer5 = Sequential(
            [
                layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 128)
                ),
                layers.ReLU(),
                layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 64)
                ),
                layers.ReLU(),
                layers.Conv2D(
                    filters=time_slices,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 32)
                ),
                layers.Activation('sigmoid'),
            ]
        )
    
    def create_model_weights(self, model, train_enc):
        state_dict = torch.load(self.path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module'  in k:
                k = 'module.pnas.' + k
            else:
                k = k.replace('pnas.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        
        for param in model.variables:
            param.trainable = train_enc
        
        dummy_input = torch.randn(1, 3, 255, 255)#.cuda()  # Modify this as needed

        torch.onnx.export(model.module, dummy_input, "PnasVolModelslast_weights/model_weights.onnx", export_params=True)  
    
    def call(self, images):
        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i == 3 :
                out3 = s1
            if i == 7:
                out4 = s1
            if i == 11:
                out5 = s1
        
        x = tf.concat([out5, out4], axis=1)
        x = self.deconv_layer1(x)

        x = tf.concat([x, out3], axis=1)
        x = self.deconv_layer2(x)

        x = tf.concat([x, out2], axis=1)
        x = self.deconv_layer3(x)
        x = tf.concat([x, out1], axis=1)

        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)
        x = x / x.max()

        return x, [out1, out2, out3, out4, out5]


class PNASBoostedModelMultiLevel(tf.keras.Model):

    def __init__(self, device, models_path, model_vol_path, time_slices, train_model=False, selected_slices=""):
        super(PNASBoostedModelMultiLevel, self).__init__()

        self.selected_slices = selected_slices
        self.linear_upsampling = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.deconv_layer1 = Sequential(
            [
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(16, 16, 512+2160+6),
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer2 = Sequential(
            [
                layers.Conv2D(
                    filters=270,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(32, 32, 1080+256+6)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer3 = Sequential(
            [
                layers.Conv2D(
                    filters=270,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(64, 64, 540+6)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_layer4 = Sequential(
            [
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(128, 128, 192+6)
                ),
                layers.ReLU(),
                self.linear_upsampling,
            ]
        )

        self.deconv_mix = Sequential(
            [
                layers.Conv2D(
                    filters=16,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 128 + 6)
                ),
                layers.ReLU(),
                layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 16)
                ),
                layers.ReLU(),
                layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True,
                    input_shape=(256, 256, 32)
                ),
                layers.Activation('sigmoid'),
            ]
        )

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model_vol = PNASVolModellast(time_slices=5, load_weight=0)
            model_sal = PNASModel(load_weight=0)
            is_succeed = self.pytorch_to_onnx(model_vol, model_sal, "")

            if is_succeed:
                self.pnas_vol = tf.saved_model.load("PNASBoostedModellast_weights/vol_weights/model_vol.onnx")
                self.pnas_sal = tf.saved_model.load("PNASBoostedModellast_weights/sal_weights/model_sal.onnx")
            else:
                raise Exception("Weights couldn't load")

    def pytorch_to_onnx(self, model_vol, model_sal, input_shape):
        model_path = "./checkpoints/multilevel_tempsal.pt"
        # Assuming your model structure and state_dict loading as defined in your code
        # model_vol = PNASVolModellast(time_slices=5, load_weight=0)  # Modify this for time slices
        # model_vol = torch.nn.DataParallel(model_vol)#.cuda()

        # Load the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        vol_state_dict = OrderedDict()
        sal_state_dict = OrderedDict()
        smm_state_dict = OrderedDict()

        # Populate the OrderedDicts with model weights
        for k, v in state_dict.items():
            if 'pnas_vol' in k:
                k = k.replace('pnas_vol.module.', '')
                vol_state_dict[k] = v
            elif 'pnas_sal' in k:
                k = k.replace('pnas_sal.module.', '')
                sal_state_dict[k] = v
            else:
                smm_state_dict[k] = v

        # Load the state dicts into the respective models
        model_vol.load_state_dict(vol_state_dict)
        # model_sal = PNASModel(load_weight=0)
        # model_sal = torch.nn.DataParallel(model_sal)#.cuda()
        model_sal.load_state_dict(sal_state_dict, strict=True)

        # Set parameters to not require gradients
        for param in model_vol.parameters():
            param.requires_grad = False
        for param in model_sal.parameters():
            param.requires_grad = False

        # Define dummy input based on your model's input shape
        dummy_input = torch.randn(1, 3, 255, 255)#.cuda()  # Modify this as needed

        # Export the PyTorch model to ONNX
        try:
            torch.onnx.export(model_vol.module, dummy_input, "PNASBoostedModellast_weights/vol_weights/model_vol.onnx", export_params=True)
            torch.onnx.export(model_sal.module, dummy_input, "PNASBoostedModellast_weights/sal_weights/model_sal.onnx", export_params=True)
            return True
        except: 
            return False

    def call(self, images):
        
        # pnas_pred = self.pnas_sal(images).unsqueeze(1)
        pnas_pred = tf.expand_dims(self.pnas_sal, axis=1)
        pnas_vol_pred, outs = self.pnas_vol(images)

        out1, out2, out3, out4, out5 = outs
        x_maps = tf.concat([pnas_pred, pnas_vol_pred], axis=1)

        x = tf.concat([out5, out4], axis=1)
        x_maps16 = tf.image_resize(x_maps, size=(16, 16), method='bicubic')

        x = tf.concat([x, x_maps16], axis=1)

        x = self.deconv_layer1(x)
        x = tf.concat([x, out3], axis=1)
        x_maps32 = tf.image_resize(x_maps, size=(32, 32), method='bicubic')
        x = tf.concat([x, x_maps32], axis=1)

        x = self.deconv_layer2(x)
        x = tf.concat([x, out2], axis=1)
        x_maps64 = tf.image_resize(x_maps, size=(64, 64), method='bicubic')
        x = tf.concat([x, x_maps64], axis=1)

        x = self.deconv_layer3(x)
        x = tf.concat([x, out1], axis=1)
        x_maps128 = tf.image_resize(x_maps, size=(128, 128), method='bicubic')
        x = tf.concat([x, x_maps128], axis=1)

        x = self.deconv_layer4(x)
        x = tf.concat([x, x_maps], axis=1)

        x = self.deconv_mix(x)

        x = tf.squeeze(x, axis=1)

        return x, pnas_vol_pred