import tensorflow as tf
from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import ConvLSTM1D, MultiHeadAttention, LayerNormalization, Dense


class AttentionResUnet(object):
    '''
    Hyper-parameters
    '''
    # input data
    INPUT_SIZE = (128, 64)
    INPUT_CHANNEL = 2   # 1-grayscale, 3-RGB scale
    OUTPUT_MASK_CHANNEL = 2
    
    # network structure
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    DOWN_SAMP_SIZE = 2 # size of pooling filters
    UP_SAMP_SIZE = 2 # size of upsampling filters

    def __init__(self, input_shape):
        self.INPUT_SIZE = (input_shape[0], input_shape[1])
        self.INPUT_CHANNEL = input_shape[2]
        self.OUTPUT_MASK_CHANNEL = self.INPUT_CHANNEL

    def expend_as(self, tensor, rep):
        return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                            arguments={'repnum': rep})(tensor)

    def double_conv_layer(self, x, filter_size, size, dropout, batch_norm=False):
        '''
        construction of a double convolutional layer using
        SAME padding
        RELU nonlinear activation function
        :param x: input
        :param filter_size: size of convolutional filter
        :param size: number of filters
        :param dropout: FLAG & RATE of dropout.
                if < 0 dropout cancelled, if > 0 set as the rate
        :param batch_norm: flag of if batch_norm used,
                if True batch normalization
        :return: output of a double convolutional layer
        '''

        axis = 3
        conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=axis)(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=axis)(conv)
        conv = layers.Activation('relu')(conv)
        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=axis)(shortcut)

        res_path = layers.add([shortcut, conv])
        return res_path

    def gating_signal(self, input, out_size, batch_norm=False):
        """
        resize the down layer feature map into the same dimension as the up layer feature map
        using 1x1 conv
        :param input:   down-dim feature map
        :param out_size:output channel number
        :return: the gating feature map with the same dimension of the up layer feature map
        """

        x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def attention_block(self, x, gating, inter_shape):
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(gating)

        theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)

        phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
        upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                    strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                    padding='same')(phi_g)  # 16

        concat_xg = layers.add([upsample_g, theta_x])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

        upsample_psi = self.expend_as(upsample_psi, shape_x[3])

        y = layers.multiply([upsample_psi, x])

        result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn


    def get_model(self, dropout_rate=0.1, batch_norm=True):
        '''
        Rsidual UNet construction, with attention gate
        convolution: 3*3 SAME padding
        pooling: 2*2 VALID padding
        upsampling: 3*3 VALID padding
        final convolution: 1*1
        :param dropout_rate: FLAG & RATE of dropout.
                if < 0 dropout cancelled, if > 0 set as the rate
        :param batch_norm: flag of if batch_norm used,
                if True batch normalization
        :return: model
        '''
        # input data
        # dimension of the image depth
        inputs = layers.Input((self.INPUT_SIZE[0], self.INPUT_SIZE[1], self.INPUT_CHANNEL), dtype=tf.float32)
        axis = 3

        # Downsampling layers
        # DownRes 1, double residual convolution + pooling
        conv_128 = self.double_conv_layer(inputs, self.FILTER_SIZE, self.FILTER_NUM, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        conv_64 = self.double_conv_layer(pool_64, self.FILTER_SIZE, 2*self.FILTER_NUM, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = self.double_conv_layer(pool_32, self.FILTER_SIZE, 4*self.FILTER_NUM, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = self.double_conv_layer(pool_16, self.FILTER_SIZE, 8*self.FILTER_NUM, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = self.double_conv_layer(pool_8, self.FILTER_SIZE, 16*self.FILTER_NUM, dropout_rate, batch_norm)
        
        # ConvLSTM layer for capturing temporal dependencies in the encoded feature space.
        # convlstm = ConvLSTM1D(128, 64, padding='same', return_sequences=True)(conv_8)

        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = self.gating_signal(conv_8, 8*self.FILTER_NUM, batch_norm)
        att_16 = self.attention_block(conv_16, gating_16, 8*self.FILTER_NUM)
        up_16 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, att_16], axis=axis)
        up_conv_16 = self.double_conv_layer(up_16, self.FILTER_SIZE, 8*self.FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = self.gating_signal(up_conv_16, 4*self.FILTER_NUM, batch_norm)
        att_32 = self.attention_block(conv_32, gating_32, 4*self.FILTER_NUM)
        up_32 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=axis)
        up_conv_32 = self.double_conv_layer(up_32, self.FILTER_SIZE, 4*self.FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 8
        gating_64 = self.gating_signal(up_conv_32, 2*self.FILTER_NUM, batch_norm)
        att_64 = self.attention_block(conv_64, gating_64, 2*self.FILTER_NUM)
        up_64 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64], axis=axis)
        up_conv_64 = self.double_conv_layer(up_64, self.FILTER_SIZE, 2*self.FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = self.gating_signal(up_conv_64, self.FILTER_NUM, batch_norm)
        att_128 = self.attention_block(conv_128, gating_128, self.FILTER_NUM)
        up_128 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
        up_128 = layers.concatenate([up_128, att_128], axis=axis)
        up_conv_128 = self.double_conv_layer(up_128, self.FILTER_SIZE, self.FILTER_NUM, dropout_rate, batch_norm)

        # 1*1 convolutional layers
        # valid padding
        # batch normalization
        # sigmoid nonlinear activation
        conv_final = layers.Conv2D(self.INPUT_CHANNEL, kernel_size=(1,1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=axis)(conv_final)

        conv_final = layers.Activation('relu')(conv_final)

        # conv_final2 = layers.Conv2D(1, kernel_size=(1,1))(up_conv_128)
        # conv_final2 = layers.BatchNormalization(axis=axis)(conv_final2)
        # conv_final2 = layers.Activation('relu')(conv_final2)

        # final = layers.concatenate([conv_final, conv_final2], axis=-1)

        # Model integration
        model = models.Model(inputs, conv_final, name="AttentionResUNet")
        return model
