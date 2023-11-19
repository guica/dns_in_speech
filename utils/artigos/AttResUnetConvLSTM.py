import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Concatenate, Multiply, Add, Activation

class ExpendAsLayer(Layer):
    '''
    Exemplo de uso:
        layer = ExpendAsLayer(rep=3)
        output = layer(input_tensor)
    '''

    def __init__(self, rep, **kwargs):
        super(ExpendAsLayer, self).__init__(**kwargs)
        self.rep = rep

    def call(self, inputs):
        return K.repeat_elements(inputs, self.rep, axis=3)

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (input_shape[3] * self.rep,)

    def get_config(self):
        config = super(ExpendAsLayer, self).get_config()
        config.update({'rep': self.rep})
        return config


class DoubleConvLayer(Layer):
    '''
    Exemplo de uso:
        double_conv_layer = DoubleConvLayer(filter_size=3, size=64, dropout=0.5, batch_norm=True)
        output = double_conv_layer(input_tensor)  # Onde input_tensor é o tensor de entrada

    '''
    def __init__(self, filter_size=3, size=32, dropout=0., batch_norm=False, **kwargs):
        super(DoubleConvLayer, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.size = size
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.conv1 = layers.Conv2D(size, (filter_size, filter_size), padding='same')
        self.conv2 = layers.Conv2D(size, (filter_size, filter_size), padding='same')
        self.shortcut_conv = layers.Conv2D(size, kernel_size=(1, 1), padding='same')

        if self.batch_norm:
            self.bn1 = layers.BatchNormalization(axis=3)
            self.bn2 = layers.BatchNormalization(axis=3)
            self.shortcut_bn = layers.BatchNormalization(axis=3)

        if self.dropout > 0:
            self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs):
        conv = self.conv1(inputs)
        if self.batch_norm:
            conv = self.bn1(conv)
        conv = layers.Activation('relu')(conv)

        conv = self.conv2(conv)
        if self.batch_norm:
            conv = self.bn2(conv)
        conv = layers.Activation('relu')(conv)

        if self.dropout > 0:
            conv = self.dropout_layer(conv)

        shortcut = self.shortcut_conv(inputs)
        if self.batch_norm:
            shortcut = self.shortcut_bn(shortcut)

        res_path = Add()([shortcut, conv])
        return res_path

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.size,)

    def get_config(self):
        config = super(DoubleConvLayer, self).get_config()
        config.update({
            'filter_size': self.filter_size,
            'size': self.size,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm
        })
        return config

class GatingSignalLayer(Layer):
    '''
    Exemplo de uso:
        gating_signal_layer = GatingSignalLayer(out_size=out_size)
        output = gating_signal_layer(x) 
    '''

    def __init__(self, out_size, batch_norm=False, **kwargs):
        super(GatingSignalLayer, self).__init__(**kwargs)
        self.out_size = out_size
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(out_size, (1, 1), padding='same')
        if self.batch_norm:
            self.batch_normalization = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_normalization(x)
        x = layers.Activation('relu')(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.out_size,)

    def get_config(self):
        config = super(GatingSignalLayer, self).get_config()
        config.update({
            'out_size': self.out_size,
            'batch_norm': self.batch_norm
        })
        return config

class AttentionBlock(Layer):
    '''
    Exemplo de uso:
        attention_layer = AttentionBlock(inter_shape=inter_shape)
        output = attention_layer([x, gating]) 
    '''

    def __init__(self, inter_shape, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.inter_shape = inter_shape

    def build(self, input_shape):
        # Definindo as camadas internas
        self.theta_x_conv = layers.Conv2D(self.inter_shape, (2, 2), strides=(2, 2), padding='same')
        self.phi_g_conv = layers.Conv2D(self.inter_shape, (1, 1), padding='same')
        self.psi_conv = layers.Conv2D(1, (1, 1), padding='same')
        self.final_conv = layers.Conv2D(input_shape[0][-1], (1, 1), padding='same')  # input_shape[0] é x
        self.batch_norm = layers.BatchNormalization()
        self.upsampleg = layers.Conv2DTranspose(self.inter_shape, (3, 3),
                                            # strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                            strides=(1, 1),
                                            padding='same')
        
        self.upsamplepsi = layers.UpSampling2D(size=(2, 2))

        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs):
        x, gating = inputs
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(gating)

        theta_x = self.theta_x_conv(x)
        shape_theta_x = K.int_shape(theta_x)

        phi_g = self.phi_g_conv(gating)

        # print(f'inter_shape: {self.inter_shape}')
        # print(f'strides[0]: {shape_theta_x[1] // shape_g[1]}')
        # print(f'strides[1]: {shape_theta_x[2] // shape_g[2]}')
        # print(f'phi_g: {phi_g}')

        # upsample_g = layers.Conv2DTranspose(self.inter_shape, (3, 3),
        #                                     # strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
        #                                     strides=(1, 1),
        #                                     padding='same')(phi_g)
        upsample_g = self.upsampleg(phi_g)

        concat_xg = Add()([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = self.psi_conv(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)

        # print(f'shape_x[1] // shape_sigmoid[1]: {shape_x[1] // shape_sigmoid[1]}')
        # print(f'shape_x[2] // shape_sigmoid[2]: {shape_x[2] // shape_sigmoid[2]}')

        # upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
        upsample_psi = self.upsamplepsi(sigmoid_xg)

        # Utilizando a classe ExpendAsLayer
        expend_as_layer = ExpendAsLayer(rep=shape_x[3])
        upsample_psi = expend_as_layer(upsample_psi)

        y = Multiply()([upsample_psi, x])

        result = self.final_conv(y)
        result_bn = self.batch_norm(result)
        return result_bn

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({'inter_shape': self.inter_shape})
        return config


class AttResUnetConvLSTM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.FILTER_SIZE = 3
        self.FILTER_NUM = 32
        self.UP_SAMP_SIZE = 2
        dropout_rate = 0.
        batch_norm = False
        axis = -1

        self.conv_128 = DoubleConvLayer(self.FILTER_SIZE, self.FILTER_NUM, dropout_rate, batch_norm)
        self.pool_64 = layers.MaxPooling2D(pool_size=(2,2))
        # DownRes 2
        self.conv_64 = DoubleConvLayer(self.FILTER_SIZE, 2*self.FILTER_NUM, dropout_rate, batch_norm)
        self.pool_32 = layers.MaxPooling2D(pool_size=(2,2))
        # DownRes 3
        self.conv_32 = DoubleConvLayer(self.FILTER_SIZE, 4*self.FILTER_NUM, dropout_rate, batch_norm)
        self.pool_16 = layers.MaxPooling2D(pool_size=(2,2))
        # DownRes 4
        self.conv_16 = DoubleConvLayer(self.FILTER_SIZE, 8*self.FILTER_NUM, dropout_rate, batch_norm)
        self.pool_8 = layers.MaxPooling2D(pool_size=(2,2))
        # DownRes 5, convolution only
        self.conv_8 = DoubleConvLayer(self.FILTER_SIZE, 16*self.FILTER_NUM, dropout_rate, batch_norm)

        # ConvLSTM layer for capturing temporal dependencies in the encoded feature space.
        # self.convlstm = layers.ConvLSTM1D(128, 64, padding='same', return_sequences=True)


        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        self.gating_16 = GatingSignalLayer(8*self.FILTER_NUM, batch_norm=batch_norm)
        self.att_16 = AttentionBlock(8*self.FILTER_NUM)
        self.up_16 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")
        # up_16 = layers.concatenate([up_16, att_16], axis=axis)
        self.up_conv_16 = DoubleConvLayer(self.FILTER_SIZE, 8*self.FILTER_NUM, dropout_rate, batch_norm)

        # UpRes 7
        self.gating_32 = GatingSignalLayer(4*self.FILTER_NUM, batch_norm=batch_norm)
        self.att_32 = AttentionBlock(4*self.FILTER_NUM)
        self.up_32 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")
        # up_32 = layers.concatenate([up_32, att_32], axis=axis)
        self.up_conv_32 = DoubleConvLayer(self.FILTER_SIZE, 4*self.FILTER_NUM, dropout_rate, batch_norm)

        # UpRes 8
        self.gating_64 = GatingSignalLayer(2*self.FILTER_NUM, batch_norm=batch_norm)
        self.att_64 = AttentionBlock(2*self.FILTER_NUM)
        self.up_64 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")
        # up_64 = layers.concatenate([up_64, att_64], axis=axis)
        self.up_conv_64 = DoubleConvLayer(self.FILTER_SIZE, 2*self.FILTER_NUM, dropout_rate, batch_norm)

        # UpRes 9
        self.gating_128 = GatingSignalLayer(self.FILTER_NUM, batch_norm=batch_norm)
        self.att_128 = AttentionBlock(self.FILTER_NUM)
        self.up_128 = layers.UpSampling2D(size=(self.UP_SAMP_SIZE, self.UP_SAMP_SIZE), data_format="channels_last")
        # up_128 = layers.concatenate([up_128, att_128], axis=axis)
        self.up_conv_128 = DoubleConvLayer(self.FILTER_SIZE, self.FILTER_NUM, dropout_rate, batch_norm)

        # finais
        self.conv_final = layers.Conv2D(2, kernel_size=(1,1))
        self.batchnorm_final = layers.BatchNormalization(axis=axis)

        self.actv_final = layers.Activation('relu')

    def call(self, input):
        
        conv_128 = self.conv_128(input)
        pool_64 = self.pool_64(conv_128)

        conv_64 = self.conv_64(pool_64)
        pool_32 = self.pool_32(conv_64)

        conv_32 = self.conv_32(pool_32)
        pool_16 = self.pool_16(conv_32)

        conv_16 = self.conv_16(pool_16)
        pool_8 = self.pool_8(conv_16)

        conv_8 = self.conv_8(pool_8)

        # convlstm = self.convlstm(conv_8)

        gating_16 = self.gating_16(conv_8)
        att_16 = self.att_16([conv_16, gating_16])
        up_16 = self.up_16(conv_8)
        up_16 = Concatenate()([up_16, att_16])
        up_conv_16 = self.up_conv_16(up_16)

        gating_32 = self.gating_32(up_conv_16)
        att_32 = self.att_32([conv_32, gating_32])
        up_32 = self.up_32(up_conv_16)
        up_32 = Concatenate()([up_32, att_32])
        up_conv_32 = self.up_conv_32(up_32)

        gating_64 = self.gating_64(up_conv_32)
        att_64 = self.att_64([conv_64, gating_64])
        up_64 = self.up_64(up_conv_32)
        up_64 = Concatenate()([up_64, att_64])
        up_conv_64 = self.up_conv_64(up_64)

        gating_128 = self.gating_128(up_conv_64)
        att_128 = self.att_128([conv_128, gating_128])
        up_128 = self.up_128(up_conv_64)
        up_128 = Concatenate()([up_128, att_128])
        up_conv_128 = self.up_conv_128(up_128)

        conv_final = self.conv_final(up_conv_128)
        bn_final = self.batchnorm_final(conv_final)
        actv = self.actv_final(bn_final)

        return actv