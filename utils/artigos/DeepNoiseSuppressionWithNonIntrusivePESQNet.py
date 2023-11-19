from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Concatenate, Bidirectional, LSTM, TimeDistributed, Reshape, BatchNormalization, ConvLSTM1D, UpSampling2D, Multiply, Add
from tensorflow.keras.backend import sigmoid

# Define a class for a deep noise suppression model with an integrated non-intrusive PESQNet.
class DeepNoiseSuppressionWithNonIntrusivePESQNet(object):

    # Initialization of class variables: input shape, filter count, kernel size, and output channels.
    input_shape = None
    F = 64  # Number of filters
    N = 5   # Kernel size
    Cout = 2  # Number of output channels

    PESQ_model = None  # Placeholder for PESQNet model
    DNS_model = None  # Placeholder for DNS model

    # Constructor to initialize the models with specified parameters.
    def __init__(self, input_shape, F=64, N=5, channels=2):
        self.input_shape = input_shape  # Define input shape for the network
        self.F = F  # Set the filter count for the convolutional layers
        self.N = N  # Set the kernel size for the convolutional layers
        self.Cout = channels  # Set the output channels count

        # Build models for PESQ estimation and DNS respectively.
        # self.PESQ_model = self.PESQNet()  
        # self.DNS_model = self.DNS()

    # Custom activation function to scale the sigmoid activation for PESQ score estimation.
    def custom_activation(self, x):
        # The custom activation function is defined to adjust the sigmoid output to a specific scale and offset.
        return 3.6 * sigmoid(x) + 1.04

    # Define the PESQNet model architecture for non-intrusive PESQ estimation.
    def PESQNet(self, filters=32, inputs=None, return_layers=False):
        # Input layer for feeding in the signal data.

        if inputs == None:
            input_layer = Input(shape=self.input_shape)
        else:
            input_layer = inputs

        # Convolutional layers with max pooling for feature extraction.
        conv1 = Conv2D(filters=filters, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
        maxpool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(maxpool1)
        maxpool2 = MaxPooling2D((2, 1))(conv2)

        # Encoder CNN - assuming the use of multiple widths for convolutional kernels
        cnn_branches = []
        for w in [5, 3, 1]:  # Example widths
            conv = Conv2D(filters=filters, kernel_size=(w, w), activation='relu', padding='same')(maxpool2)
            maxpool = MaxPooling2D((2, 2))(conv)
            cnn_branches.append(maxpool)

        # Concatenate all CNN branches to integrate features extracted by different kernel sizes.
        cnn_output = Concatenate()(cnn_branches)

        # Apply a dense layer to each timestep after flattening CNN output with TimeDistributed wrapper.
        time_distributed_output = TimeDistributed(Dense(128, activation='relu'))(cnn_output)
        
        # Reshape output for LSTM processing.
        reshape_to_lstm = Reshape((-1, 16 * 128))(time_distributed_output)
        
        # LSTM layers for temporal sequence processing.
        blstm = Bidirectional(LSTM(128, return_sequences=True))(reshape_to_lstm)

        # Fully connected layers for output generation with custom activation function.
        fc1 = Dense(128, activation='relu')(blstm)
        fc2 = Dense(32, activation='relu')(fc1)
        output_layer = Dense(1, activation=self.custom_activation)(fc2)

        # Model compilation with input and output specifications.
        if not return_layers:
            return Model(inputs=input_layer, outputs=output_layer)
        else:
            return output_layer
    
    # Define the DNS model for noise suppression.
    def DNSModel(self, inputs=None):
        # Input layer for receiving noisy speech signals.
        cache_inputs = inputs
        if inputs == None:
            inputs = Input(shape=self.input_shape)
        
        # Batch normalization to standardize input features.
        normalized = BatchNormalization()(inputs)
        
        # Encoder with convolutional layers and max pooling.
        conv1 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(normalized)
        conv2 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D((2, 1))(conv2)
        
        conv3 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(pool1)
        conv4 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(conv3)
        pool2 = MaxPooling2D((2, 1))(conv4)
        
        # ConvLSTM layer for capturing temporal dependencies in the encoded feature space.
        convlstm = ConvLSTM1D(self.F, self.N, padding='same', return_sequences=True)(pool2)
        
        # Decoder with upsampling and skip connections to restore spatial resolution and add detail.
        up1 = UpSampling2D((2, 1))(convlstm)
        conv5 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(up1)
        conv6 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(conv5)
        
        # Skip connections from encoder to decoder to help with feature reconstruction.
        skip1 = Concatenate()([conv6, conv4])
        up2 = UpSampling2D((2, 1))(skip1)
        
        # Final convolutional layers before output generation.
        conv7 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(up2)
        conv8 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(conv7)
        skip2 = Concatenate()([conv8, conv2])
        
        # Output layer with linear activation to produce the final enhanced signal.
        mask = Conv2D(self.Cout, (self.N, self.N), padding='same', activation='linear')(skip2)

        outputs = Multiply()([inputs, mask])
        
        # Construct and return the DNS model.
        if cache_inputs == None:
            return Model(inputs=inputs, outputs=outputs)
        else:
            return outputs
        
    
    # Define the DNS model for noise suppression.
    def DNSModelDouble(self, inputs_module, input_phase):
        # Input layer for receiving noisy speech signals.
        
        # Batch normalization to standardize input features.
        normalized_module = BatchNormalization()(inputs_module)
        normalized_phase = BatchNormalization()(input_phase)

        normalized = Concatenate()([normalized_module, normalized_phase])
        
        # Encoder with convolutional layers and max pooling.
        conv1 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(normalized)
        conv2 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv2)
        
        conv3 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(pool1)
        conv4 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(conv3)
        pool2 = MaxPooling2D((2, 2))(conv4)
        
        # ConvLSTM layer for capturing temporal dependencies in the encoded feature space.
        convlstm = ConvLSTM1D(self.F, self.N, padding='same', return_sequences=True)(pool2)
        
        # Decoder with upsampling and skip connections to restore spatial resolution and add detail.
        up1 = UpSampling2D((2, 2))(convlstm)
        conv5 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(up1)
        conv6 = Conv2D(2 * self.F, (self.N, self.N), padding='same', activation='relu')(conv5)
        
        # Skip connections from encoder to decoder to help with feature reconstruction.
        skip1 = Concatenate()([conv6, conv4])
        up2 = UpSampling2D((2, 2))(skip1)
        
        # Final convolutional layers before output generation.
        conv7 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(up2)
        conv8 = Conv2D(self.F, (self.N, self.N), padding='same', activation='relu')(conv7)
        skip2 = Concatenate()([conv8, conv2])
        
        # Output layer with linear activation to produce the final enhanced signal.
        output_module = Conv2D(1, (self.N, self.N), padding='same', activation='linear')(skip2)
        output_phase  = Conv2D(1, (self.N, self.N), padding='same', activation='tanh')(skip2)

        output_module = Add()([inputs_module, output_module])
        output_phase = Add()([input_phase, output_phase])

        return output_module, output_phase