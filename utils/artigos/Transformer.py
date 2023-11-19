from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras import Input, Model
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)

# Agora, vamos construir as camadas de encoder e decoder usando a camada de MultiHeadAttention
def encoder_layer(d_model, num_heads, dff, rate=0.1):
    inputs = Input(shape=(None, d_model))
    padding_mask = Input(shape=(1, 1, None))

    attention = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs, padding_mask)
    attention = Dropout(rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = Dense(dff, activation='relu')(attention)
    outputs = Dense(d_model)(outputs)
    outputs = Dropout(rate)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name="encoder_layer")

def decoder_layer(d_model, num_heads, dff, rate=0.1):
    inputs = Input(shape=(None, d_model))
    enc_outputs = Input(shape=(None, d_model))
    look_ahead_mask = Input(shape=(1, None, None))
    padding_mask = Input(shape=(1, 1, None))

    attention1 = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs, look_ahead_mask)
    attention1 = Dropout(rate)(attention1)
    attention1 = LayerNormalization(epsilon=1e-6)(inputs + attention1)

    attention2 = MultiHeadAttention(d_model, num_heads)(enc_outputs, enc_outputs, attention1, padding_mask)
    attention2 = Dropout(rate)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention1 + attention2)

    outputs = Dense(dff, activation='relu')(attention2)
    outputs = Dense(d_model)(outputs)
    outputs = Dropout(rate)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention2 + outputs)

    return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name="decoder_layer")


def getTransformerLayers(input_seq, target_seq, num_layers=2, d_model=32, num_heads=2, dff=512, max_seq_len=512, dropout_rate=0.1):
    
    # Função para criar uma única camada do Encoder
    def single_encoder_layer(d_model, num_heads, dff, rate):
        inputs = Input(shape=(None, d_model))
        attention = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs, None)
        attention = Dropout(rate)(attention)
        attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
    
        outputs = Dense(dff, activation='relu')(attention)
        outputs = Dense(d_model)(outputs)
        outputs = Dropout(rate)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)
    
        return Model(inputs=inputs, outputs=outputs)
    
    # Função para criar uma única camada do Decoder
    def single_decoder_layer(d_model, num_heads, dff, rate):
        inputs = Input(shape=(None, d_model))
        enc_outputs = Input(shape=(None, d_model))
    
        attention1 = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs, None)
        attention1 = Dropout(rate)(attention1)
        attention1 = LayerNormalization(epsilon=1e-6)(inputs + attention1)
    
        attention2 = MultiHeadAttention(d_model, num_heads)(enc_outputs, enc_outputs, attention1, None)
        attention2 = Dropout(rate)(attention2)
        attention2 = LayerNormalization(epsilon=1e-6)(attention1 + attention2)
    
        outputs = Dense(dff, activation='relu')(attention2)
        outputs = Dense(d_model)(outputs)
        outputs = Dropout(rate)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attention2 + outputs)
    
        return Model(inputs=[inputs, enc_outputs], outputs=outputs)
    
    # Criando as camadas de Encoder e Decoder
    encoder_layers = [single_encoder_layer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
    decoder_layers = [single_decoder_layer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
    
    # Definindo as entradas do modelo
    # input_seq = Input(shape=(max_seq_len, d_model))
    # target_seq = Input(shape=(max_seq_len, d_model))
    
    # Construindo o Encoder
    x = input_seq
    for encoder_layer in encoder_layers:
        x = encoder_layer(x)
    encoder_output = x
    
    # Construindo o Decoder
    y = target_seq
    for decoder_layer in decoder_layers:
        y = decoder_layer([y, encoder_output])
    decoder_output = y
    
    # Camada de saída
    final_output = Dense(d_model, activation='linear')(decoder_output)

    return final_output

# Criando o modelo Transformer
# transformer = Model(inputs=[input_seq, target_seq], outputs=final_output)

# Resumo do modelo
# transformer.summary()