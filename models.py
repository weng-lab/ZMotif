import tensorflow as tf
from keras.models import Model
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Dense, Lambda, maximum, GaussianNoise, concatenate, BatchNormalization, SpatialDropout1D
from keras import regularizers
from keras import initializers 
from keras.constraints import non_neg
from keras import backend as K
from keras.initializers import RandomUniform
from keras.optimizers import Adadelta

def construct_model(num_kernels=16, kernel_width=20, seq_len=None, dropout_prop=0.0, kernel_initializer=initializers.RandomUniform(minval=-0.00001, maxval=0.00001, seed=12), optimizer='adam', activation='linear', num_classes=1, l1_reg=0.0000001, gaussian_noise = 0.0, spatial_dropout = 0.0):
    seq_input = Input(shape=(seq_len,4))
    noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    noisy_seq_rc = rc_op(noisy_seq)
    shared_conv = Conv1D(num_kernels, kernel_width, strides=1, padding='same', activation=activation, use_bias=False, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1(l1_reg), bias_initializer='zeros')
    conv_for = shared_conv(noisy_seq)
    conv_rc = shared_conv(noisy_seq_rc)
    merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    output = Dense(num_classes, activation='sigmoid', use_bias=False, kernel_initializer=initializers.Ones(), kernel_constraint=non_neg(), bias_initializer='zeros')(pooled)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# def construct_model_tf(num_kernels=16, kernel_width=20, seq_len=None, dropout_prop=0.0, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001, seed=12), optimizer=tf.contrib.opt.AdamWOptimizer(weight_decay=0.1), activation='linear', num_classes=1, l1_reg=0.0000001, gaussian_noise = 0.0, spatial_dropout = 0.0):
#     seq_input = tf.keras.layers.Input(shape=(seq_len,4))
#     #noisy_seq = tf.keras.layers.GaussianNoise(gaussian_noise)(seq_input)
#     rc_op = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reverse(x,axes=(1,2)))
#     seq_rc = rc_op(seq_input)
#     shared_conv = tf.keras.layers.Conv1D(num_kernels, kernel_width, strides=1, padding='same', activation=activation, use_bias=False, kernel_initializer=kernel_initializer, kernel_regularizer=tf.keras.regularizers.l1(l1_reg), bias_initializer='zeros')
#     conv_for = shared_conv(seq_input)
#     conv_rc = shared_conv(seq_rc)
#     merged = tf.keras.layers.maximum([conv_for, conv_rc])
#     pooled = tf.keras.layers.GlobalMaxPooling1D()(merged)
#     output = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False, kernel_initializer=tf.keras.initializers.Ones(), kernel_constraint=non_neg(), bias_initializer='zeros')(pooled)
#     model = tf.keras.Model(inputs=seq_input, outputs=output)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     print(model.summary())
#     return model
    
    
    

    
    
def construct_scan_model(conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    seq = Input(shape=(None,4))
    conv = Conv1D(num_kernels, kernel_width, 
                  name = 'scan_conv',
                  strides=1, 
                  padding='valid', 
                  activation='linear', 
                  use_bias=False, 
                  kernel_initializer='zeros', 
                  bias_initializer='zeros',
                  trainable=False)
    
    conv_seq = conv(seq)
    
    
    model = Model(inputs=seq, outputs=conv_seq)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer('scan_conv').set_weights([conv_weights])
    print(model.summary())
    return model

def construct_fnn(num_kernels = 32,
                         kernel_width = 32,
                         num_dense = 32,
                         conv_layer_name = 'fnn_convolution'):
    seq = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq)
    shared_conv = Conv1D(num_kernels, kernel_width, 
                         strides=1, 
                         padding='same', 
                         activation='linear', 
                         use_bias=False, 
                         kernel_initializer='zeros',  
                         bias_initializer='zeros',
                         name = conv_layer_name,
                         trainable=False)
    conv_for = shared_conv(seq)
    conv_rc = shared_conv(seq_rc)
#     rev_op = Lambda(lambda x: K.reverse(x,axes=(1)))
#     conv_rc_rev = rev_op(conv_rc)
    merged = maximum([conv_for, conv_rc])
#    batch_norm = BatchNormalization()(merged)
    pooled = GlobalMaxPooling1D()(merged)
    dense = Dense(num_dense)(pooled)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=seq, outputs=output)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    print(model.summary())
    return model
    
    
