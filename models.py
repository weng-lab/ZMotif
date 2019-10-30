from keras.layers import Input, Lambda, Conv1D, GaussianNoise, maximum, GlobalMaxPooling1D, Dense
from keras import initializers
from keras.constraints import non_neg
from keras import regularizers
from keras import backend as K
from keras.models import Model

def construct_model(num_kernels=1, kernel_width=20, seq_len=None, dropout_prop=0.0, kernel_initializer=initializers.RandomUniform(minval=-.01, maxval=0.01, seed=12), optimizer='adam', activation='linear', num_classes=1, l1_reg=0.0000001, gaussian_noise = 0.0, spatial_dropout = 0.0):
    seq_input = Input(shape=(seq_len,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq_input)
    if gaussian_noise > 0:
        noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
        noisy_seq_rc = rc_op(noisy_seq)
    
    shared_conv = Conv1D(num_kernels, kernel_width, strides=1, padding='same', activation=activation, use_bias=False, kernel_initializer="orthogonal", kernel_regularizer=regularizers.l1(l1_reg), bias_initializer='zeros')
    
    if gaussian_noise > 0:
        conv_for = shared_conv(noisy_seq)
        conv_rc = shared_conv(noisy_seq_rc)
    else:
        conv_for = shared_conv(seq_input)
        conv_rc = shared_conv(seq_rc)
    
    if spatial_dropout > 0:
        sp_drp = SpatialDropout1D(spatial_dropout)
        sp_drp_for = sp_drp(conv_for)
        sp_drp_rc = sp_drp(conv_rc)
        merged = maximum([sp_drp_for, sp_drp_rc])
    else:
        merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    output = Dense(num_classes, activation='sigmoid', use_bias=False, kernel_initializer=initializers.Ones(), kernel_constraint=non_neg(), bias_initializer='zeros')(pooled)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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
    return model

def construct_fnn(num_kernels = 32,
                         kernel_width = 32,
                         num_dense = 32,
                         conv_layer_name = 'fnn_convolution',
                         optimizer='adam'):
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
                         trainable=True)
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
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    print(model.summary())
    return model
    
def construct_rnn(num_kernels = 32,
                         kernel_width = 32,
                         conv_layer_name = 'rnn_convolution'):
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
                         trainable=True)
    conv_for = shared_conv(seq)
    conv_rc = shared_conv(seq_rc)
    rev_op = Lambda(lambda x: K.reverse(x,axes=(1)))
    conv_rc_rev = rev_op(conv_rc)
    merged = maximum([conv_for, conv_rc_rev])
#    batch_norm = BatchNormalization()(merged)
    pooled = MaxPooling1D(10)(merged)
    lstm = LSTM(32)(pooled)
    output = Dense(1, activation='sigmoid')(lstm)
    model = Model(inputs=seq, outputs=output)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    print(model.summary())
    return model    
