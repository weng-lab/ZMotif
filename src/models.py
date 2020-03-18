from keras.layers import Input, Lambda, Conv1D, GaussianNoise, maximum, GlobalMaxPooling1D, Dense, concatenate, Activation, Add, BatchNormalization
from keras import initializers
from keras.constraints import non_neg
from keras import regularizers
from keras import backend as K
from keras.models import Model
import numpy as np
import json

def construct_model(num_kernels=1, kernel_width=20, seq_len=None, dropout_prop=0.0, kernel_initializer=initializers.RandomUniform(minval=-.01, maxval=0.01, seed=12), optimizer='adam', activation='linear', num_classes=1, l1_reg=0.0000001, gaussian_noise = 0.0, spatial_dropout = 0.0, rc = True, padding="same"):
    if rc:
        seq_input = Input(shape=(seq_len,4))
        rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
        seq_rc = rc_op(seq_input)
        if gaussian_noise > 0:
            noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
            noisy_seq_rc = rc_op(noisy_seq)
        print(activation)
        shared_conv = Conv1D(num_kernels, kernel_width,
                             strides=1, padding=padding, 
                             activation=activation,
                             use_bias=False,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizers.l1(l1_reg),
                             bias_initializer='zeros',
                             name="conv1d_1")

        if gaussian_noise > 0:
            conv_for = shared_conv(noisy_seq)
            conv_rc = shared_conv(noisy_seq_rc)
        else:
            conv_for = shared_conv(seq_input)
            conv_rc = shared_conv(seq_rc)

        merged = maximum([conv_for, conv_rc])
        pooled = GlobalMaxPooling1D()(merged)
        output = Dense(1, activation='sigmoid',
                       use_bias=False,
                       kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.01, seed=12), 
                       kernel_constraint=non_neg(), 
                       bias_initializer='zeros',
                       name="dense_1")(pooled)
        model = Model(inputs=seq_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    else:
        from keras.models import Sequential
        model = Sequential()
        model.add(Conv1D(num_kernels, kernel_width,
                         input_shape=(seq_len,4),
                             strides=1, padding='same', 
                             activation='linear',
                             use_bias=False,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizers.l1(l1_reg),
                             bias_initializer='zeros',
                             name="conv1d_1"))
        model.add(Activation('linear'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, use_bias=False,
                       kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.01, seed=12), 
                       kernel_constraint=non_neg(), 
                       bias_initializer='zeros',
                       name="dense_1"))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        


def construct_pretrain_model(num_kernels=1, kernel_width=20, conv_weights=None, dense_weights=None, gaussian_noise=0.0):
    seq_input = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq_input)
    
    if gaussian_noise > 0:
        noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
        noisy_seq_rc = rc_op(noisy_seq)
        
    if conv_weights is not None:
        fixed_conv = Conv1D(num_kernels-1, kernel_width, 
                                 strides=1, 
                                 padding='same', 
                                 activation="linear", 
                                 use_bias=False, 
                                 kernel_initializer="orthogonal", 
                                 bias_initializer='zeros',
                                 trainable=False,
                                 name="fixed_conv")
        if gaussian_noise > 0:
            fixed_conv_for = fixed_conv(noisy_seq)
            fixed_conv_rc = fixed_conv(noisy_seq_rc)
        else:
            fixed_conv_for = fixed_conv(seq_input)
            fixed_conv_rc = fixed_conv(seq_rc)

        fixed_merged = maximum([fixed_conv_for, fixed_conv_rc])
        fixed_pooled = GlobalMaxPooling1D()(fixed_merged)
        fixed_sum = Lambda(lambda x: K.sum(x,axis=1))(fixed_pooled)
    
    trained_conv = Conv1D(1, kernel_width, 
                             strides=1, 
                             padding='same', 
                             activation="linear", 
                             use_bias=False, 
                             kernel_initializer="orthogonal", 
                             bias_initializer='zeros',
                             trainable=True,
                             name="trained_conv")
        
    trained_conv_for = trained_conv(seq_input)
    trained_conv_rc = trained_conv(seq_rc)
    
    trained_merged = maximum([trained_conv_for, trained_conv_rc])
    trained_pooled = GlobalMaxPooling1D()(trained_merged)
    trained_dense = Dense(1, activation = "linear", use_bias=False,
                          kernel_constraint=non_neg(),
                          kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.1, seed=12),
                          name="trained_dense")(trained_pooled)
    
    if conv_weights is not None:
        total_sum = Add()([fixed_sum, trained_dense])
        output = Activation("sigmoid")(total_sum)
    else:
        output = Activation("sigmoid")(trained_dense)
        
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    
    if conv_weights is not None:
        pretrained_conv_weights = [dense_weights[i]*conv_weights[i] for i in range(len(conv_weights))]
        fixed_conv_weights = np.concatenate(pretrained_conv_weights, axis=2)
        model.get_layer("fixed_conv").set_weights([fixed_conv_weights])
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


import requests
def construct_lr(motif_file="hocomoco"):
    if motif_file == "hocomoco":
        url = "http://hocomoco11.autosome.ru/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pwms_HUMAN_mono.txt"
        data = requests.get(url).text
        with open("motifs.txt","w") as f:
            f.write(data)

        PWMS_LIST = []
        motif_ids = []
        with open("motifs.txt") as f:
            lines = [line.strip().split() for line in f.readlines()]

        pwm = []
        for i, line in enumerate(lines):
            if line[0][0] == ">":
                motif_id = line[0][1:]
                if i > 0:
                    pwm = np.reshape(np.array(pwm), (-1,4))
                    PWMS_LIST.append(pwm)
                    motif_ids.append(motif_id)
                pwm = []
            else:
                pwm.append([float(x) for x in line])
        max_w = np.max(np.array([pwm.shape[0] for pwm in PWMS_LIST]))
        k = len(PWMS_LIST)
        print(k)

        conv_weights = np.zeros((max_w,4,k))
        for i in range(k):
            pwm = PWMS_LIST[i]
            w = pwm.shape[0]
            max_val = np.sum(np.max(pwm, axis=1))
            start = (max_w - w) // 2
            stop = start + w
            conv_weights[start:stop,:,i] = pwm
    else:
        print("Motif file provided")
        with open(motif_file) as f:
            data = json.load(f)

        motifs = []
        for tf in data:
            motifs.append(np.array(data[tf]))

        max_w = np.max(np.array([motif.shape[0] for motif in motifs]))
        k = len(motifs)
        conv_weights = np.zeros((max_w,4,k))
        for i, motif in enumerate(motifs):
            w = motif.shape[0]
            w_2 = w // 2
            start = max_w // 2 - w_2
            stop = start + w
            conv_weights[start:stop,:,i] = motif   
         
    seq_input = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq_input)
    
    shared_conv = Conv1D(k, 24, 
                         strides=1, 
                         padding='same', 
                         activation="linear", 
                         use_bias=False,
                         name="conv1d_1", 
                         trainable=False)
    
    conv_for = shared_conv(seq_input)
    conv_rc = shared_conv(seq_rc)
    
    bn = BatchNormalization()
    bn_for = bn(conv_for)
    bn_rc = bn(conv_rc)
    
    merged = maximum([bn_for, bn_rc])
    
#     merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    output = Dense(1, activation='sigmoid', use_bias=False,
                   kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.01, seed=12), 
                   kernel_constraint=non_neg(), 
                   bias_initializer='zeros',
                   name="dense_1")(pooled)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer("conv1d_1").set_weights([conv_weights])
    return(model)

def construct_rf(motif_file="hocomoco"):
    url = "http://hocomoco11.autosome.ru/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pwms_HUMAN_mono.txt"
    data = requests.get(url).text
    with open("motifs.txt","w") as f:
        f.write(data)

    PWMS_LIST = []
    with open("motifs.txt") as f:
        lines = [line.strip().split() for line in f.readlines()]

    pwm = []
    motif_ids = []
    for i, line in enumerate(lines):
        if line[0][0] == ">":
            motif_id = line[0][1:]
            if i > 0:
                pwm = np.reshape(np.array(pwm), (-1,4))
                PWMS_LIST.append(pwm)
                motif_ids.append(motif_id)
            pwm = []
        else:
            pwm.append([float(x) for x in line])
    max_w = np.max(np.array([pwm.shape[0] for pwm in PWMS_LIST]))
    k = len(PWMS_LIST)
    print(k)

    conv_weights = np.zeros((max_w,4,k))
    for i in range(k):
        pwm = PWMS_LIST[i]
        w = pwm.shape[0]
        max_val = np.sum(np.max(pwm, axis=1))
        start = (max_w - w) // 2
        stop = start + w
        conv_weights[start:stop,:,i] = pwm   
         
    seq_input = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq_input)
    
    shared_conv = Conv1D(k, 24, 
                         strides=1, 
                         padding='same', 
                         activation="linear", 
                         use_bias=False,
                         name="conv1d_1", 
                         trainable=False)
    
    conv_for = shared_conv(seq_input)
    conv_rc = shared_conv(seq_rc)
    
    merged = maximum([conv_for, conv_rc])
    
    pooled = GlobalMaxPooling1D()(merged)
    
    model = Model(inputs=seq_input, outputs=pooled)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer("conv1d_1").set_weights([conv_weights])
    return(model, motif_ids)