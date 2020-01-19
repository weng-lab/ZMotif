import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from src.custom_callbacks import ProgBar
from src.models import construct_model

def refine_kernels(conv_weights, dense_weights, train_gen, test_gen):
    num_kernels = conv_weights.shape[2]
    w = conv_weights.shape[0]
    steps_per_epoch = train_gen.get_steps_per_epoch()
    validation_steps = test_gen.get_steps_per_epoch()
    new_conv_weights = np.zeros((w,4,num_kernels))
    for i in range(num_kernels):
        refined_kernel = conv_weights[:,:,i]
        left_start = 0
        right_start = w-1
        for j in range(w):
            if np.max(refined_kernel[j,:]) < 0:
                left_start += 1
            else:
                break
        
        for j in range(w):
            if np.max(refined_kernel[w-j-1,:]) < 0:
                right_start -= 1
            else:
                break
        
        refined_kernel = refined_kernel[left_start:right_start+1,:]
        print(refined_kernel.shape)
        model = construct_model(num_kernels=1, kernel_width=refined_kernel.shape[0], seq_len=None, optimizer=Adam(.01))
    
        model.get_layer('conv1d_1').set_weights([refined_kernel[:,:,np.newaxis]])
        model.get_layer('dense_1').set_weights([dense_weights[i,np.newaxis]])
        
        early_stopping = EarlyStopping(patience=5)
        progbar = ProgBar(100)
        callbacks = [early_stopping, progbar]
        history = model.fit_generator(train_gen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=100,
                              validation_data=test_gen,
                              validation_steps=validation_steps,
                              callbacks=callbacks,
                              shuffle=True,
                              use_multiprocessing=False,
                              workers=1,
                              max_queue_size=10,
                              verbose=0)
        
        new_conv_weights[:refined_kernel.shape[0],:,i] = model.get_layer('conv1d_1').get_weights()[0][:,:,0]
    return(new_conv_weights)