from src.custom_callbacks import SGDRScheduler, ProgBar
from src.models import construct_pretrain_model, construct_model, construct_lr
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

def pretrain_lr(train_gen, test_gen, motif_file="hocomoco", num_epochs=100, k=32):
    model = construct_lr(motif_file=motif_file)
    
    steps_per_epoch = train_gen.get_steps_per_epoch()
    validation_steps = test_gen.get_steps_per_epoch()
    
    schedule = SGDRScheduler(min_lr=.01,
                             max_lr=.1,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=1,
                             mult_factor=1.0,
                             shape="triangular")
    
    progbar = ProgBar(num_epochs)
    
    early_stopping = EarlyStopping(patience=5)
    
    callbacks_list = [schedule, progbar, early_stopping]
    
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=test_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks_list,
                                  shuffle=True,
                                  use_multiprocessing=False,
                                  workers=1,
                                  max_queue_size=10,
                                  verbose=0)
    
    dense_weights = model.get_layer("dense_1").get_weights()[0]
    conv_weights = model.get_layer("conv1d_1").get_weights()[0]
    indices = np.ravel(dense_weights.argsort(axis=0)[-k:])
    print(indices)
    return(conv_weights[:,:,indices], dense_weights[indices,:])
       
    