import numpy as np
from src.utils import progress

def filter_seqs(seqs, model):
    num_bad = 0
    new_seqs = []
    for i, seq in enumerate(seqs):
        progress(i, len(seqs), "Filtering bad sequences")
        y_pred = model.predict(np.expand_dims(seq[0], axis=0))
        label = seq[1]
        if label == 1:
            if y_pred > 0.5:
                new_seqs.append(seq)
        else:
            if y_pred < 0.5:
                new_seqs.append(seq)
    progress(i, len(seqs), "\n")
    return(new_seqs)