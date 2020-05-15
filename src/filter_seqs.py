import numpy as np
from src.utils import progress
    
def filter_seqs(seqs, model):
    preds = []
    pos_seqs = [seq for seq in seqs if seq[1] == 1]
    for i, seq in enumerate(pos_seqs):
        progress(i, len(pos_seqs), "Filtering bad sequences")
        y_pred = model.predict(np.expand_dims(seq[0], axis=0))
        preds.append(y_pred[0,0])
    print(sorted(preds(reverse=True)))
    
    return(new_seqs)