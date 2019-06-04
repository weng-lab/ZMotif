from utils import fasta_to_array
from keras.models import load_model
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

prefix = sys.argv[1]

pos_array = fasta_to_array("/home/andrewsg/data/motif_finder/deepbind_data/fasta/" + prefix + ".pos.eval.fasta")
neg_array = fasta_to_array("/home/andrewsg/data/motif_finder/deepbind_data/fasta/" + prefix + ".neg.eval.fasta")
model = load_model(prefix + ".model.h5")

y_pred_pos = model.predict(pos_array)
y_pred_neg = model.predict(neg_array)

y_pred = np.vstack((y_pred_pos, y_pred_neg))
y_true = np.array([1 for i in range(pos_array.shape[0])] + [0 for i in range(neg_array.shape[0])])
print(roc_auc_score(y_true, y_pred))
