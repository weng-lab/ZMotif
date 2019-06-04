from utils import fasta_to_array
from keras.models import load_model
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

prefix = sys.argv[1]

pos_array = fasta_to_array(prefix + ".pos.eval.fasta")
neg_array = fasta_to_array(prefix + ".neg.eval.fasta")
#model = load_model(prefix + ".refined.model.h5")

#y_pred_pos = model.predict(pos_array)
#y_pred_neg = model.predict(neg_array)
y_pred_1 = np.loadtxt("wgEncodeAwgTfbsHaibH1hescNanogsc33759V0416102UniPk.y_pred_1")
y_pred_2 = np.loadtxt("wgEncodeAwgTfbsHaibH1hescNanogsc33759V0416102UniPk.y_pred_2")
y_pred_3 = np.loadtxt("wgEncodeAwgTfbsHaibH1hescNanogsc33759V0416102UniPk.y_pred_3")
y_pred_4 = np.loadtxt("wgEncodeAwgTfbsHaibH1hescNanogsc33759V0416102UniPk.y_pred_4")
y_pred_5 = np.loadtxt("wgEncodeAwgTfbsHaibH1hescNanogsc33759V0416102UniPk.y_pred_5")

y_pred = np.vstack((y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5))
y_pred = np.transpose(y_pred)
print(y_pred.shape)
y_pred = np.mean(y_pred, axis = 1)
print(y_pred.shape)
#y_pred = np.vstack((y_pred_pos, y_pred_neg))
y_true = np.array([1 for i in range(pos_array.shape[0])] + [0 for i in range(neg_array.shape[0])])
print(roc_auc_score(y_true, y_pred))
np.savetxt(prefix + ".y_pred", y_pred, fmt='%.5f')
