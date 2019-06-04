import sys
import numpy as np
from sklearn.metrics import roc_auc_score

test_data = sys.argv[1]
gifford_predictions = sys.argv[2]

y_true = []
with open(test_data, "r") as f:
    for line in f:
        y_true.append(int(line.strip().split()[2]))

y_true = np.array(y_true)
y_pred_gifford = np.loadtxt(gifford_predictions)
auc_gifford = roc_auc_score(y_true, y_pred_gifford)
print(auc_gifford)
