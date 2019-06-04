import sys
import numpy as np
from sklearn.metrics import roc_auc_score

acc_list = []
with open("acc_list.txt") as f:
    for line in f:
        acc = line.strip().split()[0]
        acc_list.append(acc)

results_dir = "gifford/results_10negs"
data_dir = "./gifford/motif_discovery/data"
predictions_dir = "./gifford/motif_discovery/predictions"

for acc in acc_list:
    try:
        
        auc_file = results_dir + "/" + acc + ".auroc"
        with open(auc_file) as f:
            zmotif_auc = f.readline().strip().split()[0]
        
        test_data_file = data_dir + "/" + acc + "/test.data"
        train_data_file = data_dir + "/" + acc + "/train.data"
    
        num_obs = 0
        with open(train_data_file) as f:
            for line in f:
                num_obs += 1
                
        y_true = []
        with open(test_data_file, "r") as f:
            for line in f:
                y_true.append(int(line.strip().split()[2]))

        y_true = np.array(y_true)
        
        gifford_models = ["1layer", "1layer_64motif", "1layer_128motif"]
        gifford_aucs = []
        for model in gifford_models:
            gifford_predictions_file = predictions_dir + "/" + acc + "/" + model + "/bestiter.pred"
            y_pred_gifford = np.loadtxt(gifford_predictions_file)
            auc = roc_auc_score(y_true, y_pred_gifford)
            gifford_aucs.append(auc)
  
        with open("auroc.txt", "a") as  f:
            f.write("\t".join([acc, str(zmotif_auc)] +
                              [str(auc) for auc in gifford_aucs] + 
                              [str(num_obs)]) + "\n")
            
    except:
        print("file not found")
       
    
    