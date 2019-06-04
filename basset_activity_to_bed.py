import sys
from pybedtools import BedTool
from numpy import random
random.seed(12)

train_pct = 0.7
test_pct = 0.2
eval_pct = 0.1

genome = "/project/umw_zhiping_weng/andrewsg/genome/hg38/hg38.fa"
intervals = []
basset_activity = sys.argv[1]
with open(basset_activity, "r") as f:
    acc = f.readline().strip().split()[0]
    print(acc)
    for line in f:
        chrom = line.strip().split(":")[0]
        start = line.strip().split(":")[1].split("-")[0]
        stop = line.strip().split(":")[1].split("(")[0].split("-")[1]
        label = line.strip().split()[1]
        intervals.append((chrom, start, stop, int(label)))

num_intervals = len(intervals)
print("There are {} regions in {}'s acctivity table".format(len(intervals), acc))
print("Ordering intervals")

order = random.permutation(num_intervals)
new_intervals = []
for i in range(num_intervals):
    new_intervals.append(intervals[order[i]])
    
intervals = new_intervals

num_test = int(test_pct * num_intervals)
num_eval = int(eval_pct * num_intervals)
num_train = num_intervals - num_test - num_eval
print("...{} for training".format(num_train))
print("...{} for testing".format(num_test))
print("...{} for evaluating".format(num_eval))

slice_index = 0
train_intervals = intervals[0:num_train]
slice_index += num_train
test_intervals = intervals[slice_index:slice_index+num_test]
slice_index += num_test
eval_intervals = intervals[slice_index:]

train_pos_bt = BedTool([interval for interval in train_intervals if interval[3] == 1])
train_neg_bt = BedTool([interval for interval in train_intervals if interval[3] == 0])

test_pos_bt = BedTool([interval for interval in test_intervals if interval[3] == 1])
test_neg_bt = BedTool([interval for interval in test_intervals if interval[3] == 0])

eval_pos_bt = BedTool([interval for interval in eval_intervals if interval[3] == 1])
eval_neg_bt = BedTool([interval for interval in eval_intervals if interval[3] == 0])

def bt_to_bed(bt, bed_file):
    with open(bed_file, "w") as f:
        for interval in bt:
            chrom = interval.chrom
            start = interval.start
            stop = interval.stop
            f.write("\t".join([chrom, str(start), str(stop)]) + "\n")

bt_to_bed(train_pos_bt, acc + ".train.pos.bed")
bt_to_bed(train_neg_bt, acc + ".train.neg.bed")
bt_to_bed(test_pos_bt, acc + ".test.pos.bed")
bt_to_bed(test_neg_bt, acc + ".test.neg.bed")
bt_to_bed(eval_pos_bt, acc + ".eval.pos.bed")
bt_to_bed(eval_neg_bt, acc + ".eval.neg.bed")

