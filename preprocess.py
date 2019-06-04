import numpy as np

def bedtool_to_intervals(bt, genome):
    bt  = bt.remove_invalid()
    pos_intervals = []
    neg_intervals = []
    num_intervals = 0
    num_bad = 0
    for interval in bt:
        try:
            num_intervals += 1
            chrom = interval.chrom
            start = interval.start
            stop = interval.stop
            label = int(interval.name)
            seq = genome[chrom][start:stop]
            if label == 1:
                pos_intervals.append((chrom, start, stop))
            else:
                neg_intervals.append((chrom, start, stop))
                
        except:
            num_bad += 1
            pass
    return pos_intervals, neg_intervals

def seq_lens_from_intervals(intervals):
    seq_lens = []
    for interval in intervals:
        chrom, start, stop = interval
        seq_lens.append(stop - start)
    
    return(np.array(seq_lens))