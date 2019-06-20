import sys
import itertools

trimmed_bed = sys.argv[1]

num_lines = 0
with open(trimmed_bed) as f:
    for line in f:
        if num_lines == 0:
            start = int(line.strip().split()[1])
            stop = int(line.strip().split()[2])
            w = stop - start
            #print(w)
        num_lines += 1
        
motif_ids = ["" for i in range(num_lines)]

with open(trimmed_bed) as f:
    for i, line in enumerate(f):
        motif_ids[i] = line.strip().split()[6]
        
motif_ids = list(set(motif_ids))
for motif_pair in itertools.combinations(motif_ids, r=2):
    motif_1, motif_2 = motif_pair
    with open 
