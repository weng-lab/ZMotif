import sys
import json

trimmed_bed = sys.argv[1]
output_json = sys.argv[2]


data_dict = {}

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

for motif_id in motif_ids:
    data_dict[motif_id] = []
    
with open(trimmed_bed) as f:
    for line in f:
        chrom, motif_start, motif_end, score, strand, seq, kernel_id, coord = line.strip().split()
        peak_start = int(coord.split(":")[1].split("-")[0])
        peak_end = int(coord.split(":")[1].split("-")[1])
        peak_center = (peak_end + peak_start) // 2
        #print(coord, peak_start, peak_end, peak_center)
        motif_center = (int(motif_start) + int(motif_end)) // 2
        #print(motif_center)
        data_dict[kernel_id].append(peak_center - motif_center)
        
with open(output_json, "w") as f:
    json.dump(data_dict, f)