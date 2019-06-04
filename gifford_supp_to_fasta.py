import sys
input_file = sys.argv[1]
output_prefix = sys.argv[2]
data = []
with open(input_file, "r") as f:
    for line in f:
        coord = line.strip().split()[0][1:]
        seq = line.strip().split()[1]
        label = int(line.strip().split()[2])
        data.append((coord, seq, label))

pos_fasta = output_prefix + ".pos.fasta"
neg_fasta = output_prefix + ".neg.fasta"
with open(pos_fasta, "w") as pos, open(neg_fasta, "w") as neg:
    for entry in data:
        coord, seq, label = entry
        if label == 1:
            pos.write(">" + coord + "\n")
            pos.write(seq + "\n")
        else:
            neg.write(">" + coord + "\n")
            neg.write(seq + "\n")
        

