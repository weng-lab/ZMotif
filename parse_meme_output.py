import sys
import numpy as np
import pickle

def ppms_to_meme(ppms_tup_list, output_file, prefix):
    with open(output_file, "w") as f:
        f.write("MEME version 4 \n\n")
        f.write("ALPHABET= ACGT \n\n")
        f.write("strands: + - \n\n")
        f.write("Background letter frequencies \n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")
        for i, ppm_tup in enumerate(ppms_tup_list):
            ppm = ppm_tup[0]
            nsites = ppm_tup[1]
            f.write("MOTIF " + prefix + "." + str(i+1) + "\n")
            f.write("letter-probability matrix: alength= 4 w= {} nsites= {} \n".format(str(ppm.shape[0]), nsites))
            for j in range(ppm.shape[0]):
                tempPpm = ppm[j,:]
                f.write("\t".join([str(x) for x in tempPpm]))
                f.write("\n")
            f.write("\n")


input_file = sys.argv[1]
output_prefix = sys.argv[2]

motif = []
ppms = []
in_motif = False
with open(input_file, "r") as f:
    for line in f:
        if len(line.strip().split()) > 0:
            if in_motif:
                if line.strip().split()[0][0] == '-':
                    in_motif = False
                    ppms.append((np.array(motif), n_sites))
                    motif = []
                else:
                    motif.append(line.strip().split())
            
            if line.strip().split()[0] == 'letter-probability':
                n_sites = line.strip().split()[7]
                print("found a motif")
                in_motif = True

ppms_to_meme(ppms, output_prefix + ".meme.meme", output_prefix)
with open(output_prefix + ".meme.ppms.pkl", "wb") as f:
    pickle.dump(ppms, f)
