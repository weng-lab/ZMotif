import json
import sys
from collections import OrderedDict


def get_top_match(motif_id, tomtom_file):
    lines = []
    p_values = []
    matches = []
    with open(tomtom_file) as f:
        for line in f:
            if line.strip():
                lines.append(line.strip().split())
    
    for line in lines:
        if line[0] == motif_id:
            #print("found a match")
            matches.append(line[1])
            #print(match)
            p_values.append(float(line[3]))
            #print(p_value)
            

    if len(p_values) == 0:
        p_values = [1]
    if len(matches) == 0:
        matches = ["No Match"]

    return matches, p_values


meme_input = sys.argv[1]
hoc_tomtom_input = sys.argv[2]
jaspar_tomtom_input = sys.argv[3]
output_prefix = sys.argv[4]
#tomtom_input = sys.argv[2]
lines = []
with open(meme_input) as f:
    for line in f:
        if line.strip():
            lines.append(line.strip().split())

motif_ids = []
widths = []
n_sites = []
start_indices = []
stop_indices = []
for index, line in enumerate(lines):
    if line[0] == 'MOTIF':
        motif_ids.append(line[1])
    if line[0] == 'letter-probability':
        widths.append(int(line[5]))
        n_sites.append(int(line[7]))
        start_indices.append(index + 1)
        stop_indices.append(index + widths[-1] + 1)


motif_dict = {}
for index, motif in enumerate(motif_ids):
    motif_dict[motif] = {'id' : motif,
                         'n_sites' : n_sites[index],
                         'ppm' : lines[start_indices[index]:stop_indices[index]]}

ordered_motif_dict  = OrderedDict(sorted(motif_dict.items(), key=lambda x: x[1]['n_sites'], reverse = True))
#print(motif_dict)

for key in ordered_motif_dict:
    #print(key)
    hoc_matches, hoc_p_values = get_top_match(key, hoc_tomtom_input)
    jaspar_matches, jaspar_p_values = get_top_match(key, jaspar_tomtom_input)
    #print(matches, p_values)
    ordered_motif_dict[key]['hoc_matches'] = hoc_matches
    ordered_motif_dict[key]['hoc_p_values'] = hoc_p_values
    ordered_motif_dict[key]['jaspar_matches'] = jaspar_matches
    ordered_motif_dict[key]['jaspar_p_values'] = jaspar_p_values

output_json = output_prefix + ".json"
with open(output_json, "w") as f:
    f.write(json.dumps(ordered_motif_dict))

    
