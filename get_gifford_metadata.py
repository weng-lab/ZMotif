metadata = []
with open("files.txt") as f:
    for line in f:
        for entry in line.strip().split():
            if entry.split("=")[0] ==  "tableName":
                acc = entry.split("=")[1][:-1]
            if entry.split("=")[0] ==  "quality":
                quality = entry.split("=")[1][:-1]
            if entry.split("=")[0] ==  "antibody":
                tf = entry.split("=")[1][:-1].split("_")[0]
            if entry.split("=")[0] ==  "cell":
                cell = entry.split("=")[1][:-1]
        metadata.append((acc, tf, cell, quality))
        
with open("gifford_metadata.txt", "w") as f:
    for entry in metadata:
        f.write("\t".join(entry) + "\n")
     
