acc_list = []
with open("acc_to_analyze.txt") as f:
    for line in f:
        acc_list.append(line.strip().split()[0])

metadata_list = []
with open("metadata.csv") as f:
    for line in f:
        acc_num = line.strip().split(",")[0]
        acc = "cistrome_" + acc_num
        if acc in acc_list:
            cell = line.strip().split(",")[3]
            if cell == 'None':
                cell = line.strip().split(",")[4]
                if cell == 'None':
                    cell = line.strip().split(",")[5]
            tf = line.strip().split(",")[6]
            metadata_list.append((tf, cell, acc))

with open("visualizer_metadata.csv", "w") as f:
    f.write("tf" + "," + "cell" + "," + "experiment" + "\n")
    for entry in metadata_list:
        f.write(entry[0] + "," +  entry[1] +  "," +  entry[2] + "\n")
