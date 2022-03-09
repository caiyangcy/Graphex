import os
import random
import numpy as np
import json
from shutil import copy

seed = 42
random.seed(seed)
np.random.seed(seed)


def read_data(filepath):
    data = open(filepath, "r").readlines()
    if data[-1].strip() == '':
        data = data[:-1]
    return data 

def save_data(filepath, data):
    with open(filepath, "w") as f:
        f.writelines(data)


dataset_path = "Graphine/"
save_path = "data/"

merged_data_path = "data_partial"

merged_train_defs, merged_train_names = [], []
merged_val_defs, merged_val_names = [], []
merged_test_defs, merged_test_names = [], []

merged_graphs = {}

if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(merged_data_path):
    os.mkdir(merged_data_path)

train_ratio, val_ratio = 0.7, 0.1

for subdir, _, files in os.walk(dataset_path):

    print("subdir: ", subdir)

    data_folder = subdir.split("/")[-1]

    if data_folder not in ["ago", "agro", "apo", "aro", "atmo"]: # for the purpose of reproduction - only choose few folders
        continue

    folder_to_save = save_path + data_folder

    if not os.path.exists(folder_to_save):
        os.mkdir(folder_to_save)

    for file in sorted(files):

        data_path = subdir + "/" + file
        
        print("doc_path: ", data_path)

        if file.endswith("def.txt"):
            defs = read_data(data_path)

        elif file.endswith("name.txt"):
            names = read_data(data_path)
            assert len(defs) == len(names), f"definition length does not match name length, definition: {len(defs)}; name: {len(names)}"

            data = list(zip(defs, names))
            random.shuffle(data)
            defs, names = zip(*data)
            defs = list(defs)
            names = list(names)

            train_length = int( len(defs)*train_ratio )
            val_length = int( len(defs)*val_ratio )
            train_defs, val_defs, test_defs = defs[:train_length], defs[train_length:train_length+val_length], defs[train_length+val_length:]
            train_names, val_names, test_names = names[:train_length], names[train_length:train_length+val_length], names[train_length+val_length:]

            save_data(folder_to_save + "/train_def.txt", train_defs)
            save_data(folder_to_save + "/valid_def.txt", val_defs)
            save_data(folder_to_save + "/test_def.txt", test_defs)

            save_data(folder_to_save + "/train_name.txt", train_names)
            save_data(folder_to_save + "/valid_name.txt", val_names)
            save_data(folder_to_save + "/test_name.txt", test_names)

            merged_train_defs += train_defs
            merged_train_names += train_names

            merged_val_defs += val_defs
            merged_val_names += val_names

            merged_test_defs += test_defs
            merged_test_names += test_names

        elif file.endswith(".json"):
            copy(data_path, folder_to_save + "/" + file )

            with open(data_path, 'r') as f:
                graph = json.load(f)

                for k, v in graph.items():
                    if k not in merged_graphs:
                        merged_graphs[k] = v
                    else:
                        merged_graphs[k] += v

save_data(merged_data_path + "/train_def.txt", merged_train_defs)
save_data(merged_data_path + "/valid_def.txt", merged_val_defs)
save_data(merged_data_path + "/test_def.txt", merged_test_defs)

save_data(merged_data_path + "/train_name.txt", merged_train_names)
save_data(merged_data_path + "/valid_name.txt", merged_val_names)
save_data(merged_data_path + "/test_name.txt", merged_test_names)

with open(merged_data_path + "/graph.json", 'w') as f:
    json.dump(merged_graphs, f)