import json
import pickle as pkl

file_path1 = "/data1/chenyuxuan/Project/MSMLM/data/processed_train_data_diffusion2.json"
file_path = "/data1/chenyuxuan/Project/MSMLM/data/processed_train_data_diffusion.pkl"
with open(file_path, "rb") as f:
    data = pkl.load(f)
    
print(data[0])
with open(file_path1, "w") as f:
    json.dump(data, f, indent=2)
