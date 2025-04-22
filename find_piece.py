import pickle

target_piece = " vp_h649sZ9A.pkl"
pool_path = "/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/stage02_embellish/pkl/train.pkl"

with open(pool_path, "rb") as f:
    data = pickle.load(f)

if target_piece in data:
    print(target_piece, "is in", pool_path)
else:
    print(target_piece, "not in", pool_path)