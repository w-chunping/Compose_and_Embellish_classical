import os
import pickle

# Path to your train.pkl file
train_pkl_path = '/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/stage02_embellish/pkl/valid.pkl'
new_train_pkl_path = '/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/stage02_embellish/pkl/ppvalid.pkl'
# Folder where the actual data files live
target_dir = '/home/yihsin/MidiStyleTransfer/dataset/gp-piano-parsed'

# Load the original prefix-only list from train.pkl
with open(train_pkl_path, 'rb') as f:
    prefix_ids = pickle.load(f)  # e.g. ['0rnJu1rlm90.pkl', '123abcXYZ.pkl']

# Remove '.pkl' to get just the prefix
prefixes = [os.path.splitext(f)[0] for f in prefix_ids]

# List all .pkl files in the target directory
all_files = os.listdir(target_dir)
all_pkl_files = [f for f in all_files if f.endswith('.pkl')]

# Find all files that start with one of the prefixes
new_train_files = []
for prefix in prefixes:
    matches = [f for f in all_pkl_files if f.startswith(prefix + '_')]  # e.g. 0rnJu1rlm90_
    new_train_files.extend(matches)

# Optionally sort for consistency
new_train_files.sort()

# Save the updated list to train.pkl
with open(new_train_pkl_path, 'wb') as f:
    pickle.dump(new_train_files, f)

print(f"✅ Updated train.pkl with {len(new_train_files)} entries.")
