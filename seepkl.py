import pickle

# Path to your .pkl file
file_path = '/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/stage02_embellish/pkl/train.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n✅ Loaded: {file_path}")
    print(f"Type: {type(data)}")
    print("\n--- Full Content ---\n")
    print(data)

except Exception as e:
    print(f"❌ Error reading file: {e}")
