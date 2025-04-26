from miditok import REMI, TokenizerConfig
from symusic import Score
import numpy as np
import pickle
import random
import os

BEAT_RES = {(0, 1): 24, (1, 2): 8, (2, 4): 4, (4, 8): 2}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

dict_path = "./stage02_embellish/vocab/skyline_miditok_vocab.pkl"
event2idx, idx2event = read_pickle(dict_path)

def midi2txt(midi_path, txt_path):
    midi = Score(midi_path)
    tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
    with open(txt_path, "w") as file:
        for item in tokens[0].tokens:
            file.write(item + "\n")

def gnpy2midi(npy_path, midi_path="/content/test_from_npy.mid"):
    tokens = np.load(npy_path, allow_pickle=True)
    #tokens = tokens.reshape(1, -1)
    tokens = np.array([event2idx[e] for e in tokens]).reshape(1,-1)
    converted_back_midi = tokenizer(tokens)
    converted_back_midi.dump_midi(midi_path) # Save the MIDI file

def pkl2txt(pkl_path, txt_path):
    skyline_pos, midi_pos, all_events = read_pickle(pkl_path)
    tokens = [all_events[pos[0]+1:pos[1]] for pos in skyline_pos]
    flattened = [item for row in tokens for item in row]
    
    # truncate for testing
    length = len(flattened)//10
    flattened = flattened[:1000]

    with open(txt_path, "w") as file:
        for item in flattened:
            file.write(item["name"]+"_"+item["value"] + "\n")


def pkl2orig(pkl_path, orig_path):
    skyline_pos, midi_pos, all_events = read_pickle(pkl_path)
    tokens = [all_events[pos[0]+1:pos[1]] for pos in midi_pos]
    flattened = [item for row in tokens for item in row]
    
    tokens = np.array([event2idx[e["name"]+"_"+e["value"]] for e in flattened]).reshape(1,-1)
    converted_back_midi = tokenizer(tokens)
    converted_back_midi.dump_midi(orig_path) # Save the MIDI file


train_split = read_pickle("stage02_embellish/pkl/train.pkl")
valid_split = read_pickle("stage02_embellish/pkl/valid.pkl")
compo_split = read_pickle("stage02_embellish/pkl/composer_split.pkl")

# to_test = ["Bach_JohannSebastian", "Mozart_WolfgangAmadeus", "Beethoven_Ludwigvan"]
to_test = ["Mozart_WolfgangAmadeus"]
generation_split = {}

dataset_path = "/home/yihsin/MidiStyleTransfer/dataset/gp-piano-parsed"

for composer in to_test:
    # print("composer:", composer)
    all_songs = compo_split[composer]
    # print("all_songs:", all_songs)
    all_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # print("composer:", composer)
    # print("all_files:", all_files[:50])

    # Collect all files that start with any prefix in self.pieces
    matched_files = []
    for prefix in all_songs:
        prefix = ' ' + prefix
        matched = [f for f in all_files if f.startswith(prefix)]
        matched_files.extend(matched)
    # print("matched_files:", matched_files)
    # print("train_split:", train_split)
    # temp_train = random.sample([s for s in matched_files if f"{s}.pkl" in train_split],3)
    # temp_valid = random.sample([s for s in matched_files if f"{s}.pkl" in valid_split],2)
    # generation_split[composer] = {"train":temp_train, "valid":temp_valid, "all":temp_train+temp_valid}
    sample = [f for f in os.listdir(dataset_path) if f.startswith(" qjk-YRuQZDE") and f.endswith(".pkl")]
    generation_split[composer] = {"all": sample}

test_composer = "mozart"
postfix = "k545"

for piece in generation_split["Mozart_WolfgangAmadeus"]["all"]:
    generation_home = "/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/generation"
    pkl_file = f"/home/yihsin/MidiStyleTransfer/dataset/gp-piano-parsed/{piece}"

    if not os.path.exists(f"{generation_home}/stage01_{test_composer}{postfix}"):
        os.makedirs(f"{generation_home}/stage01_{test_composer}{postfix}")

    pkl2txt(
        pkl_file, 
        f"{generation_home}/stage01_{test_composer}{postfix}/{piece}.txt")
    
    if not os.path.exists(f"{generation_home}/stage02_{test_composer}{postfix}"):
        os.makedirs(f"{generation_home}/stage02_{test_composer}{postfix}")
        
    pkl2orig(
        pkl_file,
        f"/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/generation/stage02_{test_composer}{postfix}/{piece}.mid")