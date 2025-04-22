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

def gnpy2midi(npy_path, midi_path="/content/test_from_npy.mid"):
    tokens = np.load(npy_path, allow_pickle=True)
    #tokens = tokens.reshape(1, -1)
    tokens = np.array([event2idx[e] for e in tokens]).reshape(1,-1)
    converted_back_midi = tokenizer(tokens)
    converted_back_midi.dump_midi(midi_path) # Save the MIDI file

test_composer = "mozart"
postfix = "k331"
generation_home = "/home/yihsin/MidiStyleTransfer/Compose_and_Embellish_classical/generation"
for g in os.listdir(f"{generation_home}/stage02_{test_composer}{postfix}"):
    if(g.split(".")[1]=="npy"):
        idx = g.split(".")[0]
        gnpy2midi(
            f"{generation_home}/stage02_{test_composer}{postfix}/{g}",
            #f"{generation_home}/midi_samples_0413/{test_composer}/{idx}_finetuned.mid"
            f"{generation_home}/stage02_{test_composer}{postfix}/{idx}.mid"
        )