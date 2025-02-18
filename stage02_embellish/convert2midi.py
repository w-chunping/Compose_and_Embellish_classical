import miditoolkit
import numpy as np

##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16


##############################
# containers for conversion
##############################
class ConversionEvent(object):
  def __init__(self, event, is_full_event=False):
    if not is_full_event:
      if 'Note' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Chord' in event:
        self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
      else:
        self.name, self.value = event.split('_')
    else:
      self.name, self.value = event['name'], event['value']
  def __repr__(self):
    return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
  def __init__(self, pitch, bar, position, duration, velocity, microtiming=None):
    self.pitch = pitch
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
    self.duration = duration
    self.velocity = velocity

    if microtiming is not None:
      self.start_tick += microtiming

  def set_microtiming(self, microtiming):
    self.start_tick += microtiming
  
  def set_velocity(self, velocity):
    self.velocity = velocity
  
  def __repr__(self):
    return 'Note(pitch = {}, duration = {}, start_tick = {})'.format(
      self.pitch, self.duration, self.start_tick
    )
  
class TempoEvent(object):
  def __init__(self, tempo, bar, position):
    self.tempo = tempo
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
  
  def set_tempo(self, tempo):
    self.tempo = tempo

  def __repr__(self):
    return 'Tempo(tempo = {}, start_tick = {})'.format(
      self.tempo, self.start_tick
    )

class ChordEvent(object):
  def __init__(self, chord_val, bar, position):
    self.chord_val = chord_val
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)


##############################
# conversion functions
##############################

def event_to_midi(events, mode, output_midi_path=None, is_full_event=False, 
                  return_tempos=False, enforce_tempo=False, enforce_tempo_evs=None):
  events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
  # print (events[:20])

  # assert events[0].name == 'Bar'
  temp_notes = []
  temp_tempos = []
  temp_chords = []

  cur_bar = -1
  cur_position = 0

  for i in range(len(events)):
    if events[i].name == 'Bar':
      cur_bar += 1
    elif events[i].name == 'Beat':
      cur_position = int(events[i].value)
      assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
    #   print (cur_bar, cur_position)
    elif events[i].name == 'Tempo' and 'Conti' not in events[i].value:
      temp_tempos.append(TempoEvent(
        int(events[i].value), max(cur_bar, 0), cur_position
      ))
    elif 'Note_Pitch' in events[i].name:
      if mode == 'full' and \
         (i+1) < len(events) and 'Note_Duration' in events[i+1].name and \
         (i+2) < len(events) and 'Note_Velocity' in events[i+2].name:
        # check if the 3 events are of the same instrument
        temp_notes.append(
          NoteEvent(
            pitch=int(events[i].value), 
            bar=cur_bar, position=cur_position, 
            duration=int(events[i+1].value), velocity=int(events[i+2].value)
          )
        )
      elif mode == 'skyline' and \
        (i+1) < len(events) and 'Note_Duration' in events[i+1].name:
        temp_notes.append(
          NoteEvent(
            pitch=int(events[i].value), 
            bar=cur_bar, position=cur_position, 
            duration=int(events[i+1].value), velocity=80
          )
        )
    elif 'Chord' in events[i].name and 'Conti' not in events[i].value:
      temp_chords.append(
        ChordEvent(events[i].value, cur_bar, cur_position)
      )
    elif events[i].name in ['EOS', 'PAD']:
      continue

  print ('# tempo changes:', len(temp_tempos), '| # notes:', len(temp_notes))
  midi_obj = miditoolkit.midi.parser.MidiFile()
  midi_obj.instruments = [
    miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
  ]

  for n in temp_notes:
    midi_obj.instruments[0].notes.append(
      miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )

  if enforce_tempo is False:
    for t in temp_tempos:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )
  else:
    if enforce_tempo_evs is None:
      enforce_tempo_evs = temp_tempos[1]
    for t in enforce_tempo_evs:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )

  
  for c in temp_chords:
    midi_obj.markers.append(
      miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
    )
  for b in range(cur_bar):
    midi_obj.markers.append(
      miditoolkit.Marker('Bar-{}'.format(b+1), int(DEFAULT_BAR_RESOL * b))
    )

  if output_midi_path is not None:
    midi_obj.dump(output_midi_path)

  if not return_tempos:
    return midi_obj
  else:
    return midi_obj, temp_tempos
