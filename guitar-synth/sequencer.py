#import ipdb

import numpy as np

from guitar import Guitar


time_unit = 0.12

chords = [ Guitar.C_MAJOR,
           Guitar.G_MAJOR,
           Guitar.A_MINOR,
           Guitar.E_MINOR ]


def play_strums(guitar, sequenceN, block_start_time, chord_index, precache_time):

    cur_strum_start_time = 0

    audio_buffers = []

    for i in range(13):

        chord = chords[chord_index]

        if sequenceN % 13 == 0:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 1:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 2:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 3:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 4:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 5:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 6:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 7:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 8:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 9:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 10:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, True,  1.0, chord)
        elif sequenceN % 13 == 11:
            audio_buffer = guitar.strum_chord(cur_strum_start_time, False, 0.8, chord)
        elif sequenceN % 13 == 12:
            audio_buffer = guitar.strings[2].pluck(cur_strum_start_time, 0.7, chord[2])

            # cur_strum_start_time = block_start_time + time_unit * 31.5
            # audio_buffer = guitar.strings[1].pluck(cur_strum_start_time, 0.7, chord[1])

            chord_index = (chord_index + 1) % 4
            # block_start_time += time_unit * 32

        chord_index = (chord_index + 1) % 4

        audio_buffers.append(audio_buffer)
        sequenceN += 1

    return np.concatenate(audio_buffers)


# def play_strums(guitar, sequenceN, block_start_time, chord_index, precache_time):
#     cur_strum_start_time = 0
#     chord = chords[chord_index]
#     audio_buffer = guitar.strum_chord(cur_strum_start_time, True, 1.0, chord)
#     return audio_buffer

def play_note(guitar, stringNumber, tab, freq, smoothing_factor):
        return guitar.note(1.0, stringNumber, tab, freq, smoothing_factor)
    
    
def play_guitar(guitar):
    start_sequence_N = 0
    block_start_time = 10
    start_chord_index = 0
    precache_time = 0.0
    return play_strums(guitar, start_sequence_N, block_start_time, start_chord_index, precache_time)
