#import ipdb
import numpy as np

# Local imports
from guitarstring import GuitarString


class Guitar:

    # each fret represents an increase in pitch by one semitone
    # (logarithmically, one-twelth of an octave)
    # -1: don't pluck that string
    C_MAJOR = [-1, 3, 2, 0, 0, 0]
    G_MAJOR = [ 3, 2, 0, 0, 0, 3]
    A_MINOR = [ 0, 0, 2, 2, 0, 0]
    E_MINOR = [ 0, 2, 2, 0, 3, 0]

    def __init__(self, options):

        self.options = options

        self.strings = [
            # arguments are:
            # - string number
            # - octave
            # - semitone
            GuitarString(0, 2, 4,  options),   # E2
            GuitarString(1, 2, 9,  options),   # A2
            GuitarString(2, 3, 2,  options),   # D3
            GuitarString(3, 3, 7,  options),   # G3
            GuitarString(4, 3, 11, options),   # B3
            GuitarString(5, 4, 4,  options)    # E4
        ]

    def strum_chord(self, time, downstroke, velocity, chord):
        if downstroke:
            pluckOrder = [0, 1, 2, 3, 4, 5]
        else:
            pluckOrder = [5, 4, 3, 2, 1, 0]

        note_buffers = []

        for i in range(6):
            stringNumber = pluckOrder[i]
            if chord[stringNumber] != -1:
                # buffer_result = self.strings[stringNumber].pluck(time, velocity, chord[stringNumber])
                buffer_result = self.strings[stringNumber].pluck(0, velocity, chord[stringNumber])
                note_buffers.append(buffer_result)

            time += np.random.rand() / 128  # Do we want rand or randn? Or rand() centered at 0?

        as_strided = np.lib.stride_tricks.as_strided
        all_notes = np.concatenate(note_buffers)

        # overlap = 39000  # For 1-second samples
        overlap = 78000  # For 2-second samples
        added_notes = note_buffers[0]
        for i in range(1, len(note_buffers)):
            added_notes = np.concatenate([added_notes[:-overlap], added_notes[-overlap:] + note_buffers[i][:overlap], note_buffers[i][overlap:]])

        # ipdb.set_trace()
        # added_notes = as_strided(all_notes, (6, 100), all_notes.strides * 2)

        # return np.concatenate(note_buffers)
        return added_notes
    
    
    def note(self, velocity, stringNumber, tab):
        return self.strings[stringNumber].pluck(0, velocity, tab)


    def set_mode(self, mode):
        for string in self.strings:
            string.mode = mode
