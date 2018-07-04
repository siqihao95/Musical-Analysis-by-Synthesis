#import ipdb
import math

import numpy as np

# Local imports
from audio_funcs import AudioFunctionWrapper


def generate_seed_noise(num_samples):
    return np.random.rand(num_samples) * 2 - 1


#  calculate the constant used for the low-pass filter
# used in the Karplus-Strong loop
def calculate_smoothing_factor(string, tab, options):
    if options.string_damping_calculation == "direct":
        return options.string_damping
    elif options.string_damping_calculation == "magic":
        noteNumber = (string.semitone_index + tab - 19) / 44
        return options.string_damping + math.pow(noteNumber, 0.5) * (1 - options.string_damping) * 0.5 + (1 - options.string_damping) * np.random.rand() * options.string_damping_variation


class GuitarString:
    def __init__(self, string_num, octave, semitone, options):

        self.options = options

        # work from A0 as a reference,
        # since it has a nice round frequency
        a0_hz = 27.5
        # an increase in octave by 1 doubles the frequency
        # each octave is divided into 12 semitones
        # the scale goes C0, C0#, D0, D0#, E0, F0, F0#, G0, G0#, A0, A0#, B0
        # so go back 9 semitones to get to C0
        c0_hz = a0_hz * math.pow(2, -9 / 12)
        self.basicHz = c0_hz * math.pow(2, octave + semitone / 12)

        # ipdb.set_trace()
        basic_period = 1.0 / self.basicHz
        basic_period_num_samples = int(np.round(basic_period * 40000))  # TODO: audioCtx.sample_rate == 4000 ?
        self.seed_noise = generate_seed_noise(basic_period_num_samples)

        # this is only used in a magical calculation of filter coefficients
        self.semitone_index = octave * 12 + semitone - 9

        # ranges from -1 for first string to +1 for last
        self.acoustic_location = (string_num - 2.5) * 0.4
        # self.mode = "karplus-strong"
        # self.mode = "sine"
        self.mode = options.mode
        self.func_wrapper = AudioFunctionWrapper()


    def createBuffer(self, sample_count):
        # We use 1 channel for simplicity
        return np.zeros(sample_count)


    def pluck(self, start_time, velocity, tab, freq, smoothing_factor):
        # create the buffer we're going to write into
        channels = 1
        sample_rate = 40000  # TODO: audioCtx.sample_rate == 4000 ?
        sample_count = int(2.0 * sample_rate)  # 1 second buffer

        # buffer = self.createBuffer(channels, sample_count, sample_rate)
        buffer = self.createBuffer(sample_count)

        #smoothing_factor = calculate_smoothing_factor(self, tab, self.options)
        # 'tab' represents which fret is held while plucking
        # each fret represents an increase in pitch by one semitone
        # (logarithmically, one-twelth of an octave)
        #hz = self.basicHz * math.pow(2, tab / 12)
        hz = freq

        velocity /= 4.0

        if self.mode == "karplus-strong":
            filled_buffer = self.func_wrapper.pluck(buffer,
                                                    self.seed_noise,
                                                    sample_rate,
                                                    hz,
                                                    smoothing_factor,
                                                    velocity,
                                                    self.options,
                                                    self.acoustic_location)
        elif self.mode == "sine":
            decayFactor = 8
            filled_buffer = self.func_wrapper.pluckDecayedSine(buffer,
                                                               sample_rate,
                                                               hz,
                                                               velocity,
                                                               decayFactor)

        return filled_buffer
