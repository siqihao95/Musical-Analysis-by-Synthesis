# -*- coding: utf-8 -*-


import numpy as np
import random
import librosa
import data


class karplus_strong:
    def __init__(self, pitch, sampling_freq, stretch_factor, flag):
        """Inits the string."""
        self.pitch = pitch
        # self.starting_sample = starting_sample
        self.sampling_freq = sampling_freq
        self.stretch_factor = stretch_factor
        self.flag = flag
        self.wavetable = self.init_wavetable()
        self.current_sample = 0
        self.previous_value = 0


    def init_wavetable(self):
        """Generates a new wavetable for the string."""
        wavetable_size = int(self.sampling_freq) // int(self.pitch)
        if self.flag == 0:
            self.wavetable = np.ones(wavetable_size)
        else:
            self.wavetable = (2 * np.random.randint(0, 2, wavetable_size) - 1).astype(np.float)
        return self.wavetable


    def get_samples(self):
        """Returns samples from string."""
        samples = []
        while len(samples) < self.sampling_freq:
            if self.flag != 1:
                r = np.random.binomial(1, self.flag)
                sign = float(r == 1) * 2 - 1
                self.wavetable[self.current_sample] = sign * 0.5 * (
                self.wavetable[self.current_sample] + self.previous_value)
            else:
                d = np.random.binomial(1, 1 - 1 / self.stretch_factor)
                if d == 0:
                    self.wavetable[self.current_sample] = 0.5 * (
                    self.wavetable[self.current_sample] + self.previous_value)
            samples.append(self.wavetable[self.current_sample])
            self.previous_value = samples[-1]
            self.current_sample += 1
            self.current_sample = self.current_sample % self.wavetable.size
        return np.array(samples)


def main():
    data.create_datasets()
    train_data, test_data, eval_data = data.read_dataset()
    pitch, sampling_freq, stretch_factor, flag = eval_data['parameters'][:,0]
    string = karplus_strong(pitch, 2 * sampling_freq, stretch_factor, 1)
    sample = string.get_samples()
    librosa.output.write_wav('karplus_strong_output.wav', sample, int(sampling_freq))
  
    
if __name__ == "__main__":
    main()


