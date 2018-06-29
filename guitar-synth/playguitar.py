import sys
import ipdb
import argparse

import librosa

import matplotlib.pyplot as plt

# Local imports
import sequencer
from guitar import Guitar

sys.path.insert(0, '../models')
import cqt_transform


# Settings
parser = argparse.ArgumentParser(description='Karplus-Strong Synthesizer')
parser.add_argument('--character-variation', type=float, default=0.5,
                    help='Character variation (default: 0.5)')
parser.add_argument('--string-damping', type=float, default=0.5,
                    help='String damping (default: 0.5)')
parser.add_argument('--string-damping-variation', type=float, default=0.25,
                    help='String damping variation (default: 0.5)')
parser.add_argument('--pluck-damping', type=float, default=0.5,
                    help='Pluck damping (default: 0.5)')
parser.add_argument('--pluck-damping-variation', type=float, default=0.25,
                    help='Pluck damping variation (default: 0.25)')
parser.add_argument('--string-tension', type=float, default=0.1,
                    help='String tension (default: 0.0)')
parser.add_argument('--stereo-spread', type=float, default=0.2,
                    help='Stereo spread (default: 0.2)')
parser.add_argument('--string-damping-calculation', type=str, default='magic',
                    help='Stereo spread (default: magic)')
parser.add_argument('--body', type=str, default='simple',
                    help='Stereo spread (default: simple)')
parser.add_argument('--mode', type=str, default='karplus-strong', choices=['karplus-strong', 'sine'],
                    help='Which type of audio to generate.')
args = parser.parse_args()


if __name__ == '__main__':

    guitar = Guitar(options=args)
    audio_buffer = sequencer.play_guitar(guitar)
    cqt = cqt_transform.compute_cqt_spec(audio_buffer, sr=40000, hop_length=256)
    librosa.output.write_wav('guitar_output.wav', audio_buffer, 40000)  # The 40000 is the sampling frequency
