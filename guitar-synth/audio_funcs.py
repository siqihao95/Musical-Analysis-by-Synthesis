import pdb
#import ipdb
import math

import numpy as np


class AudioFunctionWrapper:

    def __init__(self):
        self.heap = None

    def initAsm(self, heap_size):
        # rounded_heap_size = getNextValidFloat32HeapLength(heap_size)
        # self.heap = np.zeros(rounded_heap_size)
        self.heap = np.zeros(heap_size)

        # self.audio_funcs = audio_functions(window, foreignFunctions, heapBuffer)
        self.audio_funcs = audio_functions(self.heap)


    def pluckDecayedSine(self, buffer, sample_rate, hz, velocity, decay_factor):

        required_heap_size = len(buffer)

        self.initAsm(required_heap_size)

        heapOffsets = { 'targetStart': 0,
                        'targetEnd': len(buffer) - 1 }

        self.audio_funcs['renderDecayedSine'](heapOffsets['targetStart'],
                                              heapOffsets['targetEnd'],
                                              sample_rate,
                                              hz,
                                              velocity,
                                              decay_factor)

        return self.heap


    def pluck(self, buffer, seedNoise, sample_rate, hz, smoothingFactor,
              velocity, options, acousticLocation):

        required_heap_size = len(seedNoise) + len(buffer)

        self.initAsm(required_heap_size)

        for i in range(len(seedNoise)):
            self.heap[i] = seedNoise[i]

        # ipdb.set_trace()

        heapOffsets = { 'seedStart': 0,
                        'seedEnd': len(seedNoise) - 1,
                        'targetStart': len(seedNoise),
                        'targetEnd': len(seedNoise) + len(buffer) - 1
                      }

        self.audio_funcs['renderKarplusStrong'](heapOffsets['seedStart'],
                                                heapOffsets['seedEnd'],
                                                heapOffsets['targetStart'],
                                                heapOffsets['targetEnd'],
                                                sample_rate,
                                                hz,
                                                velocity,
                                                smoothingFactor,
                                                options.string_tension,
                                                options.pluck_damping,
                                                options.pluck_damping_variation,
                                                options.character_variation)

        if options.body == "simple":
            self.audio_funcs['resonate'](heapOffsets['targetStart'], heapOffsets['targetEnd'])

        self.audio_funcs['fadeTails'](heapOffsets['targetStart'], heapOffsets['targetEnd'] - heapOffsets['targetStart'] + 1)

        # string.acousticLocation is set individually for each string such that
        # the lowest note has a value of -1 and the highest +1

        # stereoSpread = options.stereoSpread * acousticLocation

        # for negative stereoSpreads, the note is pushed to the left
        # for positive stereoSpreads, the note is pushed to the right

        return self.heap[heapOffsets['targetStart']:]


# the byte length must be 2^n for n in [12, 24],
# or for bigger heaps, 2^24 * n for n >= 1
# def getNextValidFloat32HeapLength(desiredLengthFloats):
#     desiredLengthBytes = desiredLengthFloats << 2

#     if desiredLengthBytes <= math.pow(2, 12):
#         heapLengthBytes = math.pow(2, 12)
#     elif (desiredLengthBytes < math.pow(2, 24)):
#         heapLengthBytes = math.pow(2, math.ceil(math.log2(desiredLengthBytes)))
#     else:
#         print("Heap length greater than 2^24 bytes not implemented")

#     return int(heapLengthBytes)


def audio_functions(heapBuffer):

    heap = heapBuffer

    # simple discrete-time low-pass filter from Wikipedia
    def lowPass(lastOutput, currentInput, smoothingFactor):
        currentOutput = smoothingFactor * currentInput + (1.0 - smoothingFactor) * lastOutput
        return currentOutput


    # simple discrete-time high-pass filter from Wikipedia
    def highPass(lastOutput, lastInput, currentInput, smoothingFactor):
        currentOutput = smoothingFactor * lastOutput + smoothingFactor * (currentInput - lastInput)
        return currentOutput


    def resonate(heapStart, heapEnd):
        # explicitly initialise all variables so types are declared
        r00 = 0.0
        f00 = 0.0
        r10 = 0.0
        f10 = 0.0
        f0 = 0.0
        c0 = 0.0
        c1 = 0.0
        r0 = 0.0
        r1 = 0.0
        i = 0

        resonatedSample = 0.0
        resonatedSamplePostHighPass = 0.0
        # by making the smoothing factor large, we make the cutoff
        # frequency very low, acting as just an offset remover
        highPassSmoothingFactor = 0.999
        lastOutput = 0.0
        lastInput = 0.0

        c0 = 2.0 * np.sin(np.pi * 3.4375 / 44100.0)
        c1 = 2.0 * np.sin(np.pi * 6.124928687214833 / 44100.0)
        r0 = 0.98
        r1 = 0.98

        # This for loop replaced by a while loop
        # for (i = heapStart << 2; (i|0) <= (heapEnd << 2); i = (i + 4)|0):
        i = heapStart << 2

        while i <= heapEnd << 2:
            r00 = r00 * r0
            r00 = r00 + (f0 - f00) * c0
            f00 = f00 + r00
            f00 = f00 - f00 * f00 * f00 * 0.166666666666666
            r10 = r10 * r1
            r10 = r10 + (f0 - f10) * c1
            print(" 156 f10: %.6f, r10: %.6f, " % (f10, r10))
            f10 = f10 + r10
            print(" 157 f10 %.6f" % f10)
            f10 = f10 - f10 * f10 * f10 * 0.166666666666666
            f0 = heap[i >> 2]
            resonatedSample = f0 + (f00 + f10) * 2.0

            # I'm not sure why, but the resonating process plays
            # havok with the DC offset - it jumps around everywhere.
            # We put it back to zero DC offset by adding a high-pass
            # filter with a super low cutoff frequency.
            resonatedSamplePostHighPass = highPass(lastOutput, lastInput, resonatedSample, highPassSmoothingFactor)
            heap[i >> 2] = resonatedSamplePostHighPass
            lastOutput = resonatedSamplePostHighPass
            lastInput = resonatedSample

            i = i + 4


    # apply a fade envelope to the end of a buffer
    # to make it end at zero ampltiude
    # (to avoid clicks heard when sample otherwise suddenly
    #  cuts off)
    def fadeTails(heapStart, length):
        heapEnd = 0
        tailProportion = 0.0
        tailSamples = 0
        tailSamplesStart = 0
        i = 0
        samplesThroughTail = 0
        proportionThroughTail = 0.0
        gain = 0.0

        tailProportion = 0.1
        tailSamples = int(np.floor(length * tailProportion))
        tailSamplesStart = heapStart + length - tailSamples

        heapEnd = heapStart + length
        i = tailSamplesStart << 2
        samplesThroughTail = 0

        # for (i = tailSamplesStart << 2, samplesThroughTail = 0; (i|0) < (heapEnd << 2); i = (i + 4)|0, samplesThroughTail = (samplesThroughTail+1)|0):
        while i < heapEnd << 2:
            proportionThroughTail = samplesThroughTail / tailSamples
            gain = 1.0 - proportionThroughTail
            heap[i >> 2] = heap[i >> 2] * gain

            i += 4
            samplesThroughTail += 1


    # the "smoothing factor" parameter is the coefficient
    # used on the terms in the main low-pass filter in the
    # Karplus-Strong loop
    def renderKarplusStrong(seedNoiseStart,
                            seedNoiseEnd,
                            targetArrayStart,
                            targetArrayEnd,
                            sample_rate, hz, velocity,
                            smoothingFactor, stringTension,
                            pluckDamping,
                            pluckDampingVariation,
                            characterVariation):

        period = 0.0
        periodSamples = 0
        sampleCount = 0
        lastOutputSample = 0.0
        curInputSample = 0.0
        noiseSample = 0.0
        skipSamplesFromTension = 0
        curOutputSample = 0.0
        pluckDampingMin = 0.0
        pluckDampingMax = 0.0
        pluckDampingVariationMin = 0.0
        pluckDampingVariationMax = 0.0
        pluckDampingVariationDifference = 0.0
        pluckDampingCoefficient = 0.0

        # the index of the heap as a whole that we get noise samples from
        heapNoiseIndexBytes = 0
        # the index of the portion of the heap that we'll be writing to
        targetIndex = 0
        # the index of the heap as a whole where we'll be writing
        heapTargetIndexBytes = 0
        # the index of the heap as a whole of the start of the last period of samples
        lastPeriodStartIndexBytes = 0
        # the index of the heap as a whole from where we'll be taking samples from the last period, after
        # having added the skip from tension
        lastPeriodInputIndexBytes = 0

        period = 1.0 / hz
        periodSamples = int(np.round(period * sample_rate))
        sampleCount = targetArrayEnd - targetArrayStart + 1

        # /*
        # |- pluckDampingMax
        # |
        # |               | - pluckDampingVariationMax         | -
        # |               | (pluckDampingMax - pluckDamping) * |
        # |               | pluckDampingVariation              | pluckDamping
        # |- pluckDamping | -                                  | Variation
        # |               | (pluckDamping - pluckDampingMin) * | Difference
        # |               | pluckDampingVariation              |
        # |               | - pluckDampingVariationMin         | -
        # |
        # |- pluckDampingMin
        # */
        pluckDampingMin = 0.1
        pluckDampingMax = 0.9
        pluckDampingVariationMin = pluckDamping - (pluckDamping - pluckDampingMin) * pluckDampingVariation
        pluckDampingVariationMax = pluckDamping + (pluckDampingMax - pluckDamping) * pluckDampingVariation
        pluckDampingVariationDifference = pluckDampingVariationMax - pluckDampingVariationMin
        pluckDampingCoefficient = pluckDampingVariationMin + np.random.rand() * pluckDampingVariationDifference

        # for (targetIndex = 0; (targetIndex|0) < (sampleCount|0); targetIndex = (targetIndex + 1)|0):
        targetIndex = 0
        while targetIndex < sampleCount:

            heapTargetIndexBytes = (targetArrayStart + targetIndex) << 2

            if targetIndex < periodSamples:
                # for the first period, feed in noise remember, heap index has to be bytes...
                heapNoiseIndexBytes = (seedNoiseStart + targetIndex) << 2
                noiseSample = heap[heapNoiseIndexBytes >> 2]
                # create room for character variation noise
                noiseSample = noiseSample * (1.0 - characterVariation)
                # add character variation
                noiseSample = noiseSample + characterVariation * (-1.0 + 2.0 * np.random.rand())
                # also velocity
                noiseSample = noiseSample * velocity
                # by varying 'pluck damping', we can control the spectral content of the input noise
                curInputSample = lowPass(curInputSample, noiseSample, pluckDampingCoefficient)
            elif stringTension != 1.0:
                # for subsequent periods, feed in the output from about one period ago
                lastPeriodStartIndexBytes = heapTargetIndexBytes - (periodSamples << 2)
                skipSamplesFromTension = int(np.floor(stringTension * periodSamples))
                lastPeriodInputIndexBytes = lastPeriodStartIndexBytes + (skipSamplesFromTension << 2)
                curInputSample = heap[lastPeriodInputIndexBytes >> 2]
            else:
                # if stringTension == 1.0, we would be reading from the
                # same sample we were writing to
                # ordinarily, self would have the effect that only the first
                # period of noise was preserved, and the rest of the buffer
                # would be silence, but because we're reusing the heap,
                # we'd actually be reading samples from old waves
                curInputSample = 0.0

            # the current period is generated by applying a low-pass
            # filter to the last period
            curOutputSample = lowPass(lastOutputSample, curInputSample, smoothingFactor)

            heap[heapTargetIndexBytes >> 2] = curOutputSample
            lastOutputSample = curOutputSample

            targetIndex += 1


    def renderDecayedSine(targetArrayStart,
                          targetArrayEnd,
                          sample_rate, hz, velocity,
                          decay_factor):
        period = 0.0
        periodSamples = 0
        sampleCount = 0
        # the index of the portion of the heap that we'll be writing to
        targetIndex = 0
        # the index of the heap as a whole where we'll be writing
        heapTargetIndexBytes = 0

        time = 0.0

        period = 1.0 / hz
        periodSamples = np.round(period * sample_rate)
        sampleCount = targetArrayEnd - targetArrayStart + 1

        # for (targetIndex = 0; (targetIndex|0) < (sampleCount|0); targetIndex = (targetIndex + 1)|0):
        targetIndex = 0
        while targetIndex < sampleCount:

            heapTargetIndexBytes = (targetArrayStart + targetIndex) << 2

            time = targetIndex / sample_rate
            heap[heapTargetIndexBytes >> 2] = velocity * math.pow(2.0, -decay_factor*time) * np.sin(2.0 * np.pi * hz * time)

            targetIndex += 1

    return { 'renderKarplusStrong': renderKarplusStrong,
             'renderDecayedSine': renderDecayedSine,
             'fadeTails': fadeTails,
             'resonate': resonate }
