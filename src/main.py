import numpy as np
from scipy.io import wavfile
from transcribe import service
from scipy.io.wavfile import write
from os import listdir
from math import ceil
import logging
import argparse

#============== DEFAULTS
BACKGROUND_NOISE_FOLDER = "audio/backgroundNoise"
TEST_FOLDER = "audio/in"
TEST_EXPECTATION_DOC = "audio/in/e.txt"
WRITE_TEMP_AUDIO = "audio/out"
WRITE_TEMP_AUDIO_FILE = "temp.wav"
DEFAULT_RANDOM_NOISE_ITER = 5
DEFAULT_STD_DIFF = 100
#==============

#============== ARG PARSER

parser = argparse.ArgumentParser(
                    prog='Microphone rating')
parser.add_argument('-s', '--stdDiff', type=float, default=DEFAULT_STD_DIFF)
parser.add_argument('-i', '--iterations', type=int, default=DEFAULT_RANDOM_NOISE_ITER)
parser.add_argument('-o', '--out', type=str, default=WRITE_TEMP_AUDIO)
parser.add_argument('-b', '--backgroundNoise', type=str, default=BACKGROUND_NOISE_FOLDER)
parser.add_argument('-d', '--debug', type=bool, default=False)
parser.add_argument('-p', '--serviceProxy', default=service)
parser.add_argument('-r', '--rateAudioFolder', type=str, default=TEST_FOLDER)
parser.add_argument('-t', '--matchText', type=str, default=TEST_EXPECTATION_DOC)
args = parser.parse_args()
#============== 

#============== LOGGER

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()
#==============


def random_noise(std, num_samples):
    """
    Generate a random noise
    """
    return (np.random.normal(0, std, size=num_samples)).astype("float32")


def scoreTranscriptionDiff(res, target):
    return 1 if target in res else 0


def loadBackgroundNoise(backgroundNoiseDir):
    return [f"{backgroundNoiseDir}/{f}" for f in listdir(backgroundNoiseDir) if f[-4:] == ".wav"]


def wr(data, rate, out):
    scaled = np.int16(data / np.max(np.abs(data)) * 32768)
    write(f"{out}/{WRITE_TEMP_AUDIO_FILE}", rate, scaled)

def adjustBackgroundNoiseLength(noiseArray, requiredLength):
    if len(noiseArray) >= requiredLength:
        return noiseArray[:requiredLength]
    
    newNoise = noiseArray
    requiredCopies = ceil(requiredCopies / len(noiseArray))
    
    for _ in range(requiredCopies):
        if len(newNoise) * 2 > requiredLength:
            newNoise = np.array(newNoise, newNoise)
        else:
            newNoise = np.array(newNoise, newNoise[:requiredLength - len(newNoise)])


def rate(data, transcriptionRef, stdDiff, iterations, out, backgroundNoiseDir, debug):
    """
    Take a list of pairs of audio (absolute path to a wav file) and their transcription
    Take a reference to transcription function

    Generate white noise of varying intensities
    Generate background noise of varying intensities and envs

    for each pair, score against the generated white and background noise, then score them
    """


    """
    For each pair:
        generate a list of increasing random noise according to the length of the audio
        add them
        run the transcriptionRef
        score against the data

        do it again with the background noise
    """
    score = 0
    run = 0

    if debug:
        logger.info("START NOISE TESTING")

    for i in data:
        samplerate, audioArray = wavfile.read(i[0])
        audioArray = audioArray.astype("float32")
        
        for noiseIdx in range(iterations):
            if debug:
                logger.info(f"TESTING WITH WHITE NOISE of std of {std_diff * (noiseIdx + 1)}")

            noise = random_noise(stdDiff * (noiseIdx + 1), len(audioArray))
            testedSample = noise + audioArray
            wr(testedSample, samplerate, out)
            res = transcriptionRef(f"{out}/{WRITE_TEMP_AUDIO_FILE}")
            score += scoreTranscriptionDiff(res, i[1])
            run += 1
            if debug:
                logger.info(f"SCORE {score}")

        if debug:
            logger.info("START BACKGROUND NOISE")

        for backgroundNoise in loadBackgroundNoise(backgroundNoiseDir):
            samplerate, backgroundNoiseArray = wavfile.read(backgroundNoise)
            backgroundNoiseArray = backgroundNoiseArray.astype("float32")

            testedSample = backgroundNoiseArray[:len(audioArray)] + audioArray
            wr(testedSample, samplerate, out)
            res = transcriptionRef(f"{out}/{WRITE_TEMP_AUDIO_FILE}")
            score += scoreTranscriptionDiff(res, i[1])
            run += 1
            if debug:
                logger.info(f"SCORE {score}")

    
    return score, run. score/run


# TARGET AUDIO LIST
# Used service

def getTestingData(audioFolder, matchingDoc):
    audios = [f"{audioFolder}/{f}" for f in listdir(audioFolder) if f[-4:] == ".wav"]
    text = []
    with open(matchingDoc, 'r') as file:
        text = file.read().split("\n")

    if len(audios) != len(text):
        logger.fatal("The number of audios doesn't match the text matches. Exit the app")
        import sys
        sys.exit()
    

    return zip(audios.sort(), text)

a = rate(getTestingData(args.rateAudioFolder, args.matchText), args.serviceProxy, args.stdDiff, args.iterations, args.out, args.backgroundNoise, args.debug)
print(f"SCORE: {a[0]} OUT OF {a[1]}, success rate of {a[2]}")