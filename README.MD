# Microphone Testing
The package provide an automated method for testing the quality of transcription service. The package eliminates the necessity of testing the complete process every time a change occurs on the microphone or the transcription service, e.g, by modifying or adding filters. The package output the number of times the service successfully recognized the transcription out of the audio files. All audio files must be in .wav format.

#
## Pipeline

The package load the .wav audio files in the test audio (rateAudioFolder), then for each of them, it will add a random white noise in a increasing magnitude of power that spans a standard deviation of (stdDiff) n times (iterations). Then it will add each background noise from the background noise folder (backgroundNoise) into the audio file. In each run, it will run the transcription service to check if the transcription is correct. It then will output the result of the run in form of number of correct detections overall, along with the success rate.

#
## Repo structure
### Audio:

- #### in: The folder that contains audio files to run the test on along with a text file that contains the transcription of audio files in each line. The text on the transcription file must be ordered on the order as the audio file. i.e, the first line is the transcription of the first file. The files will be ordered alphabetically.

- #### out: A folder that contains a temp file that is being used to output the resulted audio transcription added with the noise. You can safely ignore it, but it is currently necessary to the process of transcription.

- #### backgroundNoise: A folder that contains various background noises from different environments.

#

## Arguments

- -s (stdDiff): The amount of increase in the standard deviation of the generated noise. Default is 100
- -i (iterations): The number of times a white noise will be added. Default is 5
- -o (out): The relative path of the output folder. Default is "audio/out"
- -b (backgroundNoise): The relative path of the background noise folder. Default is  "audio/backgroundNoise"
- -d (debug): Boolean of logger is on. Default is False
- -p (serviceProxy): A reference to the transcription service function. Default is service in transcribe.py
- -r (rateAudioFolder): The relative path of the folder that contains the test audios.  Default is "audio/in"
- -t (matchText): The relative path of the text file that contains the transcriptions of audio. Default is "audio/backgroundNoise"