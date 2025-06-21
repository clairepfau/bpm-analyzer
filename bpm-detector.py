import librosa
import numpy as np

y, sr = librosa.load('test_sound_for_bpm_option1_170bpm.wav')

print(f"Sample Rate: {sr} Hz") #print the sample rate in Hz
print(f"Audio length: {len(y)} samples ---> {len(y)/sr} seconds") #samples / sample rate

onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
#onset strength is an array of frames (time intervals of 512 seconds) which relative strength score of each frame

autocorr = librosa.autocorrelate(onset_strength)

bpm_candidates = librosa.util.peak_pick(autocorr, #parameter numbers suggested by Claude
                                        pre_max=3,
                                        post_max=3,
                                        pre_avg=3,
                                        post_avg=5,
                                        delta=0.5,
                                        wait=10)
#bpm_candidates is an array of indexes of frames where the onset strength was decided to be within the criteria specified by the peak_pick function
print(bpm_candidates)
#now, a new array must be made that returns the time of each beat candidate rather than the frame index.
#the default hop_value for the frames is 512, meaning that each frame is 512 samples long.
bpm_candidates_times = bpm_candidates * (512 / sr)
print(bpm_candidates_times)

#first bpm candidate example (claude is calling it peaks)

first_candidate = bpm_candidates_times[1]
print(first_candidate)

intervals = np.diff(bpm_candidates_times)
print(intervals)

t_avg = np.mean(intervals) #average time in between each beat
print(t_avg) #in between each waveform spike is about t_avg seconds

bpm = 60 / t_avg # this finds the number of times a spike in the waveform, significant enough to be comporable to other spikes in the waveform in a consistent manner, would be present in one minute, hence 'beats per minute'. (waveform spikes/average time length between waveform spikes) = quantity of waveform spikes in a 60 second time window. beats per minute

print(f"The bpm is: {bpm}")