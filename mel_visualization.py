from dataloader import audio_file_load
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np


filename = os.path.join('sc09','train','Nine_6c968bd9_nohash_0.wav')
audio_data, label = audio_file_load(filename)
sr = 16000

song, _ = librosa.effects.trim(audio_data)
librosa.display.waveplot(song, sr=sr)
plt.show()

feature = librosa.feature.melspectrogram(audio_data, sr, n_fft=2048, hop_length=1024, n_mels=80)
S_DB = librosa.power_to_db(feature)
print(S_DB.shape)
librosa.display.specshow(S_DB, sr=sr, hop_length=1024, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()