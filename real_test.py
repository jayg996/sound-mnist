import numpy as np
import math
from dataloader import audio_file_load, mel_spectrogram
from main import model_load_and_test, methods
import os
import librosa


test_file_paths = [os.path.join('real_test', i +'.m4a') for i in  ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']]

x = list()
y = list()
for i in test_file_paths:
    audio_samples, label = audio_file_load(i)
    feature = mel_spectrogram(audio_samples)
    x.append(feature)
    y.append(label)

x_real, y_real = np.array(x), np.array(y)
x_real = np.reshape(x_real, (x_real.shape[0], -1))
print(x_real.shape)
print(y_real.shape)

for method in methods:
    model_load_and_test(method, x_real, y_real)