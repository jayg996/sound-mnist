import numpy as np
import math
from dataloader import audio_file_load, mel_spectrogram, data_load
from main import model_load_and_test, methods
import os
import librosa


test_file_paths = [os.path.join('real_test', i +'.m4a') for i in  ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']]

z_normalization = True

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

x_train, y_train = data_load(split='train')
x_train = np.reshape(x_train, (x_train.shape[0], -1))
if z_normalization:
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_real = (x_real - mean) / std

for method in methods:
    model_load_and_test(method, x_train, y_train, x_real, y_real)