import librosa
import os
import numpy as np
from tqdm import tqdm

fname_to_label = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}

def get_all_audio_filepaths(audio_dir):
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]

def audio_file_load(filepath, window_length=16000, fs=16000):
    """
    Audio sample generator
    """
    audio_data, _ = librosa.load(filepath, sr=fs)
    # Clip amplitude
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 1:
        print("clipping amplitude : %s" % filepath)
        audio_data /= max_amp

    audio_len = len(audio_data)
    # Pad audio to at least a single frame
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        audio_len = len(audio_data)
    # Use central part of audio
    if audio_len > window_length:
        print("audio length is too long : %s" % filepath)
        remove_length = audio_len - window_length
        left_remove = remove_length // 2
        right_remove = remove_length - left_pad

        audio_data = audio_data[left_remove: -right_remove]
        audio_len = len(audio_data)

    assert audio_len == window_length

    # Label extraction from file path
    for key, value in fname_to_label.items():
        if key in filepath:
            label = value
            break

    return audio_data, label

# mel_spectrogram
def mel_spectrogram(audio_samples, sr=16000):
    feature = librosa.feature.melspectrogram(audio_samples, sr, n_fft=2048, hop_length=1024, n_mels=80)
    return feature

def data_load(audio_dir='./sc09', data_dir='./data', split='train', window_length=16000, fs=16000):
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))
    save_path = os.path.join(data_dir, split + '.npy')
    if not os.path.exists(save_path):
        file_pahts = get_all_audio_filepaths(os.path.join(audio_dir, split))
        x = list()
        y = list()
        for i in tqdm(file_pahts):
            audio_samples, label = audio_file_load(i)
            feature = mel_spectrogram(audio_samples)
            x.append(feature)
            y.append(label)
        np.save(save_path, (x, y))
    else:
        x, y = np.load(save_path, allow_pickle=True)
        x, y = x.tolist(), y.tolist()

    x, y = np.array(x), np.array(y)
    print("%s dataset loaded" % split)
    return x, y

if __name__ == "__main__":
    x_train, y_train = data_load(split='train')
    print(x_train.shape)
    print(y_train.shape)

    x_valid, y_valid = data_load(split='valid')
    print(x_valid.shape)
    print(y_valid.shape)

    x_test, y_test = data_load(split='test')
    print(x_test.shape)
    print(y_test.shape)
