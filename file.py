import numpy as np
import os
import scipy
import wave

def dtw(template, test):
    template = np.array(template)
    test = np.array(test)
    template_length = len(template)
    test_length = len(test)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((template_length + 1, test_length + 1), np.inf) # Tried to set as 0 but it would not work, must use inf
    dtw_matrix[0, 0] = 0  # Start point is 0

    for i in range(1, template_length + 1):
        for j in range(1, test_length + 1):
            local_distance = np.sqrt(np.sum((template[i - 1] - test[j - 1]) ** 2))  # Euclidean distance
            dtw_matrix[i, j] = local_distance + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])  # Horizontal, vertical, diagonal value filling

    dtw_dist = np.round(dtw_matrix[template_length, test_length], 2)  # Distance = last value of matrix

    dtw_path = []
    i, j = template_length, test_length  # As required in the assignment instruction
    while i > 0 and j > 0:
        dtw_path.append((i, j))
        if dtw_matrix[i - 1, j - 1] <= dtw_matrix[i - 1, j] and dtw_matrix[i - 1, j - 1] <= dtw_matrix[i, j - 1]:
            i, j = i - 1, j - 1  # Diagonal prioritized
        elif dtw_matrix[i - 1, j] <= dtw_matrix[i, j - 1]:
            i -= 1  # Vertical
        else:
            j -= 1  # Horizontal

    dtw_path.reverse()  # Add the path

    return dtw_dist, dtw_path


def custom_mfcc(audio_path, sr=22050, n_mfcc=13, n_fft=512, hop_length=128, n_mels=40, fmin=0, fmax=None):
    # n_fft=512 for speech
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio = wf.readframes(num_frames)

    audio = np.frombuffer(audio, dtype=np.int16)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Sliding windows
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(audio, window_shape=n_fft)[::hop_length]
    frames = frames.copy() # So it's writable
    window = scipy.signal.get_window("hamming", n_fft, fftbins=True)  # Slightly better reserved harmonics (Vass in SS1 course)
    frames *= window

    # DFT
    fft_frames = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spectrum = np.abs(fft_frames) ** 2

    # Create Mel filter bank
    mel_filters = generate_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)

    # Apply Mel filter bank
    mel_energies = np.dot(power_spectrum, mel_filters.T)

    # Log scale
    log_mel_energies = np.log(mel_energies + 1e-10)

    # DCT
    mfccs = scipy.fft.dct(log_mel_energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfccs


def generate_mel_filterbank(sr, n_fft, n_mels, fmin, fmax): # GPT did this for me I have no idea how to do this myself :(
    if fmax is None:
        fmax = sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (np.linspace(0, 1, bin_points[i] - bin_points[i - 1]))
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (np.linspace(1, 0, bin_points[i + 1] - bin_points[i]))

    return filters


def dtw_match(template_dict, test_dir):

    # Precompute MFCCs for templates
    template_mfccs = {}
    for label, path in template_dict.items():
        template_mfccs[label] = custom_mfcc(path)

    # Initialize result dictionary
    result_dict = {}

    # Iterate over test files in the directory
    for test_file in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_file)

        # Extract MFCCs for the test file
        test_mfcc = custom_mfcc(test_path)

        # Compare with each template and find the best match
        best_matched_label = None
        smallest_distance = float("inf")
        for label, template_mfcc in template_mfccs.items():
            dtw_distance, _ = dtw(template_mfcc, test_mfcc)
            if dtw_distance < smallest_distance:
                smallest_distance = dtw_distance
                best_matched_label = label

        # Assign the best match label to the test file
        result_dict[test_file] = best_matched_label

    return result_dict


