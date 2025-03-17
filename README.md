# Speech Recognition using Dynamic Time Warping (DTW)

This repository contains a Python implementation of speech recognition using **Dynamic Time Warping (DTW)** for sequence alignment. The project includes custom **MFCC feature extraction** and DTW-based template matching for spoken word classification.

## 📌 Features
- **Custom DTW Implementation**: Computes alignment paths and distances between sequences.
- **MFCC Extraction**: Implements Mel Frequency Cepstral Coefficients (MFCC) without external libraries like `librosa`.
- **Speech Recognition with DTW**: Matches test speech signals to predefined template phrases.

## 🛠 Installation
To use this project, ensure you have Python 3 installed along with the required dependencies.

```bash
pip install numpy scipy
```

## 📂 Project Structure
```
├── file.py                # Main Python script containing DTW and speech recognition functions
├── README.md                  # Project documentation
├── template/                   # Folder containing template audio files
│   ├── hey_android.wav
│   ├── hey_snapdragon.wav
│   ├── hi_lumina.wav
│   ├── hi_galaxy.wav
├── test/                       # Folder containing test audio files
│   ├── QOAF2CAT003.wav
│   ├── QOAF2CAT004.wav
│   ├── QOAF2CAT006.wav
│   ├── ...
```

## 🔧 Usage
### 1️⃣ Compute Dynamic Time Warping Distance
```python
from file import dtw

template = [1, 2, 4, 3, 2, 3]
test = [2, 3, 1, 2]

dtw_dist, dtw_path = dtw(template, test)
print(dtw_dist, dtw_path)
```
**Output:**
```
(4.0, [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 4)])
```

### 2️⃣ Speech Recognition using DTW Matching
```python
from file import dtw_match

template_dict = {
    "hey_android": "./template/hey_android.wav",
    "hey_snapdragon": "./template/hey_snapdragon.wav",
    "hi_lumina": "./template/hi_lumina.wav",
    "hi_galaxy": "./template/hi_galaxy.wav"
}

test_dir = "./test"
result = dtw_match(template_dict, test_dir)

print(result)
```
**Output:**
```
{
  "QOAF2CAT003.wav": "hi_galaxy",
  "QOAF2CAT004.wav": "hey_android",
  "QOAF2CAT006.wav": "hi_lumina",
  ...
}
```

## 📜 Function Overview
### `dtw(template, test)`
- Computes DTW distance and alignment path between two sequences.
- Uses **Euclidean distance** as the local distance metric.
- Returns a tuple `(dtw_dist, dtw_path)`, where:
  - `dtw_dist` is the minimum alignment cost.
  - `dtw_path` is a list of indices representing the best alignment.

### `custom_mfcc(audio_path, sr=22050, n_mfcc=13, n_fft=512, hop_length=128, n_mels=40, fmin=0, fmax=None)`
- Extracts **MFCC features** from an audio file.
- Uses **pre-emphasis, windowing, FFT, Mel filtering, and DCT** for feature extraction.
- Returns an array of MFCC coefficients.

### `dtw_match(template_dict, test_dir)`
- Matches test audio files to predefined template phrases using **DTW on MFCCs**.
- Inputs:
  - `template_dict`: Dictionary `{label: audio_path}` of template speech samples.
  - `test_dir`: Directory containing test audio files.
- Returns:
  - Dictionary `{test_filename: predicted_label}` mapping test files to their closest matching template.

## 🎯 Requirements & Grading Criteria
- Returns correct **DTW distance and path** for all test inputs.
- DTW-based speech recognition must **correctly classify at least 90%** of test samples.
- If the **custom MFCC function is used**, the project is tested with only `numpy` and `scipy`.


---

🔹 **Author**: Stella Siu  
🔹 **License**: MIT  
🔹 **Contact**: My Github(https://github.com/stellasiu/asrdtw2024)  

---
Feel free to contribute or report issues! 🚀
