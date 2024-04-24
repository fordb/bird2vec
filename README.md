# bird2vec
Fine tune [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) for classifying bird audio.

## Install
* Create a virtual environment with `virtualenv venv`
* Activate virtual environment with `source venv/bin/activate`
* Install required packages with `pip install -r requirements.txt`

## Data
Data is from xeno-canto.org, and is focused only on recordings from owls in the United States.

Data is retrieved from `data/xeno_canto.py`:
* Creates a metadata file, `datasets/metadata.csv`, which contains info on all of the recordings
* Retrieves audio recordings from xeno-canto and saves as mp3 files in `datasets/xeno_canto/`
* Each bird species is its own directory. Ex: all Barred Owl recordings are stored in `datasets/xeno_canto/BarredOwl/<file_name>.mp3`
* For each audio recording, the recording is split up into N 5 second subsets via the `data/preprocessing.py:create_audio_subsets` function. This finds the highest signal to noise subsets of the audio files, splits them out, and converts them to 16K sample rate
* These cleaned audio files are saved to `datasets/xeno_canto_clean/`. These follow the same pattern, but also have subset identifiers appended. For example, `datasets/xeno_canto_clean/BarredOwl/<file_name>_sample_0.mp3`, `datasets/xeno_canto_clean/BarredOwl/<file_name>_sample_1.mp3`

## Model Training
* Model parameters are controlled in the config in `config/settings.py`
* Model training performance is logged to Weights and Biases by default