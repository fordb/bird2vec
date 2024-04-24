import os
import urllib.parse

from tqdm import tqdm
import pandas as pd
from xenopy import Query

from preprocessing import clean_and_save_audio_file

def create_filepath(row):
    dir = "datasets/xeno_canto"
    name = row["en"].replace(" ", "")
    track_id = row["id"]
    audio_file = str(track_id) + ".mp3"
    return os.path.join(dir, name, audio_file)


# Pre-encode the country parameter including quotes
encoded_country = '"' + urllib.parse.quote_plus("United States") + '"'

# create query
q = Query(name="owl", cnt=encoded_country)

# retrieve metadata
metadata = q.retrieve_meta(verbose=True)["recordings"]
metadata_df = pd.DataFrame(metadata)
metadata_df["filepath"] = metadata_df.apply(create_filepath, axis=1)
metadata_df.to_csv("datasets/xeno_canto/metadata.csv", index=False)
print(metadata_df.shape)

# retrieve recordings
q.retrieve_recordings(multiprocess=True, nproc=8, attempts=10, outdir="datasets/xeno_canto/")

# clean and save audio files
audio_file_paths = metadata_df["filepath"].tolist()
for audio_file_path in tqdm(audio_file_paths):
    clean_and_save_audio_file(audio_file_path, max_peaks=5)