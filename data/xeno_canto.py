import urllib.parse

import pandas as pd
from xenopy import Query

# Pre-encode the country parameter including quotes
encoded_country = '"' + urllib.parse.quote_plus("United States") + '"'

q = Query(name="owl", cnt=encoded_country)
# retrieve metadata
metadata = q.retrieve_meta(verbose=True)["recordings"]
metadata_df = pd.DataFrame(metadata)
print(metadata_df.shape)

q.retrieve_recordings(multiprocess=True, nproc=8, attempts=10, outdir="datasets/xeno_canto/")
metadata_df.to_csv("datasets/xeno_canto/metadata.csv", index=False)