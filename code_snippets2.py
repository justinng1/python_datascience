import numpy as np
import pandas as pd

# read by chunks.
filename = 'train_dataset.csv'
df = pd.read_csv(filename, chunksize=10000)

# sampling common events, while keep all rare events using chunks.
df_list = []
for df_chunk in df:
  # sample common events.
  idx = df_chunk['ind'] == 0
  common_df_samp = df_chunk[idx].sample(frac=0.05)
  
  # get all rare evetns.
  idx = df_chunk['ind'] == 1
  rare_df = df_chunk[idx]
  
  df_list.append(common_df_sample)
  df_list.append(rare_df)
  
df_sampled = pd.concat(df_list)

