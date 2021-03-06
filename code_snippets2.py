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

# getting the value counts of the entire dataset by chunks.
value_counts_all = {}
for df_chunk in df:
  val_counts_chunk = df_chunk['col1'].value_counts()
  for val in val_counts_chunk.index:
    if val in value_counts_all:
      value_counts_all[val] = value_counts_all[val] + val_counts_chunk[val]
    else:
      value_counts_all[val] = val_counts_chunk[val]

