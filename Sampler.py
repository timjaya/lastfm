
# coding: utf-8

# In[1]:


import pandas as pd

col_names = ["user_id","artist_mbid","artist_name","plays"]
df = pd.read_csv("usersha1-artmbid-artname-plays.tsv", sep = "\t", header = None, names = col_names)

df = df.sample(n = 9000).reset_index(drop = True)

df.to_csv("lastfm_9000sample.csv")

