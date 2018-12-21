# Last.Fm Music Personalization and Recommmender System
This project aims to create a personalization/recommender system using the Last.fm Music dataset (http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html).

## Data
You can download the full dataset (~1.64 GB) here: 
https://www.dropbox.com/s/49qekc2surkgamu/lastfm-dataset-360K.zip?dl=0

For this project, we worked with a subset of the dataset that contained a total of 9,000 users and 47,102 artists. This dataset and all relevant jupyter notebooks can be found in the 'Code' folder, with Main.ipynb being our primary markdown file. Visualizations generated from our findings can be viewed from the 'Plots' folder.

We also worked with metadata that contained contextual information for each user, namely age, gender and country. Where possible, we utilized this information as additional features for our recommender system.

## Models
We implemented and evaluated two main models to recommend artists to Last.fm users, one driven by Alternating Least Squares (ALS) and the other by Factorization Machines. The nature of Factorization Machines allowed us to add in user features in the form of the metadata we possess. We also established a baseline model where the same top-k most popular artists (by total plays) are recommended to each user.

Our ALS model was implemented through Ben Frederickson's implicit package (https://github.com/benfred/implicit). Our Factorization Machines model was implemented using LightFM (https://lyst.github.io/lightfm/docs/home.html). 

## Metrics
The evaluation metrics used to judge each model include Precision, Recall and Catalog Coverage. Overall, our results suggest that an ALS-based approach is most effective in providing quality recommendations. A Factorization Machines approach, on the other hand, seemed to perform slightly worse in comparison.

