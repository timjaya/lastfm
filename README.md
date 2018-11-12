# Last.Fm Music Personalization and Recommmender System

This project aims to create a personalization/recommender system using the Last.fm Music dataset (http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html).

## Data
You can download the full dataset (~1.64 GB) here: 
https://www.dropbox.com/s/49qekc2surkgamu/lastfm-dataset-360K.zip?dl=0

For this project, we worked with a subset of the dataset that contained a total of 9,000 users and 47,102 artists. This dataset and all relevant jupyter notebooks can be found in the 'Code' folder, with evaluate_model.ipynb being our primary markdown file. Visualizations generated from our findings can be viewed from the 'Plots' folder.

## Models
We implemented neighborhood-based (KNN) and model-based (matrix factorization) collaborative filtering algorithms to recommend items to users and then compared their resulting performance against an established baseline model. The models were mainly implemented through the use of Ben Frederickson's implicit package: https://github.com/benfred/implicit

## Metrics
The evaluation metrics used to judge each model include Recall, Non-Cumulative Discounted Gain (NDCG) and Catalog Coverage. Overall, our results suggest that an ALS-based approach is most effective in providing quality recommendations.

