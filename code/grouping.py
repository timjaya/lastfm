from analysis_helper import *
import scipy
import numpy as np


def active_users(data, n=5):
    """
    Identifies and returns CSR matrices representing groups of users by activity,
    determined by the number of plays they have logged

    Parameters:
    - data: csr_matrix (no_users, no_items) with users as rows and items as columns
    - n: number of distinct groups (classified by level of activity) to output

    Output:
    - groups: an array of CSR matrices representing the users in each activity group
    """
    groups = [[] for i in range(n)]
    plays_sum = np.asarray(data.sum(axis=1))
    group_range = [np.percentile(
        plays_sum, [(100 / n) * i, (100 / n) * (i + 1)]) for i in range(n)]

    for user in range(data.shape[0]):
        for j in range(0, n):
            if group_range[j][0] <= plays_sum[user] <= group_range[j][1]:
                groups[j].append(data.getrow(user).toarray()[0])

    groups = [scipy.sparse.csr_matrix(i) for i in groups]

    return groups


def diverse_users(data, n=5):
    """
    Identifies and returns CSR matrices representing groups of users by diversity,
    determined by the diversity of artists listened to.

    Diversity of each user is calculated by the spread of listens across all artists played.

    Parameters:
    - data: csr_matrix (no_users, no_items) with users as rows and items as columns
    - n: number of distinct groups (classified by level of diversity) to output

    Output:
    - groups: an array of CSR matrices representing the users in each diversity group
    """
    groups = [[] for i in range(n)]
    plays_norm = []
    for user in range(0, data.shape[0]):
        row = data.getrow(user).toarray()[0]
        plays_norm.append(sum(np.interp(row, (row.min(), row.max()), (0, 1))))
    group_range = [np.percentile(
        plays_norm, [(100 / n) * i, (100 / n) * (i + 1)]) for i in range(n)]

    for user in range(data.shape[0]):
        for j in range(0, n):
            if group_range[j][0] <= plays_norm[user] <= group_range[j][1]:
                groups[j].append(data.getrow(user).toarray()[0])

    groups = [scipy.sparse.csr_matrix(i) for i in groups]

    return groups


# In[4]:


def mainstream_users(data, top_artists=20, n=5):
    """
    Identifies and returns CSR matrices representing the most/least mainstream users,
    determined by their weighted listens of popular artists

    Popularity of each artist is calculated by the total plays across all users. Indicator for how
    mainstream a user is is computed by the weighted total of listens to the top n artists.

    Parameters:
    - data: csr_matrix (no_users, no_items) with users as rows and items as columns
    - top_artists: benchmark for an artist to be considered 'popular' e.g. top_artists = 20
    means that top 20 artists with most listens are considered mainstream
    - n: number of distinct groups (classified by level of diversity) to output

    Output:
    - groups: an array of CSR matrices representing the users in each mainstream level group
    """
    groups = [[] for i in range(n)]
    artist_sum = np.asarray(data.sum(axis=0))[0]
    popular = artist_sum.argsort()[-top_artists:]
    weights = list(range(1, top_artists + 1))
    score_popular = []

    for user in range(data.shape[0]):
        row = data.getrow(user).toarray()[0]
        row_scaled = np.interp(row, (row.min(), row.max()), (0, 1))
        score_popular.append(np.dot(row_scaled[popular], weights))
    group_range = [np.percentile(
        score_popular, [(100 / n) * i, (100 / n) * (i + 1)]) for i in range(n)]

    for user in range(data.shape[0]):
        for j in range(0, n):
            if group_range[j][0] <= score_popular[user] <= group_range[j][1]:
                groups[j].append(data.getrow(user).toarray()[0])

    groups = [scipy.sparse.csr_matrix(i) for i in groups]

    return groups
