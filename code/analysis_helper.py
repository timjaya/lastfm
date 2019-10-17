from lightfm.evaluation import recall_at_k
from lightfm.evaluation import precision_at_k
from math import log
import numpy as np
import random
import scipy
from tqdm import tqdm_notebook as tqdm
from operator import itemgetter
from lightfm import cross_validation

# DATA PREPARATION
# =====================================================================

# create_sparse_matrix and calculate_sparsity are functions that allow us
# to set up the dataset in sparse matrix form.

# Create sparse matrix from dataframe object


def create_sparse_matrix(data, user_user=True):
    """
    Creates sparse matrix (csr_matrix) out of pandas dataframe.

    Parameters:
    - data: Dataframe of user/artist data
    - user_user: determines whether output will be for user-to-user or item-to-item collaborative filtering
                 if user_user is True, then rows will be items and columns will be users

    Output:
    - plays_sparse: a sparse csr_matrix

    """
    print("Creating sparse matrix...")
    # grab unique users/artist IDS
    users = list(np.sort(data.user_id.unique()))
    artists = list(data.artist_mbid.unique())
    plays = list(data.plays)

    # user-user set-up
    if (user_user == True):
        rows = data.user_id.astype('category', categories=users).cat.codes
        cols = data.artist_mbid.astype(
            'category', categories=artists).cat.codes
        plays_sparse = scipy.sparse.csr_matrix(
            (plays, (rows, cols)), shape=(
                len(users), len(artists)))

    # item-item set-up
    else:
        rows = data.artist_mbid.astype(
            'category', categories=artists).cat.codes
        cols = data.user_id.astype('category', categories=users).cat.codes
        plays_sparse = scipy.sparse.csr_matrix(
            (plays, (rows, cols)), shape=(
                len(artists), len(users)))

    return plays_sparse

def calculate_sparsity(M):
    """
    Computes sparsity of matrix

    Parameters:
    - M: matrix to be computed

    Output:
    - sparsity: float, percentage matrix that is filled with actual values

    """
    matrix_size = float(
        M.shape[0] *
        M.shape[1])  # Number of possible interactions in the matrix
    num_plays = len(M.nonzero()[0])  # Number of items interacted with
    sparsity = 100 * (1 - float(num_plays / matrix_size))
    return sparsity


# Split Train Test

# Split_train_test_per_user creates train/test/tune, making sure that a certain proportion 
# of each user is held off for the test set. Our train test is not to split the users, 
# but the interactions of each user.
# 
# split_train_test_per_user also allows for the option to create train/test with 
# k-fold cross validation.
def split_train_test_per_user(data, k, interactions=20, cross_valid=False):
    """
    Create train/test matrix by masking values and output a dictionary of test values. 
    Only mask values of users with a certain number of interactions.

    Parameters:
    - data: csr_matrix, assuming matrix is user-user (item as rows, columns as users)
    - k: int, sets how many k-folds to create
    - interactions: int, threshold to determine which users to mask i.e. users with at least x interactions are masked
    - cross_valid: boolean, used to calculate percentage of items to mask per user (total interactions/k)

    Output:
    - train: masked matrix
    - test: list of tuples of held out data ((user_idx, item_idx), plays)

    """
    random.seed(0)  # for reproducibility

    train = data.copy()  # transpose to make procedure easier/more intuitive

    test = dict()  # dict to keep track of masked user-item values

    user_count = 0
    test_list = []
    train_list = []

    if cross_valid: 
        for i in range(k):
            test_list.append(dict())
            train_list.append(train)

    for user_idx in tqdm(range(train.get_shape()[0])):

        # Get indices of interactions of this user
        nonzero_idx = train[user_idx].nonzero()

        # Only hold out users that have enough data (greater than interactions)
        if nonzero_idx[1].shape[0] >= interactions:
            user_count += 1
            # Create list of tuples: interaction index (row, col) with the
            # number of plays
            nonzero_pairs = [(
                (user_idx, item_idx), 
                train[user_idx, item_idx]
                ) for item_idx in nonzero_idx[1]]

            # Sort tuples by descending value to get top-interactions
            nonzero_sorted = sorted(
                    nonzero_pairs, key=itemgetter(1), reverse=True
                )

            # Get top interaction # values, then sample test_pct% randomly from subset
            top_values = nonzero_sorted[0:interactions]

            # Sample random number of item_indexes without replacement
            num_samples = int(np.floor(interactions / float(k)))
            if (cross_valid == False):
                samples = random.sample(top_values, num_samples)

                # Append user_idx, item_
                test[user_idx] = [pair[0][1] for pair in samples]

                # Mask the randomly chosen items of this user
                for pair in samples:
                    train[pair[0][0], pair[0][1]] = 0

            # if creating cross-validation folds, 
            else:
                for i in range(k):
                    train = train_list[i]
                    k_test = test_list[i]
                    random.shuffle(top_values)
                    samples = top_values[0:num_samples]
                    top_values = top_values[num_samples:]
                    # Append user_idx, item_
                    k_test[user_idx] = [pair[0][1] for pair in samples]
                    test_list[i] = k_test  # update test
                    # Mask the randomly chosen items of this user
                    for pair in samples:
                        train[pair[0][0], pair[0][1]] = 0
                    train.eliminate_zeros()
                    # Update train
                    train_list[i] = train
    
    if not cross_valid:
        # Convert matrix back to initial shape
        return train.T.tocsr(), test, user_count
    else:
        for i in range(k):
            train_list[i] = train_list[i].T.tocsr()
        # Convert matrix back to initial shape
        return train_list, test_list, user_count

# Calculate how many interactions are masked compared to previous dataset


def pct_masked(original, altered):
    """
    Function that computes the percentage of interactions that have been masked compared to the original data

    Parameters:
    - original: matrix of unmasked data
    - altered: matrix with masked interactions

    Output:
    - percent_masked: percentage of interactions that have been masked
    """
    altered_n = altered.nonzero()[0].shape[0]
    original_n = original.nonzero()[0].shape[0]
    percent_masked = (original_n - altered_n) / float(altered_n)
    return percent_masked


# ### Baseline Implementation
#
# Below is how we generated our baseline recommendations (taking the most
# popular artists across the entire dataset and recommending them to
# everyone)


class Baseline():
    """
    Baseline model. Take most popular artist across entire dataset.
    """

    def __init__(self, n_recs):
        self.n_recs = n_recs
        self.popular_artists_idx = None

    def fit(self, item_user):
        """
        Input: item_user, csr matrix of n_items, n_users. Calculate most popular artists.
        """
        print("Fitting baseline...")
        plays = item_user.toarray()
        # sum up total artists in this dataset
        total_plays = np.sum(plays, axis=1)

        # get index of most popular artists
        self.popular_artists_idx = (-total_plays).argsort()[:self.n_recs]

    def predict(self, X=None):
        # returns index of most popular artists
        return self.popular_artists_idx


# ## Evaluation/Metrics
#
# The following are the functions we wrote to determine the NDCG/recall of the baseline, ALS, and KNN recommendations.
#
# auto_tune_parameter is a function written to determine the best
# hyperparameters to use for a given model.

# NDCG Metrics

def zeros_list(n):
    """
    Creates a list of zeroes

    Parameters:
    - n: Number of zeroes required in list

    Output:
    - listofzeros: list of n zeroes
    """
    listofzeros = [0] * n
    return listofzeros


def dcg_at_k(scores):
    """
    Computes Discounted Cumulative Gain

    Parameters:
    - scores: scores for each user

    Output: discounted cumulative gain
    """
    assert scores
    return scores[0] + sum(sc / log(ind, 2)
                           for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


def ndcg_at_k(rec_items, holdout_items):
    """
    Computes Normalized Discounted Cumulative Gain

    Parameters:
    - rec_items: recommended k items from model
    - heldout_items: held out items

    Output:
    - ndcg: computed normalized discounted cumulative gain
    """
    assert len(rec_items) == len(holdout_items)
    idcg = dcg_at_k(sorted(holdout_items, reverse=True))
    ndcg = (dcg_at_k(rec_items) / idcg) if idcg > 0.0 else 0.0

    return ndcg


def evaluate_lightfm(model, original, train, test,
                     user_features=None, item_features=None, n_rec=20):
    """
    Calculates evaluation metrics (Precision, Recall and Coverage) for LightFM model

    Parameters:
    - model: specified model for recommender system
    - original: matrix of full interactions
    - train: matrix of training set
    - test: matrix of test set
    - user_features: a matrix of user features built from metadata
    - item_features: a matrix of item features built from metadata
    - n_rec: nu,ber of recommendations to be made

    Output:
    - coverage: catalog coverage of model specified
    - precision: precision of model specified
    - recall: recall of model specified


    """

    print("Evaluating LightFM...")

    print("Calculating Coverage...")
    catalog = []
    for user in tqdm(range(0, original.shape[0])):
        # get scores for this particular user for all items
        rec_scores = model.predict(
            user,
            np.arange(
                original.shape[1]),
            user_features=user_features,
            item_features=item_features)

        # get top k items to recommend
        rec_items = (-rec_scores).argsort()[:20]

        # calculate coverage
        # coverage calculation
        for recs in rec_items:
            if recs not in catalog:
                catalog.append(recs)

    coverage = len(catalog) / float(original.shape[1])

    print("Calculating Recall at k...")
    recall = recall_at_k(
        model,
        test,
        user_features=user_features,
        item_features=item_features,
        k=n_rec).mean()

    print("Calculating Precision at k...")
    precision = precision_at_k(
        model,
        test,
        user_features=user_features,
        item_features=item_features,
        k=n_rec).mean()

    return coverage, precision, recall

# Used to evaluate model


def evaluate(model, model_name, test, M_train, n_rec=20, liked=None):
    """
    Calculate evaluation metrics (precision@k, recall@k, NDCG@k)

    Parameters:
    - model: fitted implicit model that will perform recommendations
    - model_name: name of package for switch case
    - test: list containing tuples that are heldout for each user
    - M_train: original csr_matrix of user-item pairs (NEED TO RENAME TO M_original)
    - n_rec: how many recommendations the system outputs

    Output:
    - two numpy arrays containing precision and recall
    """
    print('Evaluating model...')

    # to store results
    ndcg = []
    catalog = []
    user_n = 0.0
    test_n = 0.0  # keep track of number of heldout items
    tp = 0.0  # true positive

    for user, holdout_items in tqdm(test.items()):

        user_n += 1
        test_n += len(holdout_items)

        # for NDCG
        predicted_items = zeros_list(n_rec)
        true_items = zeros_list(n_rec)

        # TODO: Refactor with polymorphism here instead of ifelse
        # get recommended items from models for user
        if model_name == "baseline":
            rec_items = model.predict()

        elif model_name == "implicit":
            rec_items = model.recommend(
                user,
                M_train.T.tocsr(),
                N=n_rec,
                filter_already_liked_items=True)  # returns (item_id, score)
            rec_items = [pair[0] for pair in rec_items]  # get only item_

        elif model_name == "lightfm":

            #           CUSTOM EVALUATE, NOT WORKING ONLY USED FOR COVERAGE
            # get scores for this particular user for all items
            rec_scores = model.predict(user, np.arange(M_train.shape[1]))

            # remove already liked items
            liked_idx = liked[user].nonzero()[1]

            rec_scores[liked_idx] = -float("Inf")

            # get top k items to recommend
            rec_items = (-rec_scores).argsort()[:20]

        else:
            raise ValueError(
                "Model may not be supported. Check if model name is correct.")

        # if np array change to list
        if isinstance(rec_items, np.ndarray):
            rec_items = rec_items.tolist()

        # coverage calculation
        for recs in rec_items:
            if recs not in catalog:
                catalog.append(recs)

        # index for holdout items
        i = 0

        # calculate True Positive and NDCG Placement
        for item in holdout_items:
            value = M_train[user, item]  # get plays value of this holdout item
            true_items[i] = value
            i += 1

            if item in rec_items:
                tp += 1
                # get plays value of true positive
                predicted_items[rec_items.index(item)] = value

#         print(predicted_items)
        ndcg.append(ndcg_at_k(predicted_items, true_items))

    recall = tp / test_n
    precision = tp / (n_rec * user_n)
    avg_ndcg = np.mean(ndcg)
    coverage = len(catalog) / float(M_train.shape[1])

    return coverage, precision, recall, avg_ndcg


def auto_tune_parameter(k, interactions, model, data, param1,
                        param_type="components", user_features=None, item_features=None):
    """
    Function that identifies the optimal values of parameters which maximizes performance of model

    Parameters:
    - k: Number of folds used to tune parameters
    - interactions: matrix of interactions between users and artists
    - model: specified model used in recommender system
    - data: sparse user-item matrix
    - param1: list of values to try for hyperparameter
    - param_type: name of the parameter we want to optimize;
        options are:
        - "components"
        - "learning rate"
        - "loss function"
    - user_features: parameter used for evaluating and fitting the LightFM model.
    - item_features: parameter used for evaluating and fitting the LightFM model.

    Ouput:
    - max_recall_list: a list of k tuples, one for each fold.
        each tuple is in the form (max_recall,max_first_param,max_precision,max_coverage)
        which records the best recall, and the param that achieved it,
        and the max_precision and max_coverage achieved (which may be from different param values).
    - heatmap_list: a list of k heatmaps of the recall values for the tested
        parameter (one heatmap per fold). Useful for visualizations
    """
    # Train model
    # Create list of MAX Recall depending on # params
    max_recall_list = []  # will end up being length k list of tuples of best param values
    heatmap_list = []
    train_and_tune, test = cross_validation.random_train_test_split(
        data, test_percentage=.2, random_state=None)
    train_list = []
    tune_list = []
    for i in range(k):
        trainvals, tunevals = cross_validation.random_train_test_split(
            train_and_tune, test_percentage=.2, random_state=None)
        train_list.append(trainvals)
        tune_list.append(tunevals)
    test_recall = 0
    test_first_param = param1[0]
    # create recall matrix storing for each combination of params
    for fold in range(
            k):  # For each fold; there are k-1 folds within train_and_tune
        recall_heatmap = [0 for y in range(len(param1))]
        train = train_list[fold]
        tune = tune_list[fold]
        # initialize best value of first_param for this fold
        max_first_param = param1[0]
        max_recall = 0
        max_precision = 0
        max_coverage = 0
        value1_index = 0  # index for heatmap
        print("Fitting fold number...", fold)
        for value1 in param1:
            print("Trying ", (value1))
            if param_type == "components":
                usemodel = model(
                    learning_rate=0.05,
                    no_components=value1,
                    loss='warp')
            elif param_type == "learning_rate":
                usemodel = model(
                    learning_rate=value1,
                    no_components=50,
                    loss='warp')
            elif param_type == "loss_function":
                usemodel = model(
                    learning_rate=0.05,
                    no_components=50,
                    loss=value1)

            usemodel.fit(
                train,
                user_features=user_features,
                item_features=item_features,
                epochs=25)
            coverage, precision, recall = evaluate_lightfm(
                usemodel, data, train, tune, item_features=item_features, user_features=user_features)

            print(value1_index)
            recall_heatmap[value1_index] = recall  # update heatmap
            # update maximum values
            max_precision = max(max_precision, precision)
            max_coverage = max(max_coverage, coverage)
            if recall > max_recall:
                max_recall = recall
                max_first_param = value1
            value1_index = value1_index + 1
        max_recall_list.append(
            [max_recall, max_first_param, max_precision, max_coverage])
        if max_recall > test_recall:
            print("Fold ", fold, " beat the record for recall!")
            print("New best recall is ", max_recall)
            print("New best param is ", (max_first_param))
            test_recall = max_recall
            test_first_param = max_first_param
        heatmap_list.append(recall_heatmap)
        print("end of fold---------------------------")

    # Now, test_first_param should be optimized
    if param_type == "components":
        usemodel = model(
            learning_rate=0.05,
            no_components=test_first_param,
            loss='warp')
    elif param_type == "learning_rate":
        usemodel = model(
            learning_rate=test_first_param,
            no_components=50,
            loss='warp')
    elif param_type == "loss_function":
        usemodel = model(
            learning_rate=0.05,
            no_components=50,
            loss=test_first_param)
    usemodel.fit(
        train_and_tune,
        user_features=user_features,
        item_features=item_features,
        epochs=25)
    final_coverage, final_precision, final_recall = evaluate_lightfm(
        usemodel, data, train_and_tune, test, user_features=user_features, item_features=item_features)

    print("The recall on the test set is ", final_recall,
          ", after hyperparameter optimization")
    print(
        "The precision on the test set is ",
        final_precision,
        ", after hyperparameter optimization")
    print(
        "The coverage on the test set is ",
        final_coverage,
        ", after hyperparameter optimization")

    return max_recall_list, heatmap_list
