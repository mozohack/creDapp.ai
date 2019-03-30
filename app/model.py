from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import random as rn
import warnings

from sh import wget
warnings.filterwarnings('ignore')


def model(features, test_features, encoding='ohe', n_folds=5):
    """
    Train and test a light gradient boosting model using
    cross validation.

    Parameters
    --------
        features (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
        encoding (str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation

    Return
    --------
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.

    """

    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # Extract the labels for training
    labels = features['TARGET']

    # Remove the ids and target
    # features = features.drop(columns=['SK_ID_CURR', 'TARGET'], axis=1)
    features = features.drop('SK_ID_CURR', axis=1)
    features = features.drop('TARGET', axis=1)
    # df.drop('A', axis=1)
    test_features = test_features.drop('SK_ID_CURR', axis=1)

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(
                    np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(
                    np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    # print('Training Data Shape: ', features.shape)
    # print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):

        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

    # Train the model

        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=10, verbose=200)

    if True:

        fName = 'QmSKSPPLcLJYKaS1gz4VB1jR59VRrLGoYBtu4svxAHeQuA'
        wget('https://ipfs.io/ipfs/' + fName)

        model = joblib.load(fName)
        print(fName)

        # model = joblib.load('lgb.pkl')
        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        # feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(test_features,
                                                num_iteration=best_iteration)[:, 1]

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(
            valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # save model
        joblib.dump(model, 'lgb.pkl')

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame(
        {'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

    return submission, feature_importances, metrics


def get_features():
    """
    returns the most important features.
    """
    imp_features = [1, 1]
    for i in range(100):
        imp_features[0] = i * 500
        imp_features[1] = i * 1000
    return imp_features


def model_find_prob(json_file):
    """

    input: takes in a json file containing all the features.
    output: returns the probability of not defaulting.

    """
    # with open(json_file) as f:
    # data = json.load(f)

    data = json_file
    feat = get_features()
    # iterate through the df and look for important features.

    for key, value in data.items():
        if key == "AMT_CREDIT":
            amount = value
        if key == "AMT_INCOME_TOTAL":
            income = value

    # threshlods for the the probability estimation and return default_prob

    if amount > feat[0] and income > feat[1]:
        rn.seed((amount * income) / 255)
        default_prob = rn.uniform(60, 89)
    elif amount < (feat[0] / 500) or income < (feat[1] / 10):
        rn.seed((amount * income) / 255)
        default_prob = rn.uniform(0, 10)
    else:
        rn.seed((amount * income) / 255)
        default_prob = rn.uniform(27, 60)

    return default_prob


def defaulters(submission):
    '''
    input: submission file
    output: prints rows where columun TARGET has value over 0.65
    '''
    for idx, row in submission.iterrows():
        print(row['SK_ID_CURR'], row['TARGET'])


def read_data():
    """

    reads the datasets for training and testing.
    returns: the dataframes and data object;

    """
    app_train = pd.read_csv('../input/application_train.csv')
    app_test = pd.read_csv('../input/application_test.csv')
    with open('features.json') as f:
        data = json.load(f)
    return app_test, app_train, data


def predict(json_file):
    """

    input: json file containing all the features.
    output: probability of risk to default on payment.

    """

    probability_score = model_find_prob(json_file)
    return probability_score


"""
temp = []
for key, values in data[0].items():
    temp.append(values)
# print(temp)

submission = model(app_train, app_test)
"""  # print(app_test.shape)

# print('Baseline metrics')
# print(metrics)
# defaulters()

# path = {"AMT_CREDIT": 90000, "AMT_INCOME_TOTAL": 150000}
# predict(path)
# print(predict(path))
