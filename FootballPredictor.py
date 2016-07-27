import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from boruta_py import BorutaPy
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
LIGHT_PURPLE = '\033[94m'
PURPLE = '\033[95m'
END = '\033[0m'

def print_color(text, colorcode):
    print (colorcode+" {}\033[00m" .format(text))

features_df = pd.read_csv('kaggle_features_v2.csv')
labels_df = pd.read_csv('kaggle_labels_v2.csv')
features_df = features_df.drop('Unnamed: 0', 1)
labels_df = labels_df.replace('HOME', 0)
labels_df = labels_df.replace('DRAW', 1)
labels_df = labels_df.replace('AWAY', 2)

features_df = features_df.replace('right', 0)
features_df = features_df.replace('left', 1)

# Lel, what kind of words are these??!!
features_df = features_df.replace('_0', 0)
features_df = features_df.replace('es', 0)
features_df = features_df.replace('tocky', 0)
features_df = features_df.replace('o', 0)
features_df = features_df.replace('stoc', 0)
features_df = features_df.replace('y', 0)
features_df = features_df.replace('le', 0)
features_df = features_df.replace('None', 0)
features_df = features_df.replace('low', 1)
features_df = features_df.replace('ean', 2)
features_df = features_df.replace('ormal', 3)
features_df = features_df.replace('norm', 3)
features_df = features_df.replace('medium', 4)
features_df = features_df.replace('high', 5)

features_df = features_df.fillna(0)

features_df = features_df.sort_values(by=['date', 'match_id'])
labels_df = labels_df.sort_values(by=['date', 'match_id'])

train_length = int(len(features_df) * 0.9)
test_length = len(features_df) - train_length

train_features_df = features_df.head(train_length)
train_labels_df = labels_df.head(train_length)
test_features_df = features_df.tail(test_length)
test_labels_df = labels_df.tail(test_length)

print(np.bincount(labels_df['result']))

def RF_feature_selection(features, labels, n_features):
    rf = RandomForestClassifier(n_estimators=2500, class_weight='auto', n_jobs=-1, bootstrap=True)
    rf.fit(features, labels)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    feature_importance = []

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(pd.DataFrame(features).shape[1]):
        print("%3d. feature %-25s [%2d] (%9f)" % (
                f + 1, features.columns[indices[f]], indices[f], importances[indices[f]]))
        feature_importance.append(indices[f])

    return [features.columns[x] for x in feature_importance[:n_features]]

def boruta_py_feature_selection(features, labels, column_names, verbose=False, percentile=85, alpha=0.05):
    """
    :param alpha:
    :param percentile:
    :param features: dataframe of the features
    :param labels: vector containing the correct labels
    :param column_names: The column names of the dataframe of the features
    :param verbose:Whether to print info about the feature importance or not
    :return: vector containing the indices of the most important features (as column number)
    """
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto')
    feat_selector = BorutaPy(rf, n_estimators='auto', perc=percentile, alpha=alpha, verbose=1)
    feat_selector.fit(features, labels)
    if verbose:
        print("\n\n\n\n")
        # check selected features
        # print feat_selector.support_

        # check ranking of features
        print("Ranking features: ")
        print(feat_selector.ranking_)

        # call transform() on X to filter it down to selected features
        # X_filtered = feat_selector.transform(features)
        # print X_filtered
        print("Most important features (%2d):" % sum(feat_selector.support_))
    important_features = []
    print()
    for i in range(len(feat_selector.support_)):
        if feat_selector.support_[i]:
            if verbose: print("feature %2d: %-25s" % (i, column_names[i]))
            important_features.append(i)
    return important_features

# features = boruta_py_feature_selection(train_features_df.drop('home_team', 1).drop('away_team', 1).drop('date', 1).values,
#                                        train_labels_df['result'].tolist(),
#                                        train_features_df.drop('home_team', 1).drop('away_team', 1).drop('date', 1).columns,
#                                        verbose=True)
features = RF_feature_selection(train_features_df.drop('home_team', 1).drop('away_team', 1).drop('date', 1),
                                train_labels_df['result'], 500)

svm = SVC(C=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001,
          cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
clf = RandomForestClassifier(n_estimators=750, n_jobs=-1)
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                        class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                        verbose=0, warm_start=False, n_jobs=1)
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_depth=3, init=None, random_state=None, max_features=None, verbose=0,
                                 max_leaf_nodes=None, warm_start=False, presort='auto')

svm.fit(train_features_df[features], train_labels_df['result'])
clf.fit(train_features_df[features], train_labels_df['result'])
lr.fit(train_features_df[features], train_labels_df['result'])
gbc.fit(train_features_df[features], train_labels_df['result'])

betting_thresh = 1.5
correct = 0
total_bets = 0
balance = 0
logloss_ensemble = 0
logloss_rf = 0
logloss_svm = 0
logloss_lr = 0
logloss_gbc = 0

# def rf_boosting(train_features, train_labels, n_estimators=750, n_boosts=3, misclassified_weight=2):
#     models = []
#     for boost in n_boosts:
#         clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
#         clf.fit(train_features_df[features], train_labels_df['result'])
#         models.append(clf)


for i in range(len(test_labels_df)):
    feature_record = test_features_df.iloc[i, :]
    label_record = test_labels_df.iloc[i, :]

    predictions_rf = clf.predict_proba(feature_record[features].reshape(1, -1))
    predictions_svm = svm.predict_proba(feature_record[features].reshape(1, -1))
    predictions_lr = lr.predict_proba(feature_record[features].reshape(1, -1))
    predictions_gbc = gbc.predict_proba(feature_record[features].reshape(1, -1))
    predictions = [[(w + x + y + z) / 4 for w, x, y, z in zip(predictions_rf[0], predictions_svm[0], predictions_lr[0],
                                                          predictions_gbc[0])]]

    # Risk / Profit rating: sqrt(prediction) because prediction is more important indicator than ratings (if you bet on something with a very high rating, but with a very small probability of winning, you probably will loose money)
    # home_rating = predictions[0][0]  * label_record['B365H'] ** 2
    # draw_rating = predictions[0][1]  * label_record['B365D'] ** 2
    # away_rating = predictions[0][2]  * label_record['B365A'] ** 2
    home_rating = predictions[0][0]  * label_record['B365H']
    draw_rating = predictions[0][1]  * label_record['B365D']
    away_rating = predictions[0][2]  * label_record['B365A']

    print('[', feature_record['date'], ']', feature_record['home_team'], 'vs.', feature_record['away_team'], 'RESULT:',
          label_record['home_team_goal'], '-', label_record['away_team_goal'])
    print('PREDICTIONS: ', predictions)
    print('RATING PRODUCTS:', [home_rating, draw_rating, away_rating])

    correct += np.argmax(predictions) == label_record['result']
    logloss_ensemble += -np.log2(predictions[0][label_record['result']])
    logloss_rf += -np.log2(predictions_rf[0][label_record['result']])
    logloss_svm += -np.log2(predictions_svm[0][label_record['result']])
    logloss_lr += -np.log2(predictions_lr[0][label_record['result']])
    logloss_gbc += -np.log2(predictions_gbc[0][label_record['result']])

    if home_rating >= betting_thresh:
        print('Betting 1 euro on home...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 0:
            print_color('Won '+ str(label_record['B365H'])+ ' euros', GREEN)
            balance += label_record['B365H']
        else:
            print_color('Lost it', RED)

    if draw_rating >= betting_thresh:
        print('Betting 1 euro on draw...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 1:
            print_color('Won '+ str(label_record['B365D'])+ ' euros', GREEN)
            balance += label_record['B365D']
        else:
            print_color('Lost it', RED)

    if away_rating >= betting_thresh:
        print('Betting 1 euro on away...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 2:
            print_color('Won ' + str(label_record['B365A']) + ' euros', GREEN)
            balance += label_record['B365A']
        else:
            print_color('Lost it', RED)

    print_color('Balance: '+str(balance), PURPLE)
    print('-------------------------------------------------------------------------------')

print('ACCURACY:', correct, '/', len(test_labels_df), '=', correct/len(test_labels_df))
print('ENSEMBLE LOGLOSS:', logloss_ensemble/len(test_labels_df))
print('RF LOGLOSS:', logloss_rf/len(test_labels_df))
print('SVM LOGLOSS:', logloss_svm/len(test_labels_df))
print('LR LOGLOSS:', logloss_lr/len(test_labels_df))
print('GBC LOGLOSS:', logloss_gbc/len(test_labels_df))
print('BETTED:', total_bets, '-- PROFIT:', balance)