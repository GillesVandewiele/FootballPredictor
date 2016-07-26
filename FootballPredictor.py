import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

features_df = pd.read_csv('kaggle_features_v2.csv')
labels_df = pd.read_csv('kaggle_labels_v2.csv')

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

def RF_feature_selection(features, labels, n_features):
    rf = RandomForestClassifier(n_estimators=1000, class_weight='auto', n_jobs=-1, bootstrap=True)
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

features = RF_feature_selection(train_features_df.drop('home_team', 1).drop('away_team', 1).drop('date', 1), train_labels_df['result'], 500)

clf = RandomForestClassifier(n_estimators=5000)
clf.fit(train_features_df[features], train_labels_df['result'])

betting_thresh = 1.25
correct = 0
total_bets = 0
balance = 0
for i in range(len(test_labels_df)):
    feature_record = test_features_df.iloc[i, :]
    label_record = test_labels_df.iloc[i, :]
    predictions = clf.predict_proba(feature_record[features].reshape(-1, 1))
    home_rating = predictions[0] * label_record['B365H']
    draw_rating = predictions[1] * label_record['B365D']
    away_rating = predictions[2] * label_record['B365A']

    print(feature_record['home_team'], 'vs.', feature_record['away_team'], 'RESULT:',
          label_record['home_team_goal'], '-', label_record['away_team_goal'])
    print('PREDICTIONS: ', predictions)
    print('RATING PRODUCTS:', [home_rating, draw_rating, away_rating])

    if home_rating >= betting_thresh:
        print('Betting 1 euro on home...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 0:
            print('Won', label_record['B365H'], 'euros')
            balance += label_record['B365H']
            correct += 1
        else:
            print('Lost it')

    if draw_rating >= betting_thresh:
        print('Betting 1 euro on draw...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 1:
            print('Won', label_record['B365D'], 'euros')
            balance += label_record['B365D']
            correct += 1
        else:
            print('Lost it')

    if away_rating >= betting_thresh:
        print('Betting 1 euro on away...')
        balance -= 1
        total_bets += 1
        if label_record['result'] == 2:
            print('Won', label_record['B365A'], 'euros')
            balance += label_record['B365A']
            correct += 1
        else:
            print('Lost it')

    print('-------------------------------------------------------------------------------')
print(correct, '/', len(test_labels_df), '=', correct/len(test_labels_df))
print('BETTED:', total_bets, '-- PROFIT:', balance)