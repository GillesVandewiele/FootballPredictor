import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

features_df = pd.read_csv('kaggle_features_v2.csv')
labels_df = pd.read_csv('kaggle_labels_v2.csv')
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
    rf = RandomForestClassifier(n_estimators=500, class_weight='auto', n_jobs=-1, bootstrap=True)
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

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(train_features_df[features], train_labels_df['result'])

predicted_labels = clf.predict(test_features_df[features])
correct = 0
for i in range(len(test_labels_df)):
    if predicted_labels[i] == test_labels_df.iloc[i, :]['result']:
        correct += 1
print(correct, '/', len(test_labels_df), '=', correct/len(test_labels_df))