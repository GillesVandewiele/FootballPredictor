
# coding: utf-8

#  # Old and ugly code
#  <b> WARNING!!! </b> The code below is ugly and old, for a newer and cleaner version, check the \_Spark predictor. The only reason this is handed in, is because AdaBoost from sklearn performs REALLY good (not supported in MLLib), it predicts 60% of the games correct and we make profit by betting on the result with the highest expected outcome.

# In[1]:

import pyspark
sc = pyspark.SparkContext('local[*]')

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# In[ ]:

import matplotlib.pylab as plt
import pandas as pd
import os
import re
from operator import add
import operator
import numpy as np
from lxml import etree, html
import urllib
from io import StringIO
from datetime import datetime
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[3]:

def extractTeams(tree):
    teams = {}
    
    for team in tree.findall('.//Team'):
        
        oneteam = {}
        
        team_id = int(team.attrib['uID'][1:])
        oneteam['name'] =  team.find('Name').text 
        
        players = []
        oneteam['players'] = players
        
        for player in team.findall('Player'):
            oneplayer = {}
            oneplayer['player_id']  = int(player.attrib['uID'][1:])
            oneplayer['first_name'] = player.find('PersonName/First').text
            oneplayer['last_name']  = player.find('PersonName/Last').text
            players.append(oneplayer)
            
        for manager in team.findall('TeamOfficial'):
            oneteam['manager'] = {}
            oneteam['manager']['first_name'] = manager.find('PersonName/First').text
            oneteam['manager']['last_name'] = manager.find('PersonName/Last').text
       
        teams[team_id] = oneteam
    return teams

def extractMatchDay(tree):
    for stats in tree.findall('.//Stat'):
        if stats.attrib['Type'] == 'matchday':
            return int(stats.text)
  
    return None
    
def extractMatchInfo(tree):
    info = {}
    node = tree.find('.//MatchInfo')
    if 'Weather' in node.attrib:
        info['weather'] = node.attrib['Weather']
    else:
        info['weather'] = "Rainy"  
    info['time'] = node.attrib['TimeStamp']
    if 'attendance' in node.attrib:
        info['attendance'] = int(node.find('Attendance').text)
    else:
         info['attendance'] = 0
    info['type'] = node.attrib['MatchType']
    info['period'] = node.attrib['Period']
    return info
    


def extractScoresFormationAndTeams(tree):
    info = {}
    key = None
    for team in tree.findall('//TeamData'):
        if team.attrib['Side'] == 'Home':
            key = 'home'
        else:
            key = 'away'
        
        nr_def = 0
        nr_mid = 0
        nr_strik = 0
        for player in team.findall('./PlayerLineUp/MatchPlayer'):
            if player.attrib['Position'] == "Defender": nr_def += 1
            if player.attrib['Position'] == "Midfielder": nr_mid += 1
            if player.attrib['Position'] == "Striker": nr_strik += 1
                

        info['score_'+key] = int(team.attrib['Score'])
        info['team_'+key] = team.attrib['TeamRef']    
        info['formation_'+key] = str(nr_def) + "-" + str(nr_mid) + "-" + str(nr_strik)

    return info

def extractBookingsGoalsAndSubs(tree):
    info = {}
    key = None
    for team in tree.findall('//TeamData'):
        if team.attrib['Side'] == 'Home':
            key = 'home'
        else:
            key = 'away'
            
        info['subs_'+key] = []
        info['bookings_'+key] = []
        info['goals_'+key] = []
        
        for booking in team.findall('./Booking'):
            new_booking = {}
            new_booking['type'] = booking.attrib['CardType']
            new_booking['time'] = booking.attrib['Time']
            new_booking['period'] = booking.attrib['Period']
            new_booking['player'] = booking.attrib['PlayerRef']
            info['bookings_'+key].append(new_booking)
            
        for sub in team.findall('./Substitution'):
            new_sub = {}
            new_sub['type'] = sub.attrib['Reason']
            new_sub['time'] = sub.attrib['Time']
            new_sub['period'] = sub.attrib['Period']
            new_sub['player_on'] = sub.attrib['SubOn']
            new_sub['player_off'] = sub.attrib['SubOff']
            info['subs_'+key].append(new_sub)
            
        for goal in team.findall('./Goal'):
            new_goal = {}
            new_goal['type'] = goal.attrib['Type']
            new_goal['time'] = goal.attrib['Time']
            new_goal['period'] = goal.attrib['Period']
            new_goal['player'] = goal.attrib['PlayerRef']
            for assist in goal.findall('./Assist'):
                new_goal['assist'] = assist.attrib['PlayerRef']
    
            info['goals_'+key].append(new_goal)

    return info
 
def extractGameInfo(tree):
  
    metadata = {}
    
    metadata['match_day']       = extractMatchDay(tree)
    metadata.update(extractMatchInfo(tree))
    metadata.update(extractScoresFormationAndTeams(tree))
    metadata.update(extractBookingsGoalsAndSubs(tree))

    return metadata


# In[4]:

def get_fifa_rating(player_name, season):
    url = "http://sofifa.com/players?keyword="+urllib.parse.quote_plus(player_name)+"&v="+season+"&hl=en-US"
    with urllib.request.urlopen(url) as page:
        s = page.read().decode("utf-8") 
    tree = html.fromstring(s).getroottree()
    for table_cell in tree.findall('//td'):
        if 'data-title' in table_cell.attrib and table_cell.attrib['data-title'] == "Overall rating":
            rating = table_cell.find('./span').text
            print((url, int(rating)))
            return int(rating)
        
    print((url, 50))
    return 50


# In[508]:

player_ratings = {}
def parse_game_xml_file(path):
    tree = etree.parse(path)
    first_tag =  tree.getroot().tag
    
    if first_tag != "response":
    
        # Extract team and game information
        teams = extractTeams(tree)
        game_info = extractGameInfo(tree)

        # Create feature vector and label vector (HT/FT) and return them
        home_team_id = game_info['team_home']
        away_team_id = game_info['team_away']
        
        weather = game_info['weather']
        time = datetime.strptime(game_info['time'][:-5], "%Y%m%dT%H%M%S")
        match_type = game_info['type']
        attendance = game_info['attendance']
        home_formation = game_info['formation_home']
        away_formation = game_info['formation_away']
        
        season = time.year + (time.month >= 8)  # This is the fifa_season

        home_team = teams[int(home_team_id[1:])]
        away_team = teams[int(away_team_id[1:])]

        home_team_name = home_team['name']
        away_team_name = away_team['name']

        full_time_result = ''
        if game_info['score_away'] < game_info['score_home']:
            full_time_result = 'HOME'
        elif game_info['score_home'] < game_info['score_away']:
            full_time_result = 'AWAY'
        else:
            full_time_result = 'DRAW'

        half_time_result = ''    
        goals_home = game_info['goals_home']
        goals_away = game_info['goals_away']
        half_score_home = 0
        half_score_away = 0
        for goal in goals_away:
            if goal['period'] == 'FirstHalf': half_score_home =+ 1
        for goal in goals_home:
            if goal['period'] == 'FirstHalf': half_score_away =+ 1
        if half_score_away < half_score_home:
            half_time_result = 'HOME'
        elif half_score_home < half_score_away:
            half_time_result = 'AWAY'
        else:
            half_time_result = 'DRAW'

        home_manager = home_team['manager']['first_name'] + " " + home_team['manager']['last_name']
        away_manager = away_team['manager']['first_name'] + " " + away_team['manager']['last_name']

        players_home = []
        players_away= []
        for player in home_team['players']:
            player_name = player['first_name']+" "+player['last_name']
            if player_name not in player_ratings:
                player_ratings[player_name] = get_fifa_rating(player_name, str(season)[2:])
            players_home.append(player_ratings[player_name])
            
        for player in away_team['players']:
            player_name = player['first_name']+" "+player['last_name']
            if player_name not in player_ratings:
                player_ratings[player_name] = get_fifa_rating(player_name, str(season)[2:])
            players_away.append(player_ratings[player_name])

        # TODO: the position of both teams in the league

        # TODO: laatste 5 onderlinge duels

        feature_vector = [home_team_id, away_team_id, home_team_name, away_team_name, home_manager,  
                          away_manager, weather, attendance, time, match_type, home_formation, away_formation,
                          players_home, players_away, half_time_result, full_time_result, game_info['score_home'], 
                          game_info['score_away']]
        
        return feature_vector
    
    return None
    
    


# In[509]:

## Warning: executing this cell can take a long time!
path = './F7'
feature_vectors = []
for fn in os.listdir(path):
    feature_vector = parse_game_xml_file(os.path.join(path,fn))
    if feature_vector:
        feature_vectors.append(feature_vector)
feature_df_temp = pd.DataFrame(feature_vectors)
feature_df_temp.columns = ["home_id", "away_id", "home_name", "away_name", "home_manager", "away_manager", "weather",
                           "attendance", "time", "match_type", "home_formation", "away_formation", 
                           "players_home_ratings", "players_away_ratings" ,"HT", "FT", "goals_home", "goals_away"]


# In[648]:

feature_df = feature_df_temp  # taking a copy so we don't need to execute cell above every time
#print(feature_df.head(25))


# In[649]:

names = list(feature_df['home_name'].values)
names.extend(list(feature_df['away_name'].values))

# This was fun
opta_to_bet_dict = {'1. FC Kaiserslautern': 'Kaiserslautern',  '1. FC Köln': 'FC Koln', '1. FSV Mainz 05': 'Mainz', 
                    '1899 Hoffenheim': 'Hoffenheim', 'Alemannia Aachen': 'Aachen', 
                    'Borussia Mönchengladbach': 'M\'gladbach', 'FC Augsburg': 'Augsburg',  'FC Carl Zeiss Jena': 'CZ Jena',
                    'FC Erzgebirge Aue': 'Erzgebirge Aue', 'FC St. Pauli': 'St Pauli', 'OFC Kickers 1901': 'Offenbach',
                    'SC Freiburg': 'Freiburg', 'SC Paderborn': 'Paderborn', 'SV Wehen Wiesbaden': 'Wehen',
                    'SpVgg Greuther Fürth': 'Greuther Furth', 'TSV München 1860': 'Munich 1860',
                    'TuS Koblenz': 'Koblenz', 'VfL Osnabrück': 'Osnabruck'}


# In[650]:

betting_odds = pd.read_csv('D2.csv')

# Clean the feature_df so we can fit it with random forest classifier
feature_df = feature_df.sort_values(by="time")

ft_map = {'HOME': 2, 'DRAW': 1, 'AWAY': 0}
    
ft_label = [ft_map[label] for label in feature_df['FT']]

htft_label = [feature_df.iloc[i, :]['FT']+"/"+ feature_df.iloc[i, :]['HT'] for i in range(len(feature_df.index))]
htft_map = {}
counter = 0
for label in np.unique(htft_label):
    htft_map[label] = counter
    counter += 1
htft_label = [htft_map[label] for label in htft_label]

# Features: home_formation, away_formation, home_player_ratings, away_player_ratings
# TODO: laatste matches van home en away teams, laatste onderlinge duels
    
# Random forest can deal with categorical variable simply converted to a single numerical variable for each category.
# Else we need to use dummy transform on formation and other categorical variables
map_formation = {}
counter = 0
formations = feature_df['home_formation'].values
formations = np.append(formations, feature_df['away_formation'].values)
for formation in np.unique(formations):
    map_formation[formation] = counter
    counter += 1
    
feature_df['home_formation'] = feature_df['home_formation'].map(map_formation)
feature_df['away_formation'] = feature_df['away_formation'].map(map_formation)
    
map_weather = {}
counter = 0
for weather in np.unique(feature_df['weather']):
    map_weather[weather]= counter
    counter += 1
    
feature_df['weather'] = feature_df['weather'].map(map_weather)

map_match_type = {}
counter = 0
for match_type in np.unique(feature_df['match_type']):
        map_match_type[match_type] = counter
        counter += 1
        
feature_df['match_type'] = feature_df['match_type'].map(map_match_type)

cleaned_feature_vectors = []
betting_odds_vectors = []
for i in range(len(feature_df.index)):
    record = feature_df.iloc[i, :]
    cleaned_feature_vector = [record['time'], #record['home_name'], record['away_name'],
                              record['home_id'][1:], record['away_id'][1:],  
                              record['weather'], record['attendance'], 
                              record['home_formation'], record['away_formation']]
    filtered_betting_odds = betting_odds[(betting_odds.HomeTeam == opta_to_bet_dict[record['home_name']])
                                         & (betting_odds.AwayTeam == opta_to_bet_dict[record['away_name']])]
    betting_odds_vectors.append([filtered_betting_odds['B365A'].values[0], filtered_betting_odds['B365D'].values[0], 
                                 filtered_betting_odds['B365H'].values[0], record['home_id'][1:], record['away_id'][1:]])
    home_squad_ratings = list(map(int, record['players_home_ratings'][:11]))
    cleaned_feature_vector.extend(home_squad_ratings)
    
    away_squad_ratings = list(map(int, record['players_away_ratings'][:11]))
    cleaned_feature_vector.extend(away_squad_ratings)
    
    home_squad_mean = np.mean(home_squad_ratings)
    home_squad_sum = sum(home_squad_ratings)
    home_top_sum = sum(sorted(home_squad_ratings, reverse=True)[:3])
    
    away_squad_mean = np.mean(away_squad_ratings)
    away_squad_sum = sum(away_squad_ratings)
    away_top_sum = sum(sorted(away_squad_ratings, reverse=True)[:3])
    
    cleaned_feature_vector.extend([home_squad_mean, home_squad_sum, home_top_sum, 
                                   away_squad_mean, away_squad_sum, away_top_sum])
    
    # Get fraction of points and get the goals for latest games of teams:
    N_MATCHES = 3
    home_goals_scored_away = 0
    home_goals_conceded_away = 0
    home_latest_away_games = [-1]*N_MATCHES  
    filtered_df = feature_df[(feature_df.time < record['time']) & (feature_df.away_id == record['home_id'])].tail(N_MATCHES)
    for i in range(min(len(home_latest_away_games), len(filtered_df.index))):
        home_latest_away_games[i] = filtered_df.iloc[i, :]['FT']
        home_goals_scored_away += filtered_df.iloc[i, :]['goals_away']
        home_goals_conceded_away += filtered_df.iloc[i, :]['goals_home']
        
    home_latest_home_games = [-1]*N_MATCHES
    home_goals_scored_home = 0
    home_goals_conceded_home = 0
    filtered_df = feature_df[(feature_df.time < record['time']) & (feature_df.home_id == record['home_id'])].tail(N_MATCHES)
    for i in range(min(len(home_latest_home_games), len(filtered_df.index))):
        home_latest_home_games[i] = filtered_df.iloc[i, :]['FT']
        home_goals_scored_home += filtered_df.iloc[i, :]['goals_home']
        home_goals_conceded_home += filtered_df.iloc[i, :]['goals_away']
        
    home_total_home_sum = 0
    home_total_home_score = 0
    for game in home_latest_home_games:
        if game != -1:
            home_total_home_sum += 3
            if game == "DRAW": home_total_home_score += 1
            if game == "HOME": home_total_home_score += 3
                
    home_total_away_sum = 0
    home_total_away_score = 0
    for game in home_latest_away_games:
        if game != -1:
            home_total_away_sum += 3
            if game == "DRAW": home_total_away_score += 1
            if game == "AWAY": home_total_away_score += 3
    
    if home_total_home_sum != 0:
        home_latest_home_fraction = float(home_total_home_score)/float(home_total_home_sum)
    else:
        home_latest_home_fraction = 0
    if home_total_home_sum+home_total_away_sum != 0:
        home_latest_game_fraction = (float(home_total_home_score+home_total_away_score)/
                                     float(home_total_home_sum+home_total_away_sum))
    else:
        home_latest_game_fraction = 0                             
                

    away_latest_away_games = [-1]*N_MATCHES
    away_goals_scored_away = 0
    away_goals_conceded_away = 0
    filtered_df = feature_df[(feature_df.time < record['time']) & (feature_df.away_id == record['away_id'])].tail(N_MATCHES)
    for i in range(min(len(away_latest_away_games), len(filtered_df.index))):
        away_latest_away_games[i] = filtered_df.iloc[i, :]['FT']
        away_goals_scored_away += filtered_df.iloc[i, :]['goals_away']
        away_goals_conceded_away += filtered_df.iloc[i, :]['goals_home']
        
    away_latest_home_games = [-1]*N_MATCHES
    away_goals_scored_home = 0
    away_goals_conceded_home = 0
    filtered_df = feature_df[(feature_df.time < record['time']) & (feature_df.home_id == record['away_id'])].tail(N_MATCHES)
    for i in range(min(len(away_latest_home_games), len(filtered_df.index))):
        away_latest_home_games[i] = filtered_df.iloc[i, :]['FT']
        away_goals_scored_home += filtered_df.iloc[i, :]['goals_home']
        away_goals_conceded_home += filtered_df.iloc[i, :]['goals_away']
                                 
    away_total_home_sum = 0
    away_total_home_score = 0
    for game in away_latest_home_games:
        if game != -1:
            away_total_home_sum += 3
            if game == "DRAW": away_total_home_score += 1
            if game == "HOME": away_total_home_score += 3
                
    away_total_away_sum = 0
    away_total_away_score = 0
    for game in away_latest_away_games:
        if game != -1:
            away_total_away_sum += 3
            if game == "DRAW": away_total_away_score += 1
            if game == "AWAY": away_total_away_score += 3
                
    if away_total_away_sum != 0:
        away_latest_away_fraction = float(away_total_away_score)/float(away_total_away_sum)
    else:
        away_latest_away_fraction = 0
    if away_total_away_sum+away_total_home_sum != 0:
        away_latest_game_fraction = (float(away_total_home_score+away_total_away_score)/
                                     float(away_total_away_sum+away_total_home_sum))
    else:
        away_latest_game_fraction = 0                             
    
    cleaned_feature_vector.extend([home_latest_home_fraction, home_latest_game_fraction, away_latest_away_fraction,
                                   away_latest_game_fraction])
    cleaned_feature_vector.extend([home_goals_scored_home, home_goals_conceded_home, home_goals_scored_away, 
                                   home_goals_conceded_away, away_goals_scored_home, away_goals_conceded_home,
                                   away_goals_scored_away, away_goals_conceded_away])
    
    cleaned_feature_vectors.append(cleaned_feature_vector)
    
feature_df = pd.DataFrame(cleaned_feature_vectors)
home_player_columns=['home_player_rating'+str(i) for i in range(1, 12)]
away_player_columns=['away_player_rating'+str(i) for i in range(1, 12)]
column_names = ['time', 'home_id', 'away_id', 'weather', 'attendance', 
                'home_formation', 'away_formation']
column_names.extend(home_player_columns)
column_names.extend(away_player_columns)
column_names.extend(['home_squad_mean', 'home_squad_sum', 'home_top_sum', 
                    'away_squad_mean', 'away_squad_sum', 'away_top_sum'])
column_names.extend(['home_latest_home_fraction', 'home_latest_game_fraction', 
                     'away_latest_away_fraction', 'away_latest_game_fraction', 'home_goals_scored_home',
                     'home_goals_conceded_home', 'home_goals_scored_away', 'home_goals_conceded_away',
                     'away_goals_scored_home', 'away_goals_conceded_home', 'away_goals_scored_away', 
                     'away_goals_conceded_away'])
feature_df.columns = column_names


# In[651]:

feature_df = feature_df.drop("time", axis=1)
X_train, X_test = feature_df[:round(-len(feature_df)*0.25)], feature_df[round(len(feature_df)*0.75):]
y_train, y_test = ft_label[:round(-len(feature_df)*0.25)], ft_label[round(len(feature_df)*0.75):]

betting_odds_df = pd.DataFrame(betting_odds_vectors)
betting_odds_df.columns = ['AWAY', 'DRAW', 'HOME', 'HOME_ID', 'AWAY_ID']


# In[652]:

print(y_train)
print(betting_odds_df)
print (X_train)


# In[653]:

def RF_feature_selection(features, labels, n_features):
    rf = RandomForestClassifier(n_estimators=100, class_weight='auto', n_jobs=-1, bootstrap=True)
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


# In[654]:

best_features = RF_feature_selection(X_train, y_train, 25)
best_features.append('home_id')
best_features.append('away_id')
best_features = list(set(best_features)) # drop duplicates
X_train = X_train[best_features]
X_test = X_test[best_features]


# In[590]:

support_vm = svm.SVC(C=32.0, kernel='linear', tol=1e-3, probability=True)
support_vm.fit(X_train, y_train)


# In[363]:

print("#########Support Vector Machine##########")
logloss = 0
correct = 0
gamble_correct = 0
for i in range(len(X_test.index)):
    f_vector = np.reshape(X_test.iloc[i].values, (1, -1))
    ratings = betting_odds_df[(betting_odds_df.HOME_ID == X_test.iloc[i,:]['home_id']) & 
                              (betting_odds_df.AWAY_ID == X_test.iloc[i,:]['away_id'])][['AWAY', 'DRAW', 'HOME']].values[0]
    print(support_vm.predict_proba(f_vector)[0], np.multiply(support_vm.predict_proba(f_vector)[0], ratings), y_test[i])
    logloss += -np.log2(support_vm.predict_proba(f_vector)[0][y_test[i]])
    correct += (np.argmax(support_vm.predict_proba(f_vector)[0]) == [y_test[i]])
    gamble_correct += ratings[np.argmax(np.multiply(rf.predict_proba(f_vector)[0], ratings))] *                      (np.argmax(np.multiply(rf.predict_proba(f_vector)[0], ratings)) == [y_test[i]])
print("LOGLOSS = ", float(logloss)/float(len(y_test)))
print("GAMBLE PROFIT = ", float(gamble_correct) - len(y_test))
print("CORRECT = ", float(correct)/float(len(y_test)))


# In[364]:

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features=None, bootstrap=True)
rf.fit(X_train, y_train)


# In[365]:

print("###########Random Forest############")
logloss = 0
correct = 0
gamble_correct = 0
for i in range(len(X_test.index)):
    f_vector = np.reshape(X_test.iloc[i].values, (1, -1))
    ratings = betting_odds_df[(betting_odds_df.HOME_ID == X_test.iloc[i,:]['home_id']) & 
                              (betting_odds_df.AWAY_ID == X_test.iloc[i,:]['away_id'])][['AWAY', 'DRAW', 'HOME']].values[0]
    print(rf.predict_proba(f_vector)[0], np.multiply(rf.predict_proba(f_vector)[0], ratings), y_test[i])
    logloss += -np.log2(rf.predict_proba(f_vector)[0][y_test[i]])
    gamble_correct += ratings[np.argmax(np.multiply(rf.predict_proba(f_vector)[0], ratings))] *                      (np.argmax(np.multiply(rf.predict_proba(f_vector)[0], ratings)) == [y_test[i]])
    correct += (np.argmax(rf.predict_proba(f_vector)[0]) == [y_test[i]])
print("LOGLOSS = ", float(logloss)/float(len(y_test)))
print("GAMBLE PROFIT = ", float(gamble_correct) - len(y_test))
print("CORRECT = ", float(correct)/float(len(y_test)))


# In[646]:

ada = AdaBoostClassifier(n_estimators=1500, learning_rate=0.075)
ada.fit(X_train, y_train)


# In[647]:

print("############################################Adaboost###############################################")
print("###########PROBABILITY###########################EXPECTED PROFIT##################BET ODDS####CLASS")
logloss = 0
correct = 0
gamble_correct = 0
for i in range(len(X_test.index)):
    f_vector = np.reshape(X_test.iloc[i].values, (1, -1))
    ratings = betting_odds_df[(betting_odds_df.HOME_ID == X_test.iloc[i,:]['home_id']) & 
                              (betting_odds_df.AWAY_ID == X_test.iloc[i,:]['away_id'])][['AWAY', 'DRAW', 'HOME']].values[0]
    print(ada.predict_proba(f_vector)[0], np.multiply(ada.predict_proba(f_vector)[0], ratings), ratings, y_test[i])
    logloss += -np.log2(ada.predict_proba(f_vector)[0][y_test[i]])
    gamble_correct += ratings[np.argmax(np.multiply(ada.predict_proba(f_vector)[0], ratings))] *                      (np.argmax(np.multiply(ada.predict_proba(f_vector)[0], ratings)) == [y_test[i]])
    correct += (np.argmax(ada.predict_proba(f_vector)[0]) == [y_test[i]])
print("LOGLOSS = ", float(logloss)/float(len(y_test)))
print("GAMBLE PROFIT = ", float(gamble_correct) / len(y_test))
print("CORRECT = ", float(correct)/float(len(y_test)))


# In[412]:

logreg = LogisticRegression( C=4.0, penalty='l2', solver='lbfgs', n_jobs=-1, multi_class='ovr', tol=1e-5)
logreg.fit(X_train, y_train)


# In[352]:

print("#########Logistic Regression##########")
logloss = 0
correct = 0
for i in range(len(X_test.index)):
    f_vector = np.reshape(X_test.iloc[i].values, (1, -1))
    print(logreg.predict_proba(f_vector)[0], y_test[i])
    logloss += -np.log2(logreg.predict_proba(f_vector)[0][y_test[i]])
    correct += (np.argmax(logreg.predict_proba(f_vector)[0]) == [y_test[i]])
print("LOGLOSS = ", float(logloss)/float(len(y_test)))
print("CORRECT = ", float(correct)/float(len(y_test)))


# In[ ]:



