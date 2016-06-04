
# coding: utf-8

# # Football Predictor
# 
# A football predictor has been made, using different classification algorithms on training data, with features obtained only 
# from the F7 files. The following features were extracted:
# * <b> The ratings of each of the 11 selected players in Fifa 2008: </b> Simply by going to http://sofifa.com/players?keyword=player_name&v=08&hl=en-US . Hey, Fifa 2008! That's data leakage! Well, the ratings of the players in the fifa games prior to 2013 were based on their performance of the previous season. Fifa 2008 was released 27 September 2007 and the ratings corresponded to the performance in season 06/07. Their individual ratings were used (11 features), their mean and sum and the top 3 scoring players were used another time (only sum).
# * <b> The performance of the latest home games for the home team (how many points obtained) </b> 
# * <b> The performance of the latest away games for the away team </b>
# * <b> The performance of all latest games for both teams </b>  
# * <b> The formations for both teams </b>
# * <b> The weather and attendance: </b> the latter could be excluded, as it is hard to known prior to a game, or could be replaced by maximum capacity of the home stadium (which actually seems the case (50000 people in 2. Bundesliga??))
# * <b> Goals scored and conceded in the latest games for both teams (home and away) </b>
# 
# The features based on prior games are rather poor-performing. That is because if we base these features on n prior games, than the first n match days will contain bad values for these features (e.g. all games on the first matchday have no prior matches). Since the dataset is rather small, the impact of this is large. Again, historic data could be scraped to solve this problem. Random Forest feature selection was used to select the most important features.
# 
# Of course, more features are possible, such as:
# * <b> statistics of mutual matches: </b> Club Brugge hasn't won a game against Anderlecht away since 2005. The feature is not included since there's never more than 1 mutual match between two teams. An internet scraper could be used though. 
# * <b> distance travelled for the away team: </b> in Belgiums, the teams have no private airplanes, thus a long bus journey can decrease the motivation on the pitch. Geocoding could be used for this.
# * <b> prior game statistics from F24 files: </b> many possible game statistics from previous matches can be used as features. Examples include ball possession, number of shots on goal, number of corners, and so on. It is not done since the number of features was already pretty high and because it requires a lot of work, it can of course easily be realized.
# * <b> the type of game: </b> some teams are much better in cup games than other teams that are good in the regular competition, mostly because the interest in the cup is less high than becoming champion. One example is Lokeren, who won the cup two times in three years: in 2012 and 2014 and never became champion of Belgium yet. Again, only competition games were included in the dataset, so it is of no importance here
# * <b> The manager of a team: </b> having Pep Guardiola as your manager could be quite an advantage...
# * <b> many, many more </b>
# 
# The predictions made by the classification algorithms are multiplied by the Betting365 (one of the many possible betting providers) ratings from the year 2007/2008. These ratings were downloaded from http://football-data.co.uk/ . A product higher than 1 is expected to be profitable. A betting simulation was done in parallell with the predictions (bet on the most profitable class) and the expected total profit was calculated. 
# 
# The number of classes could easily be extended by predicting Half-Time/Full-Time results (e.g. Home team wins first half, but away teams wins the game in the end). Again, this was out of scope (definitely with this dataset).
# 
# <b> In the other file Football_Predictor, the sklearn library is used in combination with pandas, there AdaBoost (not yet supported in MLLIB) achieves pretty good results. Also Random Forest feature selection was applied there, which just measures the feature importance and doesn't require a threshold (allows for selecting the n top features). In this file, PCA is used. MLLIB is still far from mature, most implemented classifiers are only binary classifiers (Logistic Regression, SVM, GradientBoostedTrees, ...) and the OneVsRest module is not yet supported for python it seems. Therefore, only Random Forest was used. </b>

# ## Imports

# In[3]:

import pyspark
sc = pyspark.SparkContext('local[*]')

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# In[505]:

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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, PCA
from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint


# ## XML extraction functions

# In[185]:

def extractPlayersAndManager(tree):
    """
    Extract the players selected for the game (including bench players) and the manager
    """
    teams = {}
    
    # Home and away team
    for team in tree.findall('.//Team'):
        
        oneteam = {}
        
        team_id = int(team.attrib['uID'][1:])
        oneteam['name'] =  team.find('Name').text 
        
        players = []
        oneteam['players'] = players
        
        # Extract the players' name and ID
        for player in team.findall('Player'):
            oneplayer = {}
            oneplayer['player_id']  = int(player.attrib['uID'][1:])
            oneplayer['first_name'] = player.find('PersonName/First').text
            oneplayer['last_name']  = player.find('PersonName/Last').text
            players.append(oneplayer)
            
        # Extract the managers name
        for manager in team.findall('TeamOfficial'):
            oneteam['manager'] = {}
            oneteam['manager']['first_name'] = manager.find('PersonName/First').text
            oneteam['manager']['last_name'] = manager.find('PersonName/Last').text
       
        teams[team_id] = oneteam
    return teams
    
def extractMatchInfo(tree):
    """
    What's the weather, what kind of game is it, what was the attendance and most important, when was the game played.
    """
    info = {}
    node = tree.find('.//MatchInfo')
    if 'Weather' in node.attrib:
        info['weather'] = node.attrib['Weather']
    else:
        info['weather'] = "Rainy"  
    info['time'] = datetime.strptime(node.attrib['TimeStamp'][:-5], "%Y%m%dT%H%M%S")
    info['attendance'] = 0
    for att in  node.findall('./Attendance'):
        info['attendance'] = int(att.text)
    info['type'] = node.attrib['MatchType']
    info['period'] = node.attrib['Period']
    return info
    


def extractScoresFormationAndTeams(tree):
    """
    In what formation are both teams playing (e.g. 4-4-2), who was the home and away team, and what was the final score
    """
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

    # Compare the scores of both teams and return result
    full_time_result = ''
    if info['score_away'] < info['score_home']:
        full_time_result = 'HOME'
    elif info['score_home'] < info['score_away']:
        full_time_result = 'AWAY'
    else:
        full_time_result = 'DRAW' 
    info['full_time'] = full_time_result
        
    return info

def extractBookingsGoalsAndSubs(tree):
    """
    Extract all bookings, goals and substitutions along with their time and player. This can be used to make
    HT/FT predictions
    """
    info = {}
    goals_teams = {}
    goals_teams['goals_home'] = []
    goals_teams['goals_away'] = []
    key = None
    for team in tree.findall('//TeamData'):
        if team.attrib['Side'] == 'Home':
            key = 'home'
        else:
            key = 'away'
            
        info['subs_'+key] = []
        info['bookings_'+key] = []
        
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
    
            goals_teams['goals_'+key].append(new_goal)
        
    half_time_result = ''    
    goals_home = goals_teams['goals_home']
    goals_away = goals_teams['goals_away']
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
    info['half_time'] = half_time_result

    return info

def parse_game_xml_file(path):
    """
    Extract all information, using the helper functions above from a file at 'path'
    """
    tree = etree.parse(path)
    first_tag =  tree.getroot().tag
    
    if first_tag != "response": # There are 2 corrupted files in between the 88 matches, filter them out
        # Extract team and game information
        metadata = {}
        metadata.update(extractPlayersAndManager(tree))
        metadata.update(extractMatchInfo(tree))
        metadata.update(extractScoresFormationAndTeams(tree))
        metadata.update(extractBookingsGoalsAndSubs(tree))

        return metadata
    
    return None


# ## Scraping functions

# In[101]:

def get_fifa_rating(player_name, season):
    url = "http://sofifa.com/players?keyword="+urllib.parse.quote_plus(player_name)+"&v="+season+"&hl=en-US"
    with urllib.request.urlopen(url) as page:
        s = page.read().decode("utf-8") 
    tree = html.fromstring(s).getroottree()
    for table_cell in tree.findall('//td'):
        if 'data-title' in table_cell.attrib and table_cell.attrib['data-title'] == "Overall rating":
            rating = table_cell.find('./span').text
            return int(rating)
        
    return 50


# ## Feature extraction functions

# In[ ]:

# Dynamic programming for getting the fifa ratings of players
player_ratings = {}

# Read the betting odds downloaded from football-data.co.uk
betting_odds = pd.read_csv('D2.csv')
# This was fun
opta_to_bet_dict = {'1. FC Kaiserslautern': 'Kaiserslautern',  '1. FC Köln': 'FC Koln', '1. FSV Mainz 05': 'Mainz', 
                    '1899 Hoffenheim': 'Hoffenheim', 'Alemannia Aachen': 'Aachen', 
                    'Borussia Mönchengladbach': 'M\'gladbach', 'FC Augsburg': 'Augsburg',  'FC Carl Zeiss Jena': 'CZ Jena',
                    'FC Erzgebirge Aue': 'Erzgebirge Aue', 'FC St. Pauli': 'St Pauli', 'OFC Kickers 1901': 'Offenbach',
                    'SC Freiburg': 'Freiburg', 'SC Paderborn': 'Paderborn', 'SV Wehen Wiesbaden': 'Wehen',
                    'SpVgg Greuther Fürth': 'Greuther Furth', 'TSV München 1860': 'Munich 1860',
                    'TuS Koblenz': 'Koblenz', 'VfL Osnabrück': 'Osnabruck'}


# In[276]:

def get_fifa_rating_features(home_player_list, away_player_list, season):
    home_squad_ratings = []
    for player in home_player_list:
        player_name = player['first_name']+" "+player['last_name']
        if player_name not in player_ratings:
            player_ratings[player_name] = get_fifa_rating(player_name, str(season)[2:])
        home_squad_ratings.append(player_ratings[player_name])  
    
    away_squad_ratings = []
    for player in away_player_list:
        player_name = player['first_name']+" "+player['last_name']
        if player_name not in player_ratings:
            player_ratings[player_name] = get_fifa_rating(player_name, str(season)[2:])
        away_squad_ratings.append(player_ratings[player_name])
    
    home_squad_ratings = home_squad_ratings[:11]
    away_squad_ratings = away_squad_ratings[:11]
    
    home_squad_mean = np.mean(home_squad_ratings)
    home_squad_sum = sum(home_squad_ratings)
    home_top_sum = sum(sorted(home_squad_ratings, reverse=True)[:3])
    
    away_squad_mean = np.mean(away_squad_ratings)
    away_squad_sum = sum(away_squad_ratings)
    away_top_sum = sum(sorted(away_squad_ratings, reverse=True)[:3])
    
    return {'home_ratings': home_squad_ratings, 'away_ratings': away_squad_ratings, 
            'home_ratings_mean': home_squad_mean, 'home_ratings_sum': home_squad_sum, 'home_top_sum': home_top_sum,
            'away_ratings_mean': away_squad_mean, 'away_ratings_sum': away_squad_sum, 'away_top_sum': away_top_sum}

def get_betting_odds(home_name, away_name):
    filtered_betting_odds = betting_odds[(betting_odds.HomeTeam == opta_to_bet_dict[home_name])
                                         & (betting_odds.AwayTeam == opta_to_bet_dict[away_name])]
    return {'AWAY': filtered_betting_odds['B365A'].values[0], 'DRAW': filtered_betting_odds['B365D'].values[0], 
            'HOME': filtered_betting_odds['B365H'].values[0]}   

def get_prior_game_statistics(home_id, away_id, time, feature_df, n_games=3):
    home_goals_scored_away = 0
    home_goals_conceded_away = 0
    home_latest_away_games = [-1]*n_games  
    filtered_df = feature_df.filter((feature_df['time'] < time) & (feature_df['away_id'] == home_id))
    filtered_df = filtered_df.sort(feature_df['time'].desc()).take(n_games)
    for i in range(np.min([n_games, len(filtered_df)])): # min gave error: _() takes 1 positional argument but 2 were given
        home_latest_away_games[i] = filtered_df[i]['result']
        home_goals_scored_away += filtered_df[i]['away_goals']
        home_goals_conceded_away += filtered_df[i]['home_goals']
        
    home_latest_home_games = [-1]*n_games
    home_goals_scored_home = 0
    home_goals_conceded_home = 0
    filtered_df = feature_df.filter((feature_df['time'] < time) & (feature_df['home_id'] == home_id))
    filtered_df = filtered_df.sort(feature_df['time'].desc()).take(n_games)
    for i in range(np.min([n_games, len(filtered_df)])):
        home_latest_home_games[i] = filtered_df[i]['result']
        home_goals_scored_home += filtered_df[i]['home_goals']
        home_goals_conceded_home += filtered_df[i]['away_goals']
        
    home_total_home_sum = 0
    home_total_home_score = 0
    for game in home_latest_home_games:
        if game != -1:
            home_total_home_sum += 3
            if game == 1: home_total_home_score += 1
            if game == 2: home_total_home_score += 3
                
    home_total_away_sum = 0
    home_total_away_score = 0
    for game in home_latest_away_games:
        if game != -1:
            home_total_away_sum += 3
            if game == 1: home_total_away_score += 1
            if game == 0: home_total_away_score += 3
    
    if home_total_home_sum != 0:
        home_latest_home_fraction = float(home_total_home_score)/float(home_total_home_sum)
    else:
        home_latest_home_fraction = 0.0
    if home_total_home_sum+home_total_away_sum != 0:
        home_latest_game_fraction = (float(home_total_home_score+home_total_away_score)/
                                     float(home_total_home_sum+home_total_away_sum))
    else:
        home_latest_game_fraction = 0.0                             
                

    away_latest_away_games = [-1]*n_games
    away_goals_scored_away = 0
    away_goals_conceded_away = 0
    filtered_df = feature_df.filter((feature_df['time'] < time) & (feature_df['away_id'] == away_id))
    filtered_df = filtered_df.sort(feature_df['time'].desc()).take(n_games)
    for i in range(np.min([n_games, len(filtered_df)])):
        away_latest_away_games[i] = filtered_df[i]['result']
        away_goals_scored_away += filtered_df[i]['away_goals']
        away_goals_conceded_away += filtered_df[i]['home_goals']
        
    away_latest_home_games = [-1]*n_games
    away_goals_scored_home = 0
    away_goals_conceded_home = 0
    filtered_df = feature_df.filter((feature_df['time'] < time) & (feature_df['home_id'] == away_id))
    filtered_df = filtered_df.sort(feature_df['time'].desc()).take(n_games)
    for i in range(np.min([n_games, len(filtered_df)])):
        away_latest_home_games[i] = filtered_df[i]['result']
        away_goals_scored_home += filtered_df[i]['home_goals']
        away_goals_conceded_home += filtered_df[i]['away_goals']
                                 
    away_total_home_sum = 0
    away_total_home_score = 0
    for game in away_latest_home_games:
        if game != -1:
            away_total_home_sum += 3
            if game == 1: away_total_home_score += 1
            if game == 2: away_total_home_score += 3
                
    away_total_away_sum = 0
    away_total_away_score = 0
    for game in away_latest_away_games:
        if game != -1:
            away_total_away_sum += 3
            if game == 1: away_total_away_score += 1
            if game == 0: away_total_away_score += 3
                
    if away_total_away_sum != 0:
        away_latest_away_fraction = float(away_total_away_score)/float(away_total_away_sum)
    else:
        away_latest_away_fraction = 0.0
    if away_total_away_sum+away_total_home_sum != 0:
        away_latest_game_fraction = (float(away_total_home_score+away_total_away_score)/
                                     float(away_total_away_sum+away_total_home_sum))
    else:
        away_latest_game_fraction = 0.0    
    
    return {'home_latest_home_fraction': home_latest_home_fraction, 'home_latest_game_fraction': home_latest_game_fraction, 
            'away_latest_away_fraction': away_latest_away_fraction, 'away_latest_game_fraction': away_latest_game_fraction, 
            'home_goals_scored_home': home_goals_scored_home, 'home_goals_conceded_home': home_goals_conceded_home, 
            'away_goals_scored_away': away_goals_scored_away, 'away_goals_conceded_away': away_goals_conceded_away}


# ## Creating feature vectors and dividing into train & test set

# In[506]:

# Mapping directories (strings should be mapped to integers)
weather_dict = {}
formation_dict = {}
ft_map = {'HOME': 2, 'DRAW': 1, 'AWAY': 0}
B365_to_label = {'B365H': 2, 'B365D': 1, 'B365A': 0}

# Read every file in the directory F7 and extract all features from it
path = './F7' 
feature_vectors = []
betting_vectors = []
for fn in os.listdir(path):
    xml_file = parse_game_xml_file(os.path.join(path,fn))
    if xml_file:
        time = xml_file['time']
        
        home_id = int(xml_file['team_home'][1:])
        away_id = int(xml_file['team_away'][1:])
        
        home_team = xml_file[home_id]
        home_name = home_team['name']
        
        home_goals = xml_file['score_home']
        away_goals = xml_file['score_away']
        
        away_team = xml_file[away_id]
        away_name = away_team['name']
        
        b365_betting_odds = get_betting_odds(home_name, away_name)
        
        weather_string = xml_file['weather']
        if weather_string in weather_dict:
            weather = weather_dict[weather_string]
        else:
            nr = len(weather_dict)
            weather_dict[weather_string] = nr
            weather = nr
            
        attendance = xml_file['attendance']
        
        home_formation_string = xml_file['formation_home']
        if home_formation_string in formation_dict:
            home_formation = formation_dict[home_formation_string]
        else:
            nr = len(formation_dict)
            formation_dict[home_formation_string] = nr
            home_formation = nr
            
        away_formation_string = xml_file['formation_away']
        if away_formation_string in formation_dict:
            away_formation = formation_dict[away_formation_string]
        else:
            nr = len(formation_dict)
            formation_dict[away_formation_string] = nr
            away_formation = nr
        
        fifa_ratings = get_fifa_rating_features(home_team['players'], away_team['players'], "2008")
        
        result = ft_map[xml_file['full_time']]
        
        feature_dict = {'time': time, 'home_id': home_id, 'away_id': away_id, 'weather': weather, 'attendance': attendance, 
                        'home_formation': home_formation, 'away_formation': away_formation, 
                        'home_squad_mean': float(fifa_ratings['home_ratings_mean']), 
                        'home_squad_sum': fifa_ratings['home_ratings_sum'], 
                        'home_top_sum': fifa_ratings['home_top_sum'], 
                        'away_squad_mean': float(fifa_ratings['away_ratings_mean']), 
                        'away_squad_sum': fifa_ratings['away_ratings_sum'], 
                        'away_top_sum': fifa_ratings['away_top_sum'], 'B365H': float(b365_betting_odds['HOME']),
                        'B365D': float(b365_betting_odds['DRAW']), 'B365A': float(b365_betting_odds['AWAY']), 
                        'home_goals': home_goals, 'away_goals': away_goals, 'result': result,
                        'home_name': home_name, 'away_name': away_name}
        
        counter = 1
        for player_rating in fifa_ratings['home_ratings']:
            feature_dict['home_player_'+str(counter)] = player_rating
            counter += 1
        counter = 1
        for player_rating in fifa_ratings['away_ratings']:
            feature_dict['away_player_'+str(counter)] = player_rating
            counter +=1
            
        feature_vectors.append(feature_dict)

feature_df = sqlContext.createDataFrame(feature_vectors)


# In[507]:

# Once we have all games in a dataframe, we can calculate prior game statistics
prior_vectors = []
for row in feature_df.rdd.collect():
    prior_vector = row.asDict().copy()
    prior_vector.update(get_prior_game_statistics(row.home_id, row.away_id, row.time, feature_df))
    prior_vectors.append(prior_vector)

feature_df = sqlContext.createDataFrame(prior_vectors)


# In[508]:

# Sort it by time, then split our feature dataframe into a training and testing set
feature_df = feature_df.sort(feature_df['time'])
times = feature_df.select('time').collect()
nr_samples = feature_df.count()
test_fraction = 0.25

feature_df = feature_df.drop('away_goals').drop('home_goals') #  Clearly, home and away goals are pretty correlated to label

train_feature_df = feature_df.filter(feature_df['time'] <= split_time)
test_feature_df = feature_df.filter(feature_df['time'] > split_time)

train_feature_df = train_feature_df.drop('time')
test_feature_df = test_feature_df.drop('time')

assembler = VectorAssembler(
    inputCols=list(set(train_feature_df.columns) - set(['result', 'home_name', 'away_name'])),
    outputCol="features")

train_df = assembler.transform(train_feature_df)
test_df = assembler.transform(test_feature_df)

labelIndexer = StringIndexer(inputCol="result", outputCol="indexedResult").fit(feature_df)

train_df = labelIndexer.transform(train_df)
test_df = labelIndexer.transform(test_df)

label_mapping = dict(enumerate(labelIndexer.labels()))
reverse_mapping = {}
for key in label_mapping:
    reverse_mapping[label_mapping[key]] = key


# ## Dimensionality reduction
# 
# Feature selection is not really supported yet in mllib, therefore, we just applied dim reduction using PCA

# In[509]:

pca = PCA(inputCol="features", outputCol="pca", k=15).fit(train_df)

train_df = pca.transform(train_df)
test_df = pca.transform(test_df)


# ## Classification algorithms

# In[ ]:

rf = RandomForestClassifier(labelCol="indexedResult", featuresCol="pca", numTrees=5000)
#rf = RandomForestClassifier(labelCol="indexedResult", featuresCol="features", numTrees=5000)
model = rf.fit(train_df)


# ## Evaluation & results

# In[ ]:

label_to_str_map = {'2': 'HOME', '1': 'DRAW', '0': 'AWAY'}
str_to_labelmap = {'HOME': '2', 'DRAW': '1', 'AWAY': '0'}
predictions = model.transform(test_df).select("home_name", "away_name", "B365A", "B365D", "B365H", "probability", 
                                              "indexedResult")

length = test_df.count()
correct = 0
total_profit = 0
for prediction in predictions.collect():
    print (prediction['home_name'] + " vs. " + prediction['away_name'])
    outcome = label_to_str_map[label_mapping[prediction['indexedResult']]]
    print("\t OUTCOME \t\t= " + outcome)
    print("\t PREDICTION \t\t=" + " HOME: "+ str(round(prediction["probability"][reverse_mapping[str_to_labelmap['HOME']]], 4)) 
          + " DRAW: " + str(round(prediction["probability"][reverse_mapping[str_to_labelmap['DRAW']]], 4)) + 
          " AWAY: " + str(round(prediction["probability"][reverse_mapping[str_to_labelmap['AWAY']]], 4)))
    pred = label_to_str_map[label_mapping[np.argmax(prediction["probability"])]]
    if pred == outcome:
        correct += 1
        print("\t\t\t\t----> CORRECT")
    else:
        print("\t\t\t\t----> WRONG")
            
    ratings = {'HOME': round(prediction['B365H'],4), 'DRAW': round(prediction['B365D'],4), 'AWAY': round(prediction['B365A'],4)}                  
    print("\t RATINGS\t\t=" + " HOME: " + str(ratings['HOME']) + " DRAW: " + str(ratings['DRAW']) + " AWAY: " + str(ratings['AWAY']))
    
    profits = {'HOME': round(prediction["probability"][reverse_mapping[str_to_labelmap['HOME']]] * prediction['B365H'], 4),
               'DRAW': round(prediction["probability"][reverse_mapping[str_to_labelmap['DRAW']]] * prediction['B365D'], 4),
               'AWAY': round(prediction["probability"][reverse_mapping[str_to_labelmap['AWAY']]] * prediction['B365A'], 4)}
    print("\t EXPECTED PROFIT \t=" + " HOME: "+ str(profits['HOME']) + " DRAW: " + str(profits['DRAW']) + " AWAY: " + str(profits['AWAY']))
    
    bet = 0
    win = 0
    for profit in profits:
        if profits[profit] > 1: 
            bet += 1
            total_profit -= 1
            if profit == outcome:
                total_profit += ratings[outcome]
                win = ratings[outcome]
    print("\t\t\t\t----> BETTING " + str(bet) + " EUROS, WINNING " + str(win) + " EUROS")
    print("--------------------------------------------------------------------------------")

print("--------------------------------------------------------------------------------")
print()
print("ACCURACY \t= ", float(correct)/float(length))
print("TOTAL PROFIT \t= ", total_profit)


# In[ ]:



