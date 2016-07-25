from datetime import datetime

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine # database connection
import numpy as np

# TODO: More features: points gained in 5 last home games for home team / away games for away team,
# TODO: points gained in all last games, number of goals scored and conceded, stadium size,
# TODO: points gained in matches against same team (e.g. history of Club - Anderlecht)

def print_full(x):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

engine = create_engine("sqlite:///database.sqlite")

leagues = pd.read_sql_query('SELECT * FROM League;', engine)
teams = pd.read_sql_query('SELECT * FROM Team;', engine)
player_stats = pd.read_sql_query('SELECT * FROM Player_Stats;', engine)
players = pd.read_sql_query('SELECT * FROM Player;', engine)
matches = pd.read_sql_query('SELECT * FROM Match;', engine)

belgium_matches = pd.read_sql_query('SELECT * FROM Match WHERE league_id=1;', engine)
england_matches = pd.read_sql_query('SELECT * FROM Match WHERE league_id=1729;', engine).sort_index()

england_matches = england_matches.head()

result_entries = []
feature_entries = []
player_stats_feature_cols = 'overall_rating', 'potential', 'preferred_foot', 'attacking_work_rate', \
                            'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', \
                            'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', \
                            'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', \
                            'stamina', 'strength', 'long_shots', 'aggression', 'interceptions', 'positioning', 'vision',\
                            'penalties', 'marking', 'standing_tackle', 'sliding_tackle', 'gk_diving', 'gk_handling', \
                            'gk_kicking', 'gk_positioning', 'gk_reflexes'
for i in range(len(england_matches)):
    record = england_matches.iloc[i, :]
    print(str(i) + '/' + str(len(england_matches)))

    result_entry = {'match_id': record['match_api_id'], 'date': record['date'],
                    'home_team_goal': record['home_team_goal'], 'away_team_goal': record['away_team_goal'],
                    'B365H': record['B365H'], 'B365D': record['B365D'], 'B365A': record['B365A']}
    result_entries.append(result_entry)

    feature_entry = {}
    feature_entry['match_id'] = record['match_api_id']
    feature_entry['date'] = record['date']
    feature_entry['home_team'] = teams[teams.team_api_id == record['home_team_api_id']]['team_long_name'].values[0]
    feature_entry['away_team'] = teams[teams.team_api_id == record['away_team_api_id']]['team_long_name'].values[0]
    feature_entry['B365H'] = record['B365H']
    feature_entry['B365D'] = record['B365D']
    feature_entry['B365A'] = record['B365A']
    for k in range(1, 12):
        feature_entry['home_player_X'+str(k)] = record['home_player_X'+str(k)]
        feature_entry['home_player_Y'+str(k)] = record['home_player_Y'+str(k)]
        feature_entry['away_player_X'+str(k)] = record['away_player_X'+str(k)]
        feature_entry['away_player_Y'+str(k)] = record['away_player_Y'+str(k)]
        
        player_all_stats_home = player_stats[player_stats.player_api_id == record['home_player_' + str(k)]]
        min = 99999999999999
        closest_stat_record = None
        for j in range(len(player_all_stats_home)):
            stat_record = player_all_stats_home.iloc[j, :]
            if abs((datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S') -
                      datetime.strptime(stat_record['date_stat'], '%Y-%m-%d %H:%M:%S')).total_seconds()) < min:
                closest_stat_record = stat_record
        
        for stat_feature_col in player_stats_feature_cols:
            feature_entry[stat_feature_col+'_home_'+str(k)] = stat_record[stat_feature_col]
            
        player_all_stats_away = player_stats[player_stats.player_api_id == record['away_player_' + str(k)]]
        min = 99999999999999
        closest_stat_record = None
        for j in range(len(player_all_stats_away)):
            stat_record = player_all_stats_away.iloc[j, :]
            if abs((datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S') -
                      datetime.strptime(stat_record['date_stat'], '%Y-%m-%d %H:%M:%S')).total_seconds()) < min:
                closest_stat_record = stat_record
        
        for stat_feature_col in player_stats_feature_cols:
            feature_entry[stat_feature_col+'_away_'+str(k)] = stat_record[stat_feature_col]
    feature_entries.append(feature_entry)

feature_df = pd.DataFrame(feature_entries)
result_df = pd.DataFrame(result_entries)
# feature_df.to_csv('kaggle_features_v1.csv')
# result_df.to_csv('kaggle_labels_v2.csv')
