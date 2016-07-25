from datetime import datetime

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine  # database connection
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

# england_matches = england_matches.sort_values(by='date').head(200)

result_entries = []
feature_entries = []
player_stats_feature_cols = 'overall_rating', 'potential', 'preferred_foot', 'attacking_work_rate', \
                            'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', \
                            'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', \
                            'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', \
                            'stamina', 'strength', 'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', \
                            'penalties', 'marking', 'standing_tackle', 'sliding_tackle', 'gk_diving', 'gk_handling', \
                            'gk_kicking', 'gk_positioning', 'gk_reflexes'

# Creating a result dataframe
for i in range(len(england_matches)):
    record = england_matches.iloc[i, :]
    print('RESULTS', str(i) + '/' + str(len(england_matches) - 1))

    home_goals = int(record['home_team_goal'])
    away_goals = int(record['away_team_goal'])
    if home_goals > away_goals:
        result = 'HOME'
    elif away_goals > home_goals:
        result = 'AWAY'
    else:
        result = 'DRAW'
    home_team = teams[teams.team_api_id == record['home_team_api_id']]['team_long_name'].values[0]
    away_team = teams[teams.team_api_id == record['away_team_api_id']]['team_long_name'].values[0]
    result_entry = {'match_id': record['match_api_id'], 'date': datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S'),
                    'home_team_goal': record['home_team_goal'], 'away_team_goal': record['away_team_goal'],
                    'B365H': record['B365H'], 'B365D': record['B365D'], 'B365A': record['B365A'], 'result': result,
                    'home_team': home_team, 'away_team': away_team}
    result_entries.append(result_entry)

result_df = pd.DataFrame(result_entries)


def get_historical_data(_home_team, _away_team, date, _result_df, n_games=5):
    # Get goals scored/conceded in n_games latest (home/away/all) games
    # Get points obtained in n_games latest (home/away/all) games
    # Get the points obtained by the home and away team in the n_games latest mutual games
    home_home_games = _result_df[(_result_df.home_team == _home_team) & (_result_df.date < date)] \
        .sort_values(by='date').tail(n_games)
    home_home_goals_scored = sum(home_home_games['home_team_goal'])
    home_home_goals_conceded = sum(home_home_games['away_team_goal'])
    home_home_points = sum(home_home_games['result'] == 'HOME') * 3 + sum(home_home_games['result'] == 'DRAW')

    home_away_games = _result_df[(_result_df.away_team == _home_team) & (_result_df.date < date)] \
        .sort_values(by='date').tail(n_games)
    home_all_goals_scored = sum(home_away_games['away_team_goal']) + home_home_goals_scored
    home_all_goals_conceded = sum(home_away_games['home_team_goal']) + home_home_goals_conceded
    home_all_points = sum(home_away_games['result'] == 'AWAY') * 3 + sum(home_away_games['result'] == 'DRAW') + \
                      home_home_points

    away_away_games = _result_df[(_result_df.away_team == _away_team) & (_result_df.date < date)] \
        .sort_values(by='date').tail(n_games)
    away_away_goals_scored = sum(away_away_games['away_team_goal'])
    away_away_goals_conceded = sum(away_away_games['home_team_goal'])
    away_away_points = sum(away_away_games['result'] == 'AWAY') * 3 + sum(away_away_games['result'] == 'DRAW')

    away_home_games = _result_df[(_result_df.home_team == _away_team) & (_result_df.date < date)] \
        .sort_values(by='date').tail(n_games)
    away_all_goals_scored = sum(away_home_games['home_team_goal']) + away_away_goals_scored
    away_all_goals_conceded = sum(away_home_games['away_team_goal']) + away_away_goals_conceded
    away_all_points = sum(away_home_games['result'] == 'HOME') * 3 + sum(away_home_games['result'] == 'DRAW') + \
                      away_away_points

    same_mutual_games = _result_df[(_result_df.home_team == _home_team) & (_result_df.away_team == _away_team) &
                                   (_result_df.date < date)].sort_values(by='date').tail(n_games)
    home_mutual_points = sum(same_mutual_games['result'] == 'HOME') * 3 + sum(same_mutual_games['result'] == 'DRAW')
    home_mutual_goals_scored = sum(same_mutual_games['home_team_goal'])
    home_mutual_goals_conceded = sum(same_mutual_games['away_team_goal'])
    away_mutual_points = sum(same_mutual_games['result'] == 'AWAY') * 3 + sum(same_mutual_games['result'] == 'DRAW')
    away_mutual_goals_scored = home_mutual_goals_conceded
    away_mutual_goals_conceded = home_mutual_goals_scored

    return home_home_goals_scored, home_home_goals_conceded, home_all_goals_scored, home_all_goals_conceded, \
           home_home_points, home_all_points, away_away_goals_scored, away_away_goals_conceded, \
           away_all_goals_scored, away_all_goals_conceded, away_away_points, away_all_points, home_mutual_points, \
           home_mutual_goals_scored, home_mutual_goals_conceded, away_mutual_points, away_mutual_goals_scored, \
           away_mutual_goals_conceded


# Creating the features dataframe
for i in range(len(england_matches)):
    record = england_matches.iloc[i, :]
    print('FEATURES', str(i) + '/' + str(len(england_matches) - 1))
    feature_entry = {}
    feature_entry['match_id'] = record['match_api_id']
    feature_entry['date'] = datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S')
    feature_entry['home_team'] = teams[teams.team_api_id == record['home_team_api_id']]['team_long_name'].values[0]
    feature_entry['away_team'] = teams[teams.team_api_id == record['away_team_api_id']]['team_long_name'].values[0]
    feature_entry['B365H'] = record['B365H']
    feature_entry['B365D'] = record['B365D']
    feature_entry['B365A'] = record['B365A']
    # Get some historical data
    home_home_goals_scored, home_home_goals_conceded, home_all_goals_scored, home_all_goals_conceded, \
    home_home_points, home_all_points, away_away_goals_scored, away_away_goals_conceded, \
    away_all_goals_scored, away_all_goals_conceded, away_away_points, away_all_points, home_mutual_points, \
    home_mutual_goals_scored, home_mutual_goals_conceded, away_mutual_points, away_mutual_goals_scored, \
    away_mutual_goals_conceded = get_historical_data(feature_entry['home_team'], feature_entry['away_team'],
                                                     feature_entry['date'], result_df.copy())
    feature_entry['home_home_goals_scored'] = home_home_goals_scored
    feature_entry['home_home_goals_conceded'] = home_home_goals_conceded
    feature_entry['home_all_goals_scored'] = home_all_goals_scored
    feature_entry['home_all_goals_conceded'] = home_all_goals_conceded
    feature_entry['home_home_points'] = home_home_points
    feature_entry['home_all_points'] = home_all_points
    feature_entry['away_away_goals_scored'] = away_away_goals_scored
    feature_entry['away_away_goals_conceded'] = away_away_goals_conceded
    feature_entry['away_all_goals_scored'] = away_all_goals_scored
    feature_entry['away_all_goals_conceded'] = away_all_goals_conceded
    feature_entry['away_away_points'] = away_away_points
    feature_entry['away_all_points'] = away_all_points
    feature_entry['home_mutual_points'] = home_mutual_points
    feature_entry['home_mutual_goals_scored'] = home_mutual_goals_scored
    feature_entry['home_mutual_goals_conceded'] = home_mutual_goals_conceded
    feature_entry['away_mutual_points'] = away_mutual_points
    feature_entry['away_mutual_goals_scored'] = away_mutual_goals_scored
    feature_entry['away_mutual_goals_conceded'] = away_mutual_goals_conceded

    # Get some player stats
    for k in range(1, 12):
        feature_entry['home_player_X' + str(k)] = record['home_player_X' + str(k)]
        feature_entry['home_player_Y' + str(k)] = record['home_player_Y' + str(k)]
        feature_entry['away_player_X' + str(k)] = record['away_player_X' + str(k)]
        feature_entry['away_player_Y' + str(k)] = record['away_player_Y' + str(k)]

        player_all_stats_home = player_stats[player_stats.player_api_id == record['home_player_' + str(k)]]
        min = 99999999999999
        closest_stat_record = None
        for j in range(len(player_all_stats_home)):
            stat_record = player_all_stats_home.iloc[j, :]
            if abs((datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S') -
                        datetime.strptime(stat_record['date_stat'], '%Y-%m-%d %H:%M:%S')).total_seconds()) < min:
                closest_stat_record = stat_record

        for stat_feature_col in player_stats_feature_cols:
            feature_entry[stat_feature_col + '_home_' + str(k)] = stat_record[stat_feature_col]

        player_all_stats_away = player_stats[player_stats.player_api_id == record['away_player_' + str(k)]]
        min = 99999999999999
        closest_stat_record = None
        for j in range(len(player_all_stats_away)):
            stat_record = player_all_stats_away.iloc[j, :]
            if abs((datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S') -
                        datetime.strptime(stat_record['date_stat'], '%Y-%m-%d %H:%M:%S')).total_seconds()) < min:
                closest_stat_record = stat_record

        for stat_feature_col in player_stats_feature_cols:
            feature_entry[stat_feature_col + '_away_' + str(k)] = stat_record[stat_feature_col]

    feature_entries.append(feature_entry)

feature_df = pd.DataFrame(feature_entries)
feature_df.to_csv('kaggle_features_v2.csv')
result_df.to_csv('kaggle_labels_v2.csv')
