from datetime import datetime

from lxml import etree
import pandas as pd


def files_to_odds(file_paths):
    df = None
    for file_path in file_paths:
        print('reading...', file_path)
        betting_odds = pd.read_csv(file_path)
        if df is None:
            df = betting_odds
        else:
            df = pd.concat([df, betting_odds])

    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%d/%m/%y'))
    df['HomeTeam'] = df['HomeTeam'].replace('Dundee', 'Dundee FC')
    df['AwayTeam'] = df['AwayTeam'].replace('Dundee', 'Dundee FC')
    df = df.rename(columns={'Date': 'date', 'AwayTeam': 'away_team', 'HomeTeam': 'home_team'})
    return df[['date', 'away_team', 'home_team', 'B365A', 'B365H', 'B365D']]

def files_to_df(file_paths):
    df = None
    for file_path in file_paths:
        print('reading...', file_path)
        df = get_matches_info(xml_to_tree(file_path), df)

    df['home_team'] = df['home_team'].replace('St.Johnstone', 'St Johnstone')
    df['away_team'] = df['away_team'].replace('St.Johnstone', 'St Johnstone')
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)


def xml_to_tree(path):
    return etree.parse(path)


def get_matches_info(tree, df):
    # TODO: currently not extracted: the sub details and goal details (in which minute, which (who) sub/goal was made)
    # TODO: subs are useful: in real-life you need the previous game formation (you don't know it for the game to come)
    # TODO: goals too (e.g: having a goal getter on the field is mostly an advantage)
    new_records = []
    for match in tree.getroot().iter('Match'):
        if match is not None:
            new_record = {}
            date = datetime.strptime(match.find('Date').text[:-6], '%Y-%m-%dT%H:%M:%S')
            new_record['date'] = datetime(year=date.year, month=date.month, day=date.day)

            # Home Team
            new_record['home_team'] = match.find('HomeTeam').text
            new_record['home_corners'] = match.find('HomeCorners').text
            new_record['home_shots'] = match.find('HomeShots').text
            new_record['home_shots_target'] = match.find('HomeShotsOnTarget').text
            new_record['home_fouls'] = match.find('HomeFouls').text
            new_record['home_yellow'] = match.find('HomeYellowCards').text
            new_record['home_red'] = match.find('HomeRedCards').text
            new_record['home_formation'] = match.find('HomeTeamFormation').text if match.find(
                'HomeTeamFormation') is not None else None

            new_record['home_gk'] = [word.lstrip() for word in match.find('HomeLineupGoalkeeper').text.split(';')] \
                if match.find('HomeLineupGoalkeeper') is not None and match.find('HomeLineupGoalkeeper').text is not \
                                                                      None else None

            new_record['home_def'] = [word.lstrip() for word in match.find('HomeLineupDefense').text.split(';')] \
                if match.find('HomeLineupDefense') is not None and match.find('HomeLineupDefense').text is not \
                                                                      None else None
            if new_record['home_def'] and new_record['home_def'][-1] == '': del new_record['home_def'][-1]

            new_record['home_mid'] = [word.lstrip() for word in match.find('HomeLineupMidfield').text.split(';')] \
                if match.find('HomeLineupMidfield') is not None and match.find('HomeLineupMidfield').text is not \
                                                                      None else None
            if new_record['home_mid'] and new_record['home_mid'][-1] == '': del new_record['home_mid'][-1]

            new_record['home_atk'] = [word.lstrip() for word in match.find('HomeLineupForward').text.split(';')] \
                if match.find('HomeLineupForward') is not None and match.find('HomeLineupForward').text is not \
                                                                      None else None
            if new_record['home_atk'] and new_record['home_atk'][-1] == '': del new_record['home_atk'][-1]

            new_record['home_FT_goals'] = match.find('HomeGoals').text
            new_record['home_FT_goals'] = match.find('HalfTimeHomeGoals').text

            # Away Team
            new_record['away_team'] = match.find('AwayTeam').text
            new_record['away_corners'] = match.find('AwayCorners').text
            new_record['away_shots'] = match.find('AwayShots').text
            new_record['away_shots_target'] = match.find('AwayShotsOnTarget').text
            new_record['away_fouls'] = match.find('AwayFouls').text
            new_record['away_yellow'] = match.find('AwayYellowCards').text
            new_record['away_red'] = match.find('AwayRedCards').text
            new_record['away_formation'] = match.find('AwayTeamFormation').text if match.find(
                'AwayTeamFormation') is not None else None
            new_record['away_gk'] = [word.lstrip() for word in match.find('AwayLineupGoalkeeper').text.split(';')] \
                if match.find('AwayLineupGoalkeeper') is not None and match.find('AwayLineupGoalkeeper').text is not \
                                                                      None else None
            new_record['away_def'] = [word.lstrip() for word in match.find('AwayLineupDefense').text.split(';')] \
                if match.find('AwayLineupDefense') is not None and match.find('AwayLineupDefense').text is not \
                                                                      None else None
            if new_record['away_def'] and new_record['away_def'][-1] == '': del new_record['away_def'][-1]

            new_record['away_mid'] = [word.lstrip() for word in match.find('AwayLineupMidfield').text.split(';')] \
                if match.find('AwayLineupMidfield') is not None and match.find('AwayLineupMidfield').text is not \
                                                                      None else None
            if new_record['away_mid'] and new_record['away_mid'][-1] == '': del new_record['away_mid'][-1]

            new_record['away_atk'] = [word.lstrip() for word in match.find('AwayLineupForward').text.split(';')] \
                if match.find('AwayLineupForward') is not None and match.find('AwayLineupForward').text is not \
                                                                      None else None
            if new_record['away_atk'] and new_record['away_atk'][-1] == '': del new_record['away_atk'][-1]

            new_record['away_FT_goals'] = match.find('AwayGoals').text
            new_record['away_FT_goals'] = match.find('HalfTimeAwayGoals').text

            new_records.append(new_record)

    if df is not None:
        return pd.concat([df, pd.DataFrame.from_records(new_records)])
    else:
        return pd.DataFrame.from_records(new_records)


df = files_to_df(['FootballData/scotland1011.xml', 'FootballData/scotland1112.xml', 'FootballData/scotland1213.xml',
                  'FootballData/scotland1314.xml', 'FootballData/scotland1415.xml', 'FootballData/scotland1516.xml'])
odds_df = files_to_odds(['FootballData/scotland1112_odds.csv', 'FootballData/scotland1213_odds.csv',
                         'FootballData/scotland1314_odds.csv', 'FootballData/scotland1415_odds.csv',
                         'FootballData/scotland1516_odds.csv'])

joined_df = pd.merge(df, odds_df, how='left', on=['date', 'home_team', 'away_team'])

print('Dataframe ready... Print null value counts')
print(joined_df.isnull().sum())
print('--------------------------------------')

joined_df.to_csv('features_dirty.csv')
