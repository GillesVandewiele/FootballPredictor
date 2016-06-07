from datetime import datetime

from lxml import etree
import pandas as pd


def files_to_df(file_paths):
    df = None
    for file_path in file_paths:
        print('reading...', file_path)
        df = get_matches_info(xml_to_tree(file_path), df)

    print(df)


def xml_to_tree(path):
    return etree.parse(path)


def get_matches_info(tree, df):
    # TODO: currently not extracted: the sub details and goal details (in which minute, which (who) sub/goal was made)
    # TODO: subs are useful: in real-life you need to  
    # TODO: goals too (e.g: having a goal getter on the field is mostly an advantage)
    new_records = []
    for match in tree.getroot().iter('Match'):
        if match is not None:
            new_record = {}
            new_record['date'] = datetime.strptime(match.find('Date').text[:-6], '%Y-%m-%dT%H:%M:%S')

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
            new_record['home_gk'] = match.find('HomeLineupGoalkeeper').text[1:].split('; ') if match.find(
                'HomeLineupGoalkeeper') is not None and match.find('HomeLineupGoalkeeper').text is not None else None
            new_record['home_def'] = match.find('HomeLineupDefense').text[1:].split('; ') if match.find(
                'HomeLineupDefense') is not None and match.find('HomeLineupDefense').text is not None else None
            new_record['home_mid'] = match.find('HomeLineupMidfield').text[1:].split('; ') if match.find(
                'HomeLineupMidfield') is not None and match.find('HomeLineupMidfield').text is not None else None
            new_record['home_atk'] = match.find('HomeLineupForward').text[1:].split('; ') if match.find(
                'HomeLineupForward') is not None and match.find('HomeLineupForward').text is not None else None
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
            new_record['away_gk'] = match.find('AwayLineupGoalkeeper').text[1:].split('; ') if match.find(
                'AwayLineupGoalkeeper') is not None and match.find('AwayLineupGoalkeeper').text is not None else None
            new_record['away_def'] = match.find('AwayLineupDefense').text[1:].split('; ') if match.find(
                'AwayLineupDefense') is not None and match.find('AwayLineupDefense').text is not None else None
            new_record['away_mid'] = match.find('AwayLineupMidfield').text[1:].split('; ') if match.find(
                'AwayLineupMidfield') is not None and match.find('AwayLineupMidfield').text is not None else None
            new_record['away_atk'] = match.find('AwayLineupForward').text[1:].split('; ') if match.find(
                'AwayLineupForward') is not None and match.find('AwayLineupForward').text is not None else None
            new_record['away_FT_goals'] = match.find('AwayGoals').text
            new_record['away_FT_goals'] = match.find('HalfTimeAwayGoals').text

            new_records.append(new_record)

    if df is not None:
        return pd.concat([df, pd.DataFrame.from_records(new_records)])
    else:
        return pd.DataFrame.from_records(new_records)


files_to_df(['FootballData/scotland1011.xml', 'FootballData/scotland1112.xml', 'FootballData/scotland1213.xml',
             'FootballData/scotland1314.xml', 'FootballData/scotland1415.xml', 'FootballData/scotland1516.xml'])
