import pandas as pd
import utils
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
import optuna
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib
# import networkx as nx
from tqdm import tqdm
"""
first year of data is 2003

TODO: Fix the seed from the w01 to 1
"""
def create_regression_labels(df):
    """
    Above 0.5: Team 1 is more likely to win.
    Below 0.5: Team 0 is more likely to win.
    Exactly 0.5: Its an even matchup.
    """
    # score_diff = (df['team_1'] - df['team_0']).values.reshape(-1, 1)
    # data_tran = MinMaxScaler(feature_range=(0,1)).fit_transform(score_diff)
    data_tran = ((df['team_1']) / (df['team_1'] + (df['team_0']))).values.reshape(-1, 1)
    # plt.hist(data_tran,color='red')
    # plt.hist(df['team_0'])
    # plt.hist(df['team_1'])
    # plt.hist(label_scaled,color='blue',alpha=0.5)
    # plt.show()
    return data_tran
 
def get_team_stats(team_mapping_df, conference_df, team_name, season, seed):
    team_id = team_mapping_df[team_mapping_df['TeamNameSpelling'] == team_name]['TeamID'].values[0]
    sub_df = season[((season['WTeamID'] == team_id) | (season['LTeamID'] == team_id))]
    sub_df = sub_df.drop(columns=['WTeamID', 'LTeamID', 'WLoc', 'Season', 'DayNum', 'NumOT'])

    team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
    team_var = sub_df.var().add_suffix('_var').to_frame().T
    team_stats = pd.concat([team_avg, team_var], axis=1)

    conference = conference_df[(conference_df['Season'] == 2025) & (conference_df['TeamID'] == team_id)]['ConfID'].values[0]

    team_stats['conference'] = conference
    team_stats['seed'] = seed
    return team_stats

def predict_bracket(matchups, team_mapping_df, reg_season, conference_df, final_model, pca_inst, min_max, region):
    os.makedirs('results',exist_ok=True)
    round_val = 1
    while len(matchups) > 1:
        n = len(matchups)
        mid = n // 2
        left_list = []
        right_list = []
        for i in range(mid):
            left = matchups[i]
            right = matchups[n - 1 - i]

            #left matchups
            team_1_stats = get_team_stats(team_mapping_df, conference_df, left[0], reg_season, left[2]).add_suffix('_team_1')
            team_0_stats = get_team_stats(team_mapping_df, conference_df, left[1], reg_season, left[3]).add_suffix('_team_0')
            curr_matchup = pd.concat([team_1_stats, team_0_stats], axis=1)
            # print(curr_matchup)
            # print(curr_matchup[min_max.feature_names_in_])
            # input()
            curr_matchup = pca_inst.transform(min_max.transform(curr_matchup))
            team_1_pred = final_model.predict(curr_matchup)[0]
            (winner_left, seed_left) = (left[0], left[2]) if team_1_pred > 0.5 else (left[1], left[3])
            left_list.append((winner_left, seed_left))
            
            #right matchups
            team_1_stats = get_team_stats(team_mapping_df, conference_df, right[0], reg_season, right[2]).add_suffix('_team_1')
            team_0_stats = get_team_stats(team_mapping_df, conference_df, right[1], reg_season, right[3]).add_suffix('_team_0')
            curr_matchup = pd.concat([team_1_stats, team_0_stats], axis=1)
            curr_matchup = pca_inst.transform(min_max.transform(curr_matchup))
            team_1_pred = final_model.predict(curr_matchup)[0]
            (winner_right, seed_right) = (right[0], right[2]) if team_1_pred > 0.5 else (right[1], right[3])
            right_list.append((winner_right, seed_right))
        #construct new matchup list
        matchups = []
        if len(left_list) > 1 and len(right_list) > 1:
            #leftside
            for i in range(0, len(left_list), 2):
                if i + 1 < len(left_list):
                    matchups.append((left_list[i][0], left_list[i+1][0], left_list[i][1], left_list[i+1][1]))
            #rightside
            for i in range(0, len(right_list), 2):
                if i + 1 < len(right_list):
                    matchups.append((right_list[i][0], right_list[i+1][0], right_list[i][1], right_list[i+1][1]))
        else:
             matchups.append((left_list[0][0],right_list[0][0],left_list[0][1],right_list[0][1]))
        #save round data
        with open(f'results/{region}_round_{round_val}_pca.txt', 'w') as file:
                for matchup in matchups:
                    file.write(f"{matchup[0]} vs {matchup[1]}\n")
        round_val += 1
    
    #get the final winner of the region
    team_1_stats = get_team_stats(team_mapping_df, conference_df, matchups[0][0], reg_season, matchups[0][2]).add_suffix('_team_1')
    team_0_stats = get_team_stats(team_mapping_df, conference_df, matchups[0][1], reg_season, matchups[0][3]).add_suffix('_team_0')
    curr_matchup = pd.concat([team_1_stats, team_0_stats], axis=1)
    
    curr_matchup = pca_inst.transform(min_max.transform(curr_matchup))
    team_1_pred = final_model.predict(curr_matchup)[0]
    (winner_region, seed_region) = (matchups[0][0], matchups[0][2]) if team_1_pred > 0.5 else (matchups[0][1], matchups[0][3])
    with open(f'results/{region}_winner_pca.txt', 'w') as file:
        for matchup in matchups:
            file.write(f"{winner_region}\n")
    return (winner_region, seed_region)

def main():
    tourney_results = pd.read_csv('ncaadata/MNCAATourneyCompactResults.csv')
    reg_season = pd.read_csv('ncaadata/MRegularSeasonDetailedResults.csv')

    conference_df = utils.conference_encoder(pd.read_csv('ncaadata/MTeamConferences.csv'),reg_season['Season'].iloc[0])
    seed_df = utils.seed_encoder(pd.read_csv('ncaadata/MNCAATourneySeeds.csv'),reg_season['Season'].iloc[0])
    final_dataset = pd.DataFrame()
    if not os.path.exists('ncaadata/training_data.csv'):
        for iteration,row in tourney_results.iterrows():
            curr_season = row['Season']
            curr_w_team_ID = row['WTeamID']
            curr_l_team_ID = row['LTeamID']
            curr_w_score = row['WScore']
            curr_l_score = row['LScore']
            if curr_season >= reg_season['Season'].iloc[0]:

                #winner
                #stats
                sub_df = reg_season[
                    (reg_season['Season'] == curr_season) & 
                    ((reg_season['WTeamID'] == curr_w_team_ID) | (reg_season['LTeamID'] == curr_w_team_ID))
                ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
                team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
                team_var = sub_df.var().add_suffix('_var').to_frame().T
                team_stats_final_winner = pd.concat([team_avg, team_var],axis=1)

                #conference
                conference_winner = conference_df[(conference_df['Season'] == curr_season) &
                            (conference_df['TeamID'] == curr_w_team_ID)]['ConfID'].values[0]

                #seed
                seed_winner = seed_df[(seed_df['Season'] == curr_season) &
                            (seed_df['TeamID'] == curr_w_team_ID)]['seedID'].values[0]
                
                #combine df
                team_stats_final_winner['conference'] = conference_winner
                team_stats_final_winner['seed'] = seed_winner
                team_stats_final_winner = team_stats_final_winner.add_suffix('_team_1')


                #loser
                #stats
                sub_df = reg_season[
                    (reg_season['Season'] == curr_season) & 
                    ((reg_season['WTeamID'] == curr_l_team_ID) | (reg_season['LTeamID'] == curr_l_team_ID))
                ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
                team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
                team_var = sub_df.var().add_suffix('_var').to_frame().T
                team_stats_final_loser = pd.concat([team_avg, team_var],axis=1)

                #conference
                conference_loser = conference_df[(conference_df['Season'] == curr_season) &
                            (conference_df['TeamID'] == curr_l_team_ID)]['ConfID'].values[0]
                
                #seeds
                seed_loser = seed_df[(seed_df['Season'] == curr_season) &
                            (seed_df['TeamID'] == curr_l_team_ID)]['seedID'].values[0]
                
                #combine df
                team_stats_final_loser['conference'] = conference_loser
                team_stats_final_loser['seed'] = seed_loser
                team_stats_final_loser = team_stats_final_loser.add_suffix('_team_0')


                #labels for both teams
                # team_stats_final_winner['team_1'] = 1 #win
                # team_stats_final_loser['team_0'] = 0 #loss
                team_stats_final_winner['team_1'] = curr_w_score #win
                team_stats_final_loser['team_0'] = curr_l_score #loss

                #data agg 
                curr_matchup = pd.concat([team_stats_final_winner,team_stats_final_loser],axis=1)

                final_dataset = pd.concat([final_dataset,curr_matchup])
        final_dataset.to_csv('ncaadata/training_data.csv',index=False)
    else:
        final_dataset = pd.read_csv('ncaadata/training_data.csv')
        # df_swapped = utils.swap_team_features(final_dataset)
        # X = df_swapped.drop(columns=['team_0','team_1'])
        # print(f'before pca: {X.shape}')
        # X = PCA(n_components=0.95).fit_transform(X)
        # print(f'after pca: {X.shape}')
        # exit()

    #ML analysis
    if not os.path.exists('models/regression_model_pca.joblib'):
        df_swapped = utils.swap_team_features(final_dataset)
        #create regression labels
        # #create one hot encoded data
        # y = df_swapped[['team_0', 'team_1']].values
        y = create_regression_labels(df_swapped)
        X = df_swapped.drop(columns=['team_0','team_1'])

        #min max before pca
        min_max = MinMaxScaler(feature_range=(0,1))
        X_min_max = min_max.fit_transform(X)
        pca_inst = PCA(n_components=0.975)
        x_pca = pca_inst.fit_transform(X_min_max)
        X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)

        joblib.dump(min_max,'models/min_max.joblib')
        joblib.dump(pca_inst,'models/pca.joblib')

        # #dumb
        # y_train = np.argmax(y_train, axis=1)
        # y_test = np.argmax(y_test, axis=1)

        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }

            model = xgb.XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
            return -np.mean(scores)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_trial.params
        final_model = xgb.XGBRegressor(**best_params,early_stopping_rounds=12)

        final_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
                        verbose=True)
        os.makedirs('models',exist_ok=True)
        joblib.dump(final_model,'models/regression_model_pca.joblib')
        results = final_model.evals_result()
        # Evaluate model
        y_pred = final_model.predict(X_test)
        final_acc = mean_absolute_error(y_test, y_pred)

        print(f"Final MAE: {final_acc:.4f}")

        train_loss = results['validation_0']['rmse']
        val_loss = results['validation_1']['rmse']

        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label='Training rmse')
        plt.plot(val_loss, label='Validation rmse')
        plt.xlabel('Epoch')
        plt.ylabel('Log rmse')
        plt.title('Training and Validation rmse over Epochs')
        plt.legend()
        os.makedirs('figures',exist_ok=True)
        plt.savefig('figures/training_curve_pca.png')

        #feature importance
        importance = final_model.get_booster().get_score(importance_type="weight")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_importance[:20])

        plt.figure(figsize=(20, 8))
        plt.barh(features, scores, color="skyblue")
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        #highest on top
        plt.gca().invert_yaxis()
        plt.savefig('figures/feature_importance_pca.png', bbox_inches='tight')
        plt.close()
    else:
        final_model = joblib.load('models/regression_model_pca.joblib')
        min_max = joblib.load('models/min_max.joblib')
        pca_inst = joblib.load('models/pca.joblib')

    #predictions - past
    run_hist = False
    if run_hist:
        season_val = 2024
        curr_season = reg_season[reg_season['Season'] == season_val]
        curr_tourney = tourney_results[tourney_results['Season'] == season_val]
        curr_tourney = curr_tourney.reset_index(drop=True)
        result = 0
        game = 1
        for iteration, row in curr_tourney.iterrows():
            curr_w_team_ID, curr_l_team_ID = row['WTeamID'], row['LTeamID']
            curr_w_score = row['WScore']
            curr_l_score = row['LScore']
            #winner
            #stats
            sub_df = curr_season[
                ((curr_season['WTeamID'] == curr_w_team_ID) | (curr_season['LTeamID'] == curr_w_team_ID))
            ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
            team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
            team_var = sub_df.var().add_suffix('_var').to_frame().T
            team_stats_final_winner = pd.concat([team_avg, team_var],axis=1)

            #conference
            conference_winner = conference_df[(conference_df['Season'] == season_val) &
                        (conference_df['TeamID'] == curr_w_team_ID)]['ConfID'].values[0]

            #seed
            seed_winner = seed_df[(seed_df['Season'] == season_val) &
                        (seed_df['TeamID'] == curr_w_team_ID)]['seedID'].values[0]
            
            #combine df
            team_stats_final_winner['conference'] = conference_winner
            team_stats_final_winner['seed'] = seed_winner
            team_stats_final_winner = team_stats_final_winner.add_suffix('_team_1')

            #loser
            #stats
            sub_df = curr_season[
                ((curr_season['WTeamID'] == curr_l_team_ID) | (curr_season['LTeamID'] == curr_l_team_ID))
            ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
            team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
            team_var = sub_df.var().add_suffix('_var').to_frame().T
            team_stats_final_loser = pd.concat([team_avg, team_var],axis=1)

            #conference
            conference_loser = conference_df[(conference_df['Season'] == season_val) &
                        (conference_df['TeamID'] == curr_l_team_ID)]['ConfID'].values[0]

            #seed
            seed_loser = seed_df[(seed_df['Season'] == season_val) &
                        (seed_df['TeamID'] == curr_l_team_ID)]['seedID'].values[0]
            
            #combine df
            team_stats_final_loser['conference'] = conference_loser
            team_stats_final_loser['seed'] = seed_loser
            team_stats_final_loser = team_stats_final_loser.add_suffix('_team_0')

            curr_matchup = pd.concat([team_stats_final_winner,team_stats_final_loser],axis=1)

            #transform
            curr_matchup = pca_inst.transform(min_max.transform(curr_matchup))

            actual = 1 if curr_w_score > curr_l_score else 0
            predict = final_model.predict(curr_matchup)
            if predict > 0.5:
                team_1 = 1
            else:
                team_1 = 0
            result += 1 if actual == team_1 else 0 
            game +=1
        print(f'Accuracy for {season_val}: {result/game}')

    ################### predictions this season ############################
    team_mapping_df = pd.read_csv('ncaadata/MTeamSpellings.csv')
    curr_season = reg_season[reg_season['Season'] == 2025]
    # initial_matchups = [
    #     # South Region (Auburn No. 1)
    #     ('alabama state', 'auburn', 16, 1),
    #     ('creighton', 'louisville', 9, 8),
    #     ('michigan', 'san diego', 5, 12),
    #     ('texas a&m', 'yale', 4, 13),
    #     ('ole miss', 'north carolina', 6, 11),
    #     ('iowa state', 'lipscomb', 3, 14),

    #     # East Region (Duke No. 1)
    #     ('duke', 'american', 1, 16),
    #     ('mississippi state', 'baylor', 8, 9),
    #     ('oregon', 'liberty', 5, 12),
    #     ('arizona', 'akron', 4, 13),
    #     ('byu', 'virginia commonwealth', 6, 11),
    #     ('wisconsin', 'montana', 3, 14),
    #     ("saint mary's", 'vanderbilt', 7, 10),
    #     ("alabama", 'robert morris', 2, 15),
        
    #     # West Region (Florida No. 1)
    #     ('florida', 'norfolk state', 1, 16),
    #     ('uconn', 'oklahoma', 8, 9),
    #     ('memphis', 'colorado state', 5, 12),
    #     ('maryland', 'grand canyon', 4, 13),
    #     ('missouri', 'drake', 6, 11),
    #     ('texas tech', 'unc wilmington', 3, 14),
    #     ('kansas', 'arkansas', 7, 10),
    #     ("st john's", 'nebraska-omaha', 2, 15),

    #     # Midwest Region (Houston No. 1)
    #     ('houston', 'southern illinois-edwardsville', 1, 16),
    #     ('gonzaga', 'georgia', 8, 9),
    #     ('clemson', 'mcneese', 5, 12),
    #     ('purdue', 'high point', 4, 13),
    #     ('illinois', 'texas', 6, 11),
    #     ('kentucky', 'troy', 3, 14),
    #     ('ucla', 'utah-state', 7, 10),
    #     ('tennessee', 'wofford', 2, 15)
    # ]

    south_region = [
        # South Region (Auburn No. 1)
        ('alabama state', 'auburn', 16, 1),
        ('creighton', 'louisville', 9, 8),
        ('michigan', 'san diego', 5, 12),
        ('texas a&m', 'yale', 4, 13),
        ('ole miss', 'north carolina', 6, 11),
        ('iowa state', 'lipscomb', 3, 14),
        ('marquette', 'new mexico', 7, 10),
        ('michigan state', 'bryant', 2, 15),
    ]

    south_winner = predict_bracket(south_region, team_mapping_df, curr_season, conference_df, final_model, pca_inst, min_max, 'south')
    
    east_region = [
        # East Region (Duke No. 1)
        ('duke', 'american', 1, 16),
        ('mississippi state', 'baylor', 8, 9),
        ('oregon', 'liberty', 5, 12),
        ('arizona', 'akron', 4, 13),
        ('byu', 'virginia commonwealth', 6, 11),
        ('wisconsin', 'montana', 3, 14),
        ("saint mary's", 'vanderbilt', 7, 10),
        ("alabama", 'robert morris', 2, 15),
    ]
    east_winner = predict_bracket(east_region, team_mapping_df, curr_season, conference_df, final_model, pca_inst, min_max, 'east')
    west_region = [
        # West Region (Florida No. 1)
        ('florida', 'norfolk state', 1, 16),
        ('uconn', 'oklahoma', 8, 9),
        ('memphis', 'colorado state', 5, 12),
        ('maryland', 'grand canyon', 4, 13),
        ('missouri', 'drake', 6, 11),
        ('texas tech', 'unc wilmington', 3, 14),
        ('kansas', 'arkansas', 7, 10),
        ("st john's", 'nebraska-omaha', 2, 15),
    ]
    west_winner = predict_bracket(west_region, team_mapping_df, curr_season, conference_df, final_model, pca_inst, min_max, 'west')

    midwest_region = [
        # Midwest Region (Houston No. 1)
        ('houston', 'southern illinois-edwardsville', 1, 16),
        ('gonzaga', 'georgia', 8, 9),
        ('clemson', 'mcneese', 5, 12),
        ('purdue', 'high point', 4, 13),
        ('illinois', 'texas', 6, 11),
        ('kentucky', 'troy', 3, 14),
        ('ucla', 'utah-state', 7, 10),
        ('tennessee', 'wofford', 2, 15)
    ]
    midwest_winner = predict_bracket(midwest_region, team_mapping_df, curr_season, conference_df, final_model, pca_inst, min_max, 'midwest')

    #final_four
    final_four_matchup= [
        (south_winner[0],west_winner[0],south_winner[1],west_winner[1]),
        (east_winner[0],midwest_winner[0],east_winner[1],midwest_winner[1])
    ]
    overall_winner = predict_bracket(final_four_matchup, team_mapping_df, curr_season, conference_df, final_model, pca_inst, min_max, 'Final_four')
    print(f'Overall Winner: {overall_winner}')
    # predict_bracket(initial_matchups, team_mapping_df, curr_season, conference_df, seed_df, final_model)
    # team_1_name, team_0_name = 'virginia commonwealth', 'illinois'
    # seed_winner, seed_loser = 11, 6
    # team_1_id = team_mapping_df[team_mapping_df['TeamNameSpelling'] == team_1_name]['TeamID'].values[0]
    # team_0_id = team_mapping_df[team_mapping_df['TeamNameSpelling'] == team_0_name]['TeamID'].values[0]

    # #team_1
    # sub_df = curr_season[
    #     ((curr_season['WTeamID'] == team_1_id) | (curr_season['LTeamID'] == team_1_id))
    # ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
    # team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
    # team_var = sub_df.var().add_suffix('_var').to_frame().T
    # team_stats_final_winner = pd.concat([team_avg, team_var],axis=1)

    # #conference
    # conference_winner = conference_df[(conference_df['Season'] == 2025) &
    #             (conference_df['TeamID'] == team_1_id)]['ConfID'].values[0]

    # #seed
    # # seed_winner = seed_df[(seed_df['Season'] == 2025) &
    # #             (seed_df['TeamID'] == team_1_id)]['seedID'].values[0]
    
    # #combine df
    # team_stats_final_winner['conference'] = conference_winner
    # team_stats_final_winner['seed'] = seed_winner
    # team_stats_final_winner = team_stats_final_winner.add_suffix('_team_1')

    # #loser
    # #stats
    # sub_df = curr_season[
    #     ((curr_season['WTeamID'] == team_0_id) | (curr_season['LTeamID'] == team_0_id))
    # ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
    # team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
    # team_var = sub_df.var().add_suffix('_var').to_frame().T
    # team_stats_final_loser = pd.concat([team_avg, team_var],axis=1)

    # #conference
    # conference_loser = conference_df[(conference_df['Season'] == 2025) &
    #             (conference_df['TeamID'] == team_0_id)]['ConfID'].values[0]

    # #seed
    # # seed_loser = seed_df[(seed_df['Season'] == 2025) &
    # #             (seed_df['TeamID'] == curr_l_team_ID)]['seedID'].values[0]
    
    # #combine df
    # team_stats_final_loser['conference'] = conference_loser
    # team_stats_final_loser['seed'] = seed_loser
    # team_stats_final_loser = team_stats_final_loser.add_suffix('_team_0')

    # curr_matchup = pd.concat([team_stats_final_winner,team_stats_final_loser],axis=1)

    # team_1_pred = final_model.predict(curr_matchup)

    # RED = "\033[91m"
    # GREEN = "\033[92m"
    # RESET = "\033[0m"
    # color = GREEN if team_1_pred[0] > 0.5 else RED

    # print(f'{color}{team_1_name} vs. {team_0_name} straight prediction: {team_1_pred[0]}{RESET}')
    # print(f'{color}{team_1_name} vs. {team_0_name} prediction divided by max: { team_1_pred[0] / max(create_regression_labels(utils.swap_team_features(final_dataset)))}{RESET}')
if __name__ == "__main__":
    main()