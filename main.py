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
    if not os.path.exists('models/regression_model.joblib'):
        df_swapped = utils.swap_team_features(final_dataset)
        #create regression labels
        # #create one hot encoded data
        # y = df_swapped[['team_0', 'team_1']].values
        y = create_regression_labels(df_swapped)
        X = df_swapped.drop(columns=['team_0','team_1'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
        final_model = xgb.XGBRegressor(**best_params,early_stopping_rounds=6)

        final_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
                        verbose=True)
        os.makedirs('models',exist_ok=True)
        joblib.dump(final_model,'models/regression_model.joblib')
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
        plt.savefig('figures/training_curve.png')

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
        plt.savefig('figures/feature_importance.png', bbox_inches='tight')
        plt.close()
    else:
        final_model = joblib.load('models/regression_model.joblib')

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
    team_1_name, team_0_name = 'virginia commonwealth', 'illinois'
    seed_winner, seed_loser = 11, 6
    team_1_id = team_mapping_df[team_mapping_df['TeamNameSpelling'] == team_1_name]['TeamID'].values[0]
    team_0_id = team_mapping_df[team_mapping_df['TeamNameSpelling'] == team_0_name]['TeamID'].values[0]

    #team_1
    sub_df = curr_season[
        ((curr_season['WTeamID'] == team_1_id) | (curr_season['LTeamID'] == team_1_id))
    ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
    team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
    team_var = sub_df.var().add_suffix('_var').to_frame().T
    team_stats_final_winner = pd.concat([team_avg, team_var],axis=1)

    #conference
    conference_winner = conference_df[(conference_df['Season'] == 2025) &
                (conference_df['TeamID'] == team_1_id)]['ConfID'].values[0]

    #seed
    # seed_winner = seed_df[(seed_df['Season'] == 2025) &
    #             (seed_df['TeamID'] == team_1_id)]['seedID'].values[0]
    
    #combine df
    team_stats_final_winner['conference'] = conference_winner
    team_stats_final_winner['seed'] = seed_winner
    team_stats_final_winner = team_stats_final_winner.add_suffix('_team_1')

    #loser
    #stats
    sub_df = curr_season[
        ((curr_season['WTeamID'] == team_0_id) | (curr_season['LTeamID'] == team_0_id))
    ].drop(columns=['WTeamID','LTeamID','WLoc','Season', 'DayNum','NumOT'])
    team_avg = sub_df.mean().add_suffix('_avg').to_frame().T
    team_var = sub_df.var().add_suffix('_var').to_frame().T
    team_stats_final_loser = pd.concat([team_avg, team_var],axis=1)

    #conference
    conference_loser = conference_df[(conference_df['Season'] == 2025) &
                (conference_df['TeamID'] == team_0_id)]['ConfID'].values[0]

    #seed
    # seed_loser = seed_df[(seed_df['Season'] == 2025) &
    #             (seed_df['TeamID'] == curr_l_team_ID)]['seedID'].values[0]
    
    #combine df
    team_stats_final_loser['conference'] = conference_loser
    team_stats_final_loser['seed'] = seed_loser
    team_stats_final_loser = team_stats_final_loser.add_suffix('_team_0')

    curr_matchup = pd.concat([team_stats_final_winner,team_stats_final_loser],axis=1)

    team_1_pred = final_model.predict(curr_matchup)

    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    color = GREEN if team_1_pred[0] > 0.5 else RED

    print(f'{color}{team_1_name} vs. {team_0_name} straight prediction: {team_1_pred[0]}{RESET}')
    print(f'{color}{team_1_name} vs. {team_0_name} prediction divided by max: { team_1_pred[0] / max(create_regression_labels(utils.swap_team_features(final_dataset)))}{RESET}')
if __name__ == "__main__":
    main()