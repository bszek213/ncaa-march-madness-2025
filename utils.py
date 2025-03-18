import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def conference_encoder(df,season_start):
    sub_df = df[df['Season'] >= season_start]
    sub_df['ConfID'] = pd.factorize(sub_df['ConfAbbrev'])[0]
    return sub_df[['ConfID','TeamID','Season']]

def seed_encoder(df,season_start):
    sub_df = df[df['Season'] >= season_start]
    # sub_df['seedID'] = pd.factorize(sub_df['Seed'])[0]
    # sub_df['seedID'] = df['Seed'].str[1:]#.astype(int)
    save_digits = []
    for val in sub_df['Seed']:
        save_digits.append(int(''.join(char for char in val if char.isdigit())))
    sub_df['seedID'] = save_digits

    return sub_df[['seedID','TeamID','Season','Seed']]

def swap_team_features(df):
    #team_1 and team_0 columns
    team_1_cols = [col for col in df.columns if col.endswith('_team_1')]
    team_0_cols = [col for col in df.columns if col.endswith('_team_0')]
    
    df_swapped = df.copy()
    
    #randomly decide which rows to swap
    swap_mask = np.random.choice([True, False], size=len(df))
    
    #swap the features for the selected rows
    for col1, col0 in zip(team_1_cols, team_0_cols):
        temp = df_swapped.loc[swap_mask, col1].copy()
        df_swapped.loc[swap_mask, col1] = df_swapped.loc[swap_mask, col0]
        df_swapped.loc[swap_mask, col0] = temp
    
    #swap the labels for the swapped rows
    # df_swapped.loc[swap_mask, 'team_1'] = 0
    # df_swapped.loc[swap_mask, 'team_0'] = 1
    df_swapped.loc[swap_mask, 'team_1'] = df.loc[swap_mask, 'team_0']
    df_swapped.loc[swap_mask, 'team_0'] = df.loc[swap_mask, 'team_1']
    
    #concat original and swapped dataframes
    return pd.concat([df, df_swapped], ignore_index=True)

def add_noise_to_features(features_df, noise_scale=1):
    columns_to_exclude = [col for col in features_df.columns if 'seed' in col or 'conference' in col]
    features_with_noise = features_df.drop(columns=columns_to_exclude)
    features_without_noise = features_df[columns_to_exclude]
    noise = np.random.normal(0, noise_scale, features_with_noise.shape)
    noisy_features = features_with_noise + noise
    noisy_features = pd.concat([noisy_features, features_without_noise], axis=1)
    return noisy_features

def upset_proba(lower_seed, higher_seed):
    """
    The probability is based on historical matchup probabilities
    """
    upset_prob_map = {
        (12, 5): 0.40,
        (11, 6): 0.37,
        (10, 7): 0.39,
        (13, 4): 0.22,
        (14, 3): 0.15,
        (15, 2): 0.07,
        (16, 1): 0.01
    }
    return upset_prob_map.get((lower_seed, higher_seed), 0.0)