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