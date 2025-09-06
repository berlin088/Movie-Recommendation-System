import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def create_synthetic_ratings():
    return pd.DataFrame({
        'userId': [1,1,1,2,2,3,3,4,4,5],
        'itemId': [10,20,30,10,30,20,40,20,30,10],
        'rating': [4,5,2,5,3,4,1,2,5,4],
        'timestamp': [0]*10
    })

def preprocess_ratings(ratings, min_user_ratings=1, min_item_ratings=1, verbose=True):
    df = ratings.copy()
    user_counts = df['userId'].value_counts()
    users_keep = user_counts[user_counts >= min_user_ratings].index
    df = df[df['userId'].isin(users_keep)]

    item_counts = df['itemId'].value_counts()
    items_keep = item_counts[item_counts >= min_item_ratings].index
    df = df[df['itemId'].isin(items_keep)]

    unique_users = df['userId'].unique()
    unique_items = df['itemId'].unique()
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {i: j for j, i in enumerate(unique_items)}

    df['u_idx'] = df['userId'].map(user2idx)
    df['i_idx'] = df['itemId'].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)
    if verbose:
        print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")

    R = csr_matrix((df['rating'], (df['u_idx'], df['i_idx'])), shape=(n_users, n_items))
    return df.reset_index(drop=True), user2idx, item2idx, R
