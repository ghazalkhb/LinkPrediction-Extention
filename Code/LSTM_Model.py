import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

def generate_negative_samples(df, num_neg_samples, interacted_pairs):
    negative_samples = []
    nodes = df['um_encoded'].unique()
    attempts = 0
    while len(negative_samples) < num_neg_samples and attempts < num_neg_samples * 10:
        um = random.choice(nodes)
        dm = random.choice(nodes)
        if um != dm and (um, dm) not in interacted_pairs:
            negative_samples.append((um, dm, 0))
        attempts += 1
    return negative_samples

def prepare_lstm_data(df, neg_pos_ratio=5, test_size=0.3, seed=42):
    interacted_pairs = set(zip(df['um_encoded'], df['dm_encoded']))
    num_positive = len(df)
    num_negative = num_positive * neg_pos_ratio
    neg_samples = generate_negative_samples(df, num_negative, interacted_pairs)

    pos_df = df[['um_encoded', 'dm_encoded']].copy()
    pos_df['label'] = 1
    neg_df = pd.DataFrame(neg_samples, columns=['um_encoded', 'dm_encoded', 'label'])
    combined = pd.concat([pos_df, neg_df], ignore_index=True)

    X = combined[['um_encoded', 'dm_encoded']]
    y = combined['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
