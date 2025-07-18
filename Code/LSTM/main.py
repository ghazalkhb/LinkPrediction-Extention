import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import random

# Function to generate negative samples
def generate_negative_samples(df, num_neg_samples, interacted_pairs):

    negative_samples = []
    nodes = df['um_encoded'].unique()
    attempts = 0

    # Generate negative samples until the desired number is reached or maximum attempts are exceeded
    while len(negative_samples) < num_neg_samples and attempts < num_neg_samples * 10:
        um = random.choice(nodes)
        dm = random.choice(nodes)
        if um != dm and (um, dm) not in interacted_pairs:
            negative_samples.append((um, dm, 0))  # Label as 0 for negative
        attempts += 1
    return negative_samples

# Load the dataset (assume df is your DataFrame)
# Create a set of positive interaction pairs
interacted_pairs = set(zip(df['um_encoded'], df['dm_encoded']))
num_positive_samples = len(df)

# Define the number of negative samples (5 times the number of positive samples)
num_neg_samples = num_positive_samples * 5
negative_samples = generate_negative_samples(df, num_neg_samples, interacted_pairs)

# Combine positive and negative samples into a single DataFrame
combined_samples = pd.DataFrame(df[['um_encoded', 'dm_encoded']].values.tolist(), columns=['um_encoded', 'dm_encoded'])
combined_samples['label'] = 1  # Label positive samples as 1
negative_df = pd.DataFrame(negative_samples, columns=['um_encoded', 'dm_encoded', 'label'])
combined_samples = pd.concat([combined_samples, negative_df])

# Prepare features and labels for training
X_data = combined_samples[['um_encoded', 'dm_encoded']]
y_data = combined_samples['label']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, stratify=y_data, random_state=42
)

# Reshape the data for LSTM input (samples, timesteps, features)
X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model using binary cross-entropy loss and the Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 10 epochs with a batch size of 32
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train).flatten()
y_test_pred = model.predict(X_test).flatten()

# Convert predicted probabilities to binary labels
y_train_pred_binary = np.where(y_train_pred > 0.5, 1, 0)
y_test_pred_binary = np.where(y_test_pred > 0.5, 1, 0)

# Function to compute evaluation metrics
def compute_metrics(y_true, y_pred, y_pred_probs):

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else float('nan')
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, auc, cm

# Compute metrics for both training and testing sets
train_metrics = compute_metrics(y_train, y_train_pred_binary, y_train_pred)
test_metrics = compute_metrics(y_test, y_test_pred_binary, y_test_pred)

# Print training metrics
print("Training Metrics:")
print(f"Accuracy: {train_metrics[0]:.4f}, Precision: {train_metrics[1]:.4f}, Recall: {train_metrics[2]:.4f}, F1 Score: {train_metrics[3]:.4f}, AUC: {train_metrics[4]:.4f}")
print("Confusion Matrix:")
print(train_metrics[5])

# Print testing metrics
print("\nTesting Metrics:")
print(f"Accuracy: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, Recall: {test_metrics[2]:.4f}, F1 Score: {test_metrics[3]:.4f}, AUC: {test_metrics[4]:.4f}")
print("Confusion Matrix:")
print(test_metrics[5])
