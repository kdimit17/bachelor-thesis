import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
from gensim.models import KeyedVectors
import joblib

# -------------------------------
# 1. Load the Pretrained Word2Vec Model
# -------------------------------
print("Loading Word2Vec embeddings...")
w2v_model_path = "GoogleNews-vectors-negative300.bin"  # Adjust path as needed
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
embedding_dim = w2v_model.vector_size
print(f"Word2Vec loaded. Embedding dimension = {embedding_dim}")


# -------------------------------
# 2. Load Your One-Hot Dataset
# -------------------------------
def load_one_hot_dataset(filepath="the_dataset2.csv"):
    """
    Expects a CSV with columns like:
    SnippetID,Text,Lawfulness,Erasure,Access,Object,Rectification,Portability,SecurityBreach,Transparency
    """
    df = pd.read_csv(filepath)
    return df


print("Loading dataset with one-hot labels...")
data = load_one_hot_dataset("the_dataset2.csv")
print(f"Loaded {len(data)} rows.")

# Define the label columns explicitly
label_columns = [
    "Lawfulness",
    "Erasure",
    "Access",
    "Object",
    "Rectification",
    "Portability",
    "SecurityBreach",
    "Transparency",
]

# -------------------------------
# 3. Extract Text and Labels
# -------------------------------
texts = data["Text"].tolist()
Y = data[label_columns].values  # shape: (num_samples, 8)


# -------------------------------
# 4. Text -> Embeddings
# -------------------------------
def sentence_to_embedding(sentence, w2v_model):
    words = sentence.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


print("Generating embeddings for text...")
X = np.array([sentence_to_embedding(text, w2v_model) for text in tqdm(texts)])

# -------------------------------
# 5. Train / Validation / Test Split
# -------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --------------------------------------
# 6. Build a Multi-Label MLP Classifier
# --------------------------------------
model = models.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(embedding_dim,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.3))
# Output layer size = number of categories, activation = 'sigmoid' for multi-label
model.add(layers.Dense(Y.shape[1], activation="sigmoid"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",  # multi-label => binary crossentropy
    metrics=["accuracy"],
)

print("Training multi-label classifier...")
history = model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=200,  # Adjust as needed
    batch_size=32,
    verbose=1,
)

# ----------------------------------
# 7. Evaluate the Model
# ----------------------------------
print("Evaluating on test set...")
Y_pred_probs = model.predict(X_test)  # shape: (num_samples, num_categories)
Y_pred = (Y_pred_probs >= 0.5).astype(int)

# Calculate a simple subset accuracy: how many predictions match exactly
subset_accuracy = np.mean(
    [1 if np.array_equal(Y_pred[i], Y_test[i]) else 0 for i in range(len(Y_test))]
)

print(f"Subset Accuracy (exact match of all labels): {subset_accuracy:.2f}")


# Calculate the accuracy of having more or equal than 50% of labels correct
def calculate_majority_accuracy(true_labels, predicted_labels):
    correct_counts = [
        sum(1 for t, p in zip(true, pred) if t == p)
        for true, pred in zip(true_labels, predicted_labels)
    ]
    majority_correct = sum(
        1 for count in correct_counts if count >= len(true_labels[0]) / 2
    )
    return majority_correct / len(true_labels)


majority_accuracy = calculate_majority_accuracy(Y_test, Y_pred)
print(f"Majority Accuracy (>= 50% labels correct): {majority_accuracy:.2f}")

# Print a few example predictions
for i in range(min(5, len(X_test))):
    true_labels = [label_columns[idx] for idx, val in enumerate(Y_test[i]) if val == 1]
    pred_labels = [label_columns[idx] for idx, val in enumerate(Y_pred[i]) if val == 1]

    print(f"\n=== Sample {i} ===")
    print(f"Text: {texts[i]}")
    print(f"True: {true_labels}")
    print(f"Pred: {pred_labels}")

# ----------------------------------
# 8. Save the Model & Label Info
# ----------------------------------
model.save("gdpr_multi_label_model.h5")
print("Model saved as gdpr_multi_label_model.h5")

joblib.dump(label_columns, "gdpr_label_columns.pkl")
print("Label columns saved as gdpr_label_columns.pkl")
