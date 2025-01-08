import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    classification_report,
    average_precision_score,
)
import tensorflow as tf
from gensim.models import KeyedVectors
import joblib

# -------------------------------
# 1. Load Pretrained Word2Vec
# -------------------------------
print("Loading Word2Vec embeddings...")
w2v_model_path = "GoogleNews-vectors-negative300.bin"  # Update path if needed
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
embedding_dim = w2v_model.vector_size
print(f"Word2Vec loaded (dim={embedding_dim}).")


# -------------------------------
# 2. Load One-Hot Dataset
# -------------------------------
def load_dataset(filepath="the_dataset2.csv"):
    """
    Same dataset used during training:
    Columns: SnippetID,Text,Lawfulness,Erasure,Access,Object,...
    """
    df = pd.read_csv(filepath)
    return df


print("Loading dataset for evaluation...")
data = load_dataset("the_dataset2.csv")
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
Y = data[label_columns].values  # shape: (num_samples, num_labels)
texts = data["Text"].tolist()


# -------------------------------
# 3. Convert Text -> Embeddings
# -------------------------------
def sentence_to_embedding(sentence, w2v_model):
    words = sentence.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


print("Generating embeddings for the full dataset...")
X = np.array([sentence_to_embedding(text, w2v_model) for text in tqdm(texts)])

# -------------------------------
# 4. Recreate Train/Test Split
# -------------------------------
# IMPORTANT: Must use the same random_state=42 and test_size=0.2 as in training
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

# -------------------------------
# 5. Load the Model & Label Info
# -------------------------------
print("Loading trained multi-label model...")
model = tf.keras.models.load_model("gdpr_multi_label_model.h5")

# If you saved label columns, you can load them, but here we already have them:
# label_columns = joblib.load("gdpr_label_columns.pkl")

# -------------------------------
# 6. Inference on Test Set
# -------------------------------
print("Running inference on test set...")
Y_pred_probs = model.predict(X_test)  # shape: (num_samples, num_labels)
Y_pred = (Y_pred_probs >= 0.5).astype(int)

# -------------------------------
# 7. Compute Evaluation Metrics
# -------------------------------


### (A) Subset Accuracy
def subset_accuracy(y_true, y_pred):
    return np.mean(
        [1 if np.array_equal(y_pred[i], y_true[i]) else 0 for i in range(len(y_true))]
    )


subset_acc = subset_accuracy(Y_test, Y_pred)
print(f"\nSubset Accuracy (exact match): {subset_acc:.2f}")

### (B) Hamming Loss
# Average fraction of wrong labels (0/1) across all classes and samples
hl = hamming_loss(Y_test, Y_pred)
print(f"Hamming Loss: {hl:.4f}")

### (C) Precision, Recall, F1 per label & Macro Averages

report = classification_report(
    Y_test, Y_pred, target_names=label_columns, zero_division=0
)
print("\nClassification Report (per label, includes macro avg):")
print(report)

### (D) Mean Average Precision (mAP)
# average_precision_score in sklearn can compute AP per label, then we average.
# 'average=None' -> returns array of AP per label. Then we take the mean.
label_aps = []
for i in range(Y_test.shape[1]):
    ap = average_precision_score(Y_test[:, i], Y_pred_probs[:, i])
    label_aps.append(ap)

mAP = np.mean(label_aps)
print("Mean Average Precision (mAP): {:.4f}".format(mAP))

# Print label-wise AP if desired
for label, ap in zip(label_columns, label_aps):
    print(f"  AP for {label}: {ap:.4f}")

# -------------------------------
# 8. Print Sample Predictions
# -------------------------------
print("\nSample Predictions:")
for i in range(3):  # just show 3 examples
    true_labels = [label_columns[j] for j in np.where(Y_test[i] == 1)[0]]
    pred_labels = [label_columns[j] for j in np.where(Y_pred[i] == 1)[0]]
    print(f"Text: {data.iloc[i]['Text']}")
    print(f" True: {true_labels}")
    print(f" Pred: {pred_labels}\n")
