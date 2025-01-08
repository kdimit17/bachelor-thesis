import numpy as np
import joblib
import tensorflow as tf
from gensim.models import KeyedVectors

# 1. Load the Word2Vec model
print("Loading Word2Vec embeddings...")
w2v_model_path = "GoogleNews-vectors-negative300.bin"  # Adjust if needed
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
embedding_dim = w2v_model.vector_size
print(f"Word2Vec loaded. Embedding dimension = {embedding_dim}")

# 2. Load the trained multi-label model and label columns
print("Loading multi-label model...")
model = tf.keras.models.load_model("gdpr_multi_label_model.h5")

print("Loading label columns...")
label_columns = joblib.load(
    "gdpr_label_columns.pkl"
)  # e.g., ["Lawfulness", "Erasure", ...]
num_labels = len(label_columns)
print("Label columns:", label_columns)


def sentence_to_embedding(sentence, w2v_model):
    """Convert a sentence into an averaged Word2Vec embedding."""
    words = sentence.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


def predict_gdpr_labels(text, threshold=0.5):
    """
    Given input text:
      1) Convert to embedding
      2) Pass through model
      3) Return label probabilities + predicted labels above 'threshold'
    """
    embedding = sentence_to_embedding(text, w2v_model).reshape(1, -1)
    probs = model.predict(embedding)[0]  # shape: (num_labels,)

    # Which labels are predicted above threshold?
    predicted_indices = np.where(probs >= threshold)[0]
    predicted_labels = [label_columns[i] for i in predicted_indices]

    return probs, predicted_labels


if __name__ == "__main__":
    test_text = (
        "Lawful basis is provided, and data is portable upon request."
    )
    print("\n=== Test Inference ===\n")
    print("Text:", test_text)

    # Run inference
    threshold = 0.5
    probabilities, labels = predict_gdpr_labels(test_text, threshold=threshold)

    # Print each label's probability
    print("\nProbabilities for each label (in %):")
    for label, p in zip(label_columns, probabilities):
        print(f"  {label}: {p*100:.2f}%")

    # Print predicted labels above threshold
    print(f"\nPredicted labels (threshold={threshold}):")
    if labels:
        for lbl in labels:
            print("  -", lbl)
    else:
        print("  None")
