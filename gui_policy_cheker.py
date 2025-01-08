import tkinter as tk
from tkinter import ttk, messagebox
import threading
import requests
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
import joblib
import tensorflow as tf
from gensim.models import KeyedVectors

def fetch_policy_text(url, max_len=2_000_000):
    """
    Fetches the webpage content from the given URL and attempts to extract main text.
    Returns a cleaned string of text.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    body = soup.find("body")
    if not body:
        return ""

    text = body.get_text(separator=" ")
    text = text[:max_len]
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, chunk_size=3):
    """
    Splits the text into sentences, then groups them into chunks of 'chunk_size' sentences.
    Returns a list of chunk strings.
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def sentence_to_embedding(sentence, w2v_model):
    """
    Averages the Word2Vec embeddings for all words in 'sentence'.
    """
    words = sentence.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)


def run_inference(url, results_text_widget, progress_var):
    """
    The main logic: loads the model, processes the URL, and updates the UI with results.
    Runs in a separate thread to avoid blocking the main GUI.
    """
    try:
        # Update progress to show "loading"
        progress_var.set(20)

        # Load model
        model = tf.keras.models.load_model("gdpr_multi_label_model.h5")

        # Load label columns
        label_columns = joblib.load("gdpr_label_columns.pkl")

        # Load Word2Vec
        w2v_model_path = "GoogleNews-vectors-negative300.bin"
        w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

        # Update progress
        progress_var.set(40)

        # Fetch policy text
        policy_text = fetch_policy_text(url)
        if not policy_text or "Error fetching" in policy_text:
            results_text_widget.insert(
                tk.END, f"Failed to extract text from policy.\n\n{policy_text}\n"
            )
            progress_var.set(0)
            return

        chunks = split_into_chunks(policy_text, chunk_size=3)

        # Update progress
        progress_var.set(60)

        found_categories = {}
        threshold = 0.7  # More stringent threshold

        for chunk in chunks:
            if len(chunk.split()) < 20:
                continue
            embedding = sentence_to_embedding(chunk, w2v_model).reshape(1, -1)
            probs = model.predict(embedding)[0]
            pred_indices = np.where(probs >= threshold)[0]

            for idx in pred_indices:
                cat = label_columns[idx]
                # Store the first snippet that triggered this category
                if cat not in found_categories:
                    found_categories[cat] = chunk

        # Calculate coverage
        total_categories = len(label_columns)
        covered_count = len(found_categories)
        coverage_percentage = (covered_count / total_categories) * 100

        # Update progress to near-completion
        progress_var.set(80)

        # Build and display the report
        build_report(results_text_widget, found_categories, label_columns, coverage_percentage)

        # Finish progress
        progress_var.set(100)

    except Exception as e:
        results_text_widget.insert(tk.END, f"An error occurred:\n{e}")
    finally:
        pass  # Could reset progress here or leave at 100


def build_report(results_text_widget, found_categories, label_columns, coverage_percentage):
    """
    Builds and formats the report to be inserted into the text widget.
    This version formats categories with color coding and a clean structure.
    """
    results_text_widget.delete("1.0", tk.END)  # Clear previous results

    # Summary
    results_text_widget.insert(tk.END, "=== GDPR Policy Coverage Report ===\n", "bold")
    results_text_widget.insert(tk.END, f"Total Categories: {len(label_columns)}\n")
    results_text_widget.insert(tk.END, f"Categories Covered: {len(found_categories)}\n")
    results_text_widget.insert(tk.END, f"Coverage: {coverage_percentage:.2f}%\n\n")

    # Detailed category info
    for cat in sorted(found_categories.keys()):
        snippet = found_categories[cat]
        status = "Covered" if found_categories[cat] else "Not Covered"
        color_tag = "covered" if status == "Covered" else "error"

        # Insert category
        results_text_widget.insert(tk.END, f"Category: {cat}\n", color_tag)
        results_text_widget.insert(tk.END, f"Status: {status}\n", color_tag)
        results_text_widget.insert(tk.END, f"Snippet: {snippet}\n\n")


def start_inference(url_entry, results_text_widget, progress_var):
    """
    Called when user clicks 'Check Policy' button.
    Spawns a thread to run inference so the GUI remains responsive.
    """
    url = url_entry.get().strip()
    if not url:
        messagebox.showwarning("No URL", "Please enter a valid URL.")
        return

    # Clear previous results
    results_text_widget.delete("1.0", tk.END)

    # Start background thread
    thread = threading.Thread(
        target=run_inference, args=(url, results_text_widget, progress_var), daemon=True
    )
    thread.start()


def create_gui():
    root = tk.Tk()
    root.title("GDPR Policy Checker")

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(".", foreground="black", background="white")
    style.configure("TLabel", foreground="black", background="white")
    style.configure("TButton", foreground="black", background="white")
    style.configure("TEntry", foreground="black", fieldbackground="white")

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

    url_label = ttk.Label(
        main_frame, text="Enter Privacy Policy URL:", foreground="black"
    )
    url_label.grid(row=0, column=0, sticky=tk.W, pady=5)

    url_entry = ttk.Entry(main_frame, width=60)
    url_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

    results_label = ttk.Label(main_frame, text="Results:", foreground="black")
    results_label.grid(row=2, column=0, sticky=tk.W)

    results_text = tk.Text(
        main_frame, width=80, height=20, wrap="word", bg="white", fg="black"
    )
    results_text.grid(row=3, column=0, sticky=(tk.W, tk.E))

    # Progress Bar
    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(
        main_frame,
        orient=tk.HORIZONTAL,
        length=300,
        mode="determinate",
        variable=progress_var,
    )
    progress_bar.grid(row=4, column=0, pady=10)

    # Button
    check_button = ttk.Button(
        main_frame,
        text="Check Policy",
        command=lambda: start_inference(url_entry, results_text, progress_var),
    )
    check_button.grid(row=5, column=0, pady=5)

    # Make the columns/rows expand with the window
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(3, weight=1)

    # Tag configurations for colored text
    results_text.tag_configure("covered", foreground="green", font=("Arial", 10, "bold"))
    results_text.tag_configure("error", foreground="red", font=("Arial", 10))
    results_text.tag_configure("bold", font=("Arial", 12, "bold"))

    return root


if __name__ == "__main__":
    app = create_gui()
    app.mainloop()
