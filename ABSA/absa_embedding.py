import json
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error
)
import math



# ---------- Files ----------
TRAIN_CSV = "steam_reviews_labeled_OneByOne_700.csv"
RAW_CSV = "steam_reviews_raw_withTime5000.csv"
OUTPUT_CSV = "aspect_sentiment_results_OneByOne_700_multiaspect.csv"

# ---------- Load embedding model ----------
model_name = "Qwen/Qwen3-Embedding-0.6B"
embed_model = SentenceTransformer(model_name)


# ---------- Semantic embedding function ----------
def embed_sentences(sentences: List[str], batch_size: int = 16) -> np.ndarray:
    total = len(sentences)
    results = []
    for i in range(0, total, batch_size):
        batch = sentences[i:i+batch_size]
        results.extend(embed_model.encode(batch, batch_size=batch_size, normalize_embeddings=True))
    return np.array(results)


# ---------- Build Training Dataset ----------
def build_training_dataset(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract snippet / aspect / sentiment directly from manually labeled training data.
    """
    examples = []
    total = len(train_df)
    checkpoint = max(1, total // 20)

    for i, row in enumerate(train_df.itertuples()):
        if (i + 1) % checkpoint == 0 or (i + 1) == total:
            print(f"[Build Train Dataset] Progress: {100*(i+1)/total:.0f}% ({i+1}/{total})")

        try:
            aspects_json = json.loads(row.aspects_json)
        except Exception:
            aspects_json = []

        for a in aspects_json:
            snippet = a.get("snippet", "").strip()
            if snippet:
                examples.append({
                    "clause": snippet,
                    "aspect": a["aspect"],
                    "sentiment": a["sentiment"]
                })

    return pd.DataFrame(examples)


# ---------- Clause Splitting Function (Reserved/Backup) ----------
def split_into_clauses(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.strip().split()
    if len(words) < 3:
        return [text.strip()]

    s = text.replace("\n", ". ").strip()
    parts = re.split(r'(?<=[.!?。！？])\s+|[;；—\-]+', s)

    clauses = []
    for p in parts:
        p = p.strip()
        if not p or re.fullmatch(r'^[^\w]+$', p):
            continue
        if len(p.split()) < 3:
            clauses.append(p)
            continue
        subparts = [c.strip() for c in re.split(r'[，,]+', p) if c.strip()]
        for sub in subparts:
            if not sub or re.fullmatch(r'^[^\w]+$', sub):
                continue
            clauses.append(sub)
    return clauses


# ---------- Logistic Classification Model ----------
class LogisticModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)


# ---------- Train Model ----------
def train_model(df_clauses: pd.DataFrame, num_epochs=80, lr=5e-4):
    X_texts = df_clauses["clause"].tolist()
    y_aspect = df_clauses["aspect"].tolist()
    y_sent = df_clauses["sentiment"].tolist()

    # 1. Sentence Embedding
    X = embed_sentences(X_texts)

    # 2. Label Encoding
    le_aspect = LabelEncoder()
    le_sent = LabelEncoder()
    Y_aspect = le_aspect.fit_transform(y_aspect)
    Y_sent = le_sent.fit_transform(y_sent)

    # 3. Split Train / Test Set
    X_train, X_test, ya_train, ya_test, ys_train, ys_test = train_test_split(
        X, Y_aspect, Y_sent, test_size=0.3, random_state=42
    )

    # 4. Standardization
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Convert to torch Tensor
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    ya_train_t = torch.tensor(ya_train, dtype=torch.long)
    ys_train_t = torch.tensor(ys_train, dtype=torch.long)
    ya_test_t = torch.tensor(ya_test, dtype=torch.long)
    ys_test_t = torch.tensor(ys_test, dtype=torch.long)

    # 6. Define Model and Optimizer
    model_aspect = LogisticModel(X_train_t.shape[1], len(le_aspect.classes_))
    model_sent = LogisticModel(X_train_t.shape[1], len(le_sent.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer_aspect = optim.Adam(model_aspect.parameters(), lr=lr)
    optimizer_sent = optim.Adam(model_sent.parameters(), lr=lr)

    # 7. Training Process
    train_losses_aspect, test_losses_aspect = [], []
    train_losses_sent, test_losses_sent = [], []

    for epoch in range(num_epochs):
        model_aspect.train()
        model_sent.train()

        out_a = model_aspect(X_train_t)
        out_s = model_sent(X_train_t)

        loss_a = criterion(out_a, ya_train_t)
        loss_s = criterion(out_s, ys_train_t)
        total_loss = loss_a + loss_s

        optimizer_aspect.zero_grad()
        optimizer_sent.zero_grad()
        total_loss.backward()
        optimizer_aspect.step()
        optimizer_sent.step()

        # Test Set Loss
        model_aspect.eval()
        model_sent.eval()
        with torch.no_grad():
            test_loss_a = criterion(model_aspect(X_test_t), ya_test_t)
            test_loss_s = criterion(model_sent(X_test_t), ys_test_t)

        train_losses_aspect.append(loss_a.item())
        test_losses_aspect.append(test_loss_a.item())
        train_losses_sent.append(loss_s.item())
        test_losses_sent.append(test_loss_s.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:03d}] "
                  f"Aspect Loss: {loss_a.item():.4f}/{test_loss_a.item():.4f} | "
                  f"Sent Loss: {loss_s.item():.4f}/{test_loss_s.item():.4f}")

    # 8. Loss Curve (Save)
    plt.figure(figsize=(8,4))
    plt.plot(train_losses_aspect, label='Train Aspect Loss')
    plt.plot(test_losses_aspect, label='Test Aspect Loss')
    plt.plot(train_losses_sent, label='Train Sent Loss', linestyle='--')
    plt.plot(test_losses_sent, label='Test Sent Loss', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.title("Training / Test Loss Curves")
    plt.tight_layout()
    plt.savefig("training_loss_curve.png")
    plt.close()

    # 9. Train / Test Accuracy
    with torch.no_grad():
        pred_a_train = model_aspect(X_train_t).argmax(dim=1)
        pred_a_test = model_aspect(X_test_t).argmax(dim=1)
        pred_s_train = model_sent(X_train_t).argmax(dim=1)
        pred_s_test = model_sent(X_test_t).argmax(dim=1)

        acc_a_train = (pred_a_train == ya_train_t).float().mean().item()
        acc_a_test = (pred_a_test == ya_test_t).float().mean().item()
        acc_s_train = (pred_s_train == ys_train_t).float().mean().item()
        acc_s_test = (pred_s_test == ys_test_t).float().mean().item()

    print(f"Aspect Train/Test: {acc_a_train:.3f} / {acc_a_test:.3f}")
    print(f"Sentiment Train/Test: {acc_s_train:.3f} / {acc_s_test:.3f}")
    print(f"Generalization Gap (Aspect): {acc_a_train - acc_a_test:.3f}")
    print(f"Generalization Gap (Sentiment): {acc_s_train - acc_s_test:.3f}")

    # --- Additional Evaluation Metrics ---
    # Convert tensors to numpy arrays
    ya_test_np = ya_test_t.numpy()
    ys_test_np = ys_test_t.numpy()
    pred_a_test_np = pred_a_test.numpy()
    pred_s_test_np = pred_s_test.numpy()

    # --- Accuracy ---
    acc_aspect = accuracy_score(ya_test_np, pred_a_test_np)
    acc_sent = accuracy_score(ys_test_np, pred_s_test_np)

    # --- F1, Precision, Recall (macro-averaged across classes) ---
    f1_aspect = f1_score(ya_test_np, pred_a_test_np, average='macro')
    f1_sent = f1_score(ys_test_np, pred_s_test_np, average='macro')
    prec_aspect = precision_score(ya_test_np, pred_a_test_np, average='macro')
    prec_sent = precision_score(ys_test_np, pred_s_test_np, average='macro')
    rec_aspect = recall_score(ya_test_np, pred_a_test_np, average='macro')
    rec_sent = recall_score(ys_test_np, pred_s_test_np, average='macro')

    # --- RMSE ---
    rmse_aspect = math.sqrt(mean_squared_error(ya_test_np, pred_a_test_np))
    rmse_sent = math.sqrt(mean_squared_error(ys_test_np, pred_s_test_np))

    # --- AUC (multi-class one-vs-rest macro average) ---
    try:
        prob_a_test = torch.softmax(model_aspect(X_test_t), dim=1).numpy()
        prob_s_test = torch.softmax(model_sent(X_test_t), dim=1).numpy()
        auc_aspect = roc_auc_score(ya_test_np, prob_a_test, multi_class='ovr', average='macro')
        auc_sent = roc_auc_score(ys_test_np, prob_s_test, multi_class='ovr', average='macro')
    except Exception:
        auc_aspect, auc_sent = float('nan'), float('nan')  # handle if any class missing

    print("\n=== Evaluation Metrics ===")
    print(f"[Aspect]  Acc={acc_aspect:.3f}, F1={f1_aspect:.3f}, "
          f"Prec={prec_aspect:.3f}, Rec={rec_aspect:.3f}, "
          f"RMSE={rmse_aspect:.3f}, AUC={auc_aspect:.3f}")
    print(f"[Sentiment]  Acc={acc_sent:.3f}, F1={f1_sent:.3f}, "
          f"Prec={prec_sent:.3f}, Rec={rec_sent:.3f}, "
          f"RMSE={rmse_sent:.3f}, AUC={auc_sent:.3f}")


    # 10. Save Model (Temporary, without prototypes)
    torch.save({
        "scaler": scaler,
        "model_aspect": model_aspect.state_dict(),
        "model_sent": model_sent.state_dict(),
        "le_aspect": le_aspect,
        "le_sent": le_sent,
        "embed_model_name": model_name
    }, "absa_torch_model.pth")

    print("✅ Model saved to absa_torch_model.pth (weights and encoders)")
    return scaler, model_aspect, model_sent, le_aspect, le_sent


# ---------- Build Aspect Prototypes (and return snippet embeddings and texts for each aspect) ----------
def build_aspect_prototypes(df_train: pd.DataFrame, embed_model, scaler):
    """
    Returns:
      - aspect_prototypes: dict[aspect] = prototype_vector (numpy 1-D)
      - aspect_snippets: dict[aspect] = [snippet_texts]
      - aspect_snippet_embeddings: dict[aspect] = numpy array shape (n_snippets, dim)
    """
    aspect_texts = {}
    for row in df_train.itertuples():
        a = row.aspect
        if a not in aspect_texts:
            aspect_texts[a] = []
        aspect_texts[a].append(row.clause)

    aspect_prototypes = {}
    aspect_snippets = {}
    aspect_snippet_embeddings = {}

    for a, texts in aspect_texts.items():
        # Compute embedding for each snippet (normalized)
        emb = embed_model.encode(texts, normalize_embeddings=True)
        # Standardization
        emb_scaled = scaler.transform(emb)
        # Prototype is the mean (in standardized space)
        prototype = np.mean(emb_scaled, axis=0)
        aspect_prototypes[a] = prototype
        aspect_snippets[a] = texts
        aspect_snippet_embeddings[a] = emb_scaled  # (n_snip, dim)

    print(f"✅ Built {len(aspect_prototypes)} aspect prototypes")
    print(aspect_prototypes)
    return aspect_prototypes, aspect_snippets, aspect_snippet_embeddings


def predict_review_multilabel_from_raw(
    text: str,
    scaler,
    clf_sent,
    le_sent,
    embed_model,
    aspect_prototypes: dict,
    sim_threshold: float = 0.22,
    top_k_snippet: int = 1
):
    """
    Perform multi-aspect detection on a single raw review.
    - Snippet candidates come from splitting the review into clauses (split_into_clauses)
    - For each aspect, compute cosine similarity between prototype and each clause's embedding, 
      select the best clause as the snippet for that aspect
    - Sentiment is predicted using the selected clause's embedding (if no clause, use the entire review)
    Returns: list of dict {aspect, sentiment, snippet, score, snippet_score}
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1) Split review into candidate snippets (sentences/clauses)
    clauses = split_into_clauses(text)
    if len(clauses) == 0:
        clauses = [text.strip()]

    # 2) Compute embeddings for these clauses (using same embed_model + scaler as training)
    #    Note: embed_model.encode(..., normalize_embeddings=True) returns L2-normalized raw embedding
    clause_emb = embed_model.encode(clauses, normalize_embeddings=True)  # shape (n_clauses, dim)
    clause_emb_scaled = scaler.transform(clause_emb)  # shape (n_clauses, dim)

    # Precompute norms
    clause_norms = np.linalg.norm(clause_emb_scaled, axis=1) + 1e-9  # (n_clauses,)

    results = []
    # 3) Match each aspect prototype
    for aspect, proto in aspect_prototypes.items():
        # proto: 1-D numpy vector in same scaled space (since build_aspect_prototypes uses emb_scaled)
        proto_norm = np.linalg.norm(proto) + 1e-9

        # Cosine similarity: each clause with prototype
        sims = (clause_emb_scaled @ proto) / (clause_norms * proto_norm + 1e-12)  # shape (n_clauses,)
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])

        if top_sim < sim_threshold:
            # Below threshold -> review does not contain this aspect
            continue

        chosen_snippet = clauses[top_idx]
        snippet_score = top_sim  # Similarity with prototype (i.e., cosine between chosen clause and prototype)

        # 4) Sentiment prediction: use selected snippet's embedding (vector after scaler)
        snippet_emb_vec = clause_emb_scaled[top_idx:top_idx+1]  # shape (1, dim)
        snippet_emb_tensor = torch.tensor(snippet_emb_vec, dtype=torch.float32)
        with torch.no_grad():
            sent_logits = clf_sent(snippet_emb_tensor)
            sent_id = sent_logits.argmax(dim=1).item()
            sentiment = le_sent.inverse_transform([sent_id])[0]

        results.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "snippet": chosen_snippet,
            "score": float(top_sim),
            "snippet_score": float(snippet_score)
        })

    # Sort (optional)
    #results.sort(key=lambda x: x["score"], reverse=True)
    return results



# ---------- Main Workflow ----------
if __name__ == "__main__":
    # 1) Read training data and build training dataset
    train_df = pd.read_csv(TRAIN_CSV)
    df_train_clauses = build_training_dataset(train_df)
    print(f"Training dataset sample count: {len(df_train_clauses)}")

    # 2) Train model
    scaler, clf_aspect, clf_sent, le_aspect, le_sent = train_model(df_train_clauses)

    # 3) Build aspect prototypes based on training samples (and save to model file)
    aspect_prototypes, aspect_snippets, aspect_snippet_embeddings = build_aspect_prototypes(
        df_train_clauses, embed_model, scaler
    )

    # Save prototypes and other info together with model (overwrite previous file)
    model_dict = {
        "scaler": scaler,
        "model_aspect": clf_aspect.state_dict(),
        "model_sent": clf_sent.state_dict(),
        "le_aspect": le_aspect,
        "le_sent": le_sent,
        "embed_model_name": model_name,
        "aspect_prototypes": aspect_prototypes,
        "aspect_snippets": aspect_snippets,
        "aspect_snippet_embeddings": aspect_snippet_embeddings
    }
    torch.save(model_dict, "absa_torch_model_with_prototypes.pth")
    print("✅ Full model + prototypes saved to absa_torch_model_with_prototypes.pth")

    # ---------- Replace inference and saving in main workflow (only output raw data) ----------
    if __name__ == "__main__":
        # ... Previous training + prototype building code remains unchanged, ensure you have obtained:
        # scaler, clf_aspect, clf_sent, le_aspect, le_sent
        # aspect_prototypes, aspect_snippets, aspect_snippet_embeddings  (last two are not used for outputting training snippets)

        # Below is the replaced "Part 4": Predict each row in raw CSV (and save results in one row/one review)
        raw_df = pd.read_csv(RAW_CSV)
        output_rows = []
        total = len(raw_df)
        for i, row in enumerate(raw_df.itertuples()):
            text = getattr(row, "text", "")
            raw_id = getattr(row, "id", None) or i
            date_time = getattr(row, "date_time", "")  # 👈 Get timestamp

            preds = predict_review_multilabel_from_raw(
                text=text,
                scaler=scaler,
                clf_sent=clf_sent,
                le_sent=le_sent,
                embed_model=embed_model,
                aspect_prototypes=aspect_prototypes,
                sim_threshold=0.22,
                top_k_snippet=1
            )

            output_rows.append({
                "raw_id": raw_id,
                "text": text,
                "date_time": date_time,  # 👈 Add this
                "predicted_aspects": json.dumps(preds, ensure_ascii=False)
            })

            # Progress indicator
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"[Predict] {i + 1}/{total} reviews processed")

        df_out = pd.DataFrame(output_rows)

        # When saving, keep column order consistent with input (here keep raw_id, text, predicted_aspects)
        df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ Prediction results saved to {OUTPUT_CSV}")
