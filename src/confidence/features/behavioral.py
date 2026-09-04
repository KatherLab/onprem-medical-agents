import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# --- Singleton-style model loading ---
# Keep the model as a global to ensure it is loaded only once per process.
_embedding_model = None

def _get_model(model_name: str):
    """Helper to lazily load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            print("INFO: Loading sentence-transformer model for the first time...")
            _embedding_model = SentenceTransformer(model_name)
            print("INFO: Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load sentence-transformer model '{model_name}'.")
            print(f"Error: {e}")
            # If loading fails, subsequent calls will return None.
    return _embedding_model

def calculate_consistency_score(texts: List[str], config: Dict) -> float:
    """
    Compute the semantic consistency score for a set of diagnostic texts.

    Args:
        texts (list): List of diagnostic texts from N runs.
        config (dict): Main config used to resolve the model name.

    Returns:
        float: Score in [0, 1], where 1 means fully consistent.
    """
    # model_name = config.get('sentence_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    model_name = config.get('sentence_embedding_model')
    if not model_name:
        print("WARNING: 'sentence_embedding_model' not found in config. Returning 0.0.")
        return 0.0

    model = _get_model(model_name)
    if model is None:
        return 0.0

    cleaned_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    # print('cleaned_texts:', cleaned_texts)

    # Fewer than 2 valid texts means consistency is undefined.
    if len(cleaned_texts) < 2:
        return float("nan") 
    
    try:
        # 1) Encode texts into embeddings. Normalize so dot product equals cosine similarity.
        embeddings = model.encode(cleaned_texts, show_progress_bar=False, normalize_embeddings=True)
        # 2) Compute pairwise cosine similarity matrix.
        # (N, D) x (D, N) -> (N, N)  (D is embedding dimension)
        sim_matrix = cosine_similarity(embeddings)
        # 3) Average the upper triangle (excluding diagonal) to avoid self-comparison
        # and duplicate A-B / B-A comparisons.
        indices = np.triu_indices_from(sim_matrix, k=1)
        
        if len(indices[0]) == 0:
            return float("nan")

        mean_cos = float(np.mean(sim_matrix[indices]))
        
        # Mapping strategy:
        # SBERT similarities are rarely negative. If negative, treat as unrelated (0).
        # Do not shift 0 to 0.5; that would mask errors.
        final_score = max(0.0, mean_cos)
        final_score = min(1.0, final_score) # Clamp to <= 1.0.

        return final_score

    except Exception as e:
        print(f"ERROR: An error occurred during consistency score calculation: {e}")
        return float("nan")


def _normalize_short_text(text: str) -> str:
    """Light normalization for short diagnosis strings before semantic clustering."""
    return " ".join(text.strip().lower().split())


def calculate_semantic_entropy(texts: List[str], config: Dict) -> float:
    """
    Compute a discrete semantic entropy by clustering semantically similar outputs
    with sentence embeddings and measuring the cluster-frequency entropy.

    Entropy direction:
        larger value => more uncertainty
        0.0 => all valid texts fall into a single semantic cluster
    """
    model_name = config.get('sentence_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    if not model_name:
        print("WARNING: 'sentence_embedding_model' not found in config. Returning 0.0.")
        return 0.0

    model = _get_model(model_name)
    if model is None:
        return 0.0

    threshold = float(config.get('semantic_entropy_similarity_threshold_dx', 0.85))
    cleaned_texts = [
        _normalize_short_text(t)
        for t in texts
        if isinstance(t, str) and t.strip()
    ]

    if len(cleaned_texts) < 2:
        return float("nan")

    try:
        embeddings = model.encode(cleaned_texts, show_progress_bar=False, normalize_embeddings=True)

        # Greedy clustering is sufficient here because each case has only a small
        # number of diagnosis samples (for example 5 or 10 runs).
        cluster_members: list[list[int]] = []
        cluster_centroids: list[np.ndarray] = []

        for idx, embedding in enumerate(embeddings):
            assigned_cluster = None
            for cluster_idx, centroid in enumerate(cluster_centroids):
                similarity = float(np.dot(embedding, centroid))
                if similarity >= threshold:
                    assigned_cluster = cluster_idx
                    break

            if assigned_cluster is None:
                cluster_members.append([idx])
                cluster_centroids.append(np.array(embedding, dtype=float))
                continue

            cluster_members[assigned_cluster].append(idx)
            member_embeddings = embeddings[cluster_members[assigned_cluster]]
            updated_centroid = np.mean(member_embeddings, axis=0)
            norm = np.linalg.norm(updated_centroid)
            if norm > 0:
                updated_centroid = updated_centroid / norm
            cluster_centroids[assigned_cluster] = updated_centroid

        cluster_sizes = np.array([len(members) for members in cluster_members], dtype=float)
        probabilities = cluster_sizes / cluster_sizes.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        return max(0.0, entropy)

    except Exception as e:
        print(f"ERROR: An error occurred during semantic entropy calculation: {e}")
        return float("nan")


def _levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute edit distance between two tool-trace token sequences."""
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            substitution_cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    return dp[len1][len2]


def _normalized_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Map edit distance to a [0, 1] similarity score."""
    if not seq1 and not seq2:
        return float("nan")
    if not seq1 or not seq2:
        return 0.0

    distance = _levenshtein_distance(seq1, seq2)
    return max(0.0, 1.0 - (distance / max(len(seq1), len(seq2))))


def calculate_trace_consistency(traces: List[List[str]]) -> float:
    """
    Compute mean pairwise similarity across ordered diagnostic tool traces.
    """
    cleaned_traces = [
        trace for trace in traces
        if isinstance(trace, list) and trace
    ]

    if len(cleaned_traces) < 2:
        return float("nan")

    similarities = []
    for i in range(len(cleaned_traces)):
        for j in range(i + 1, len(cleaned_traces)):
            similarities.append(_normalized_sequence_similarity(cleaned_traces[i], cleaned_traces[j]))

    if not similarities:
        return float("nan")

    return float(np.mean(similarities))
