import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression

# ----------------------------
# Constants
# ----------------------------
models = (
    "Course Similarity",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "KNN",
    "NMF",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Data loaders (CLOUD SAFE)
# ----------------------------
def load_ratings():
    return pd.read_csv(os.path.join(BASE_DIR, "ratings.csv"))

def load_courses():
    df = pd.read_csv(os.path.join(BASE_DIR, "course_processed.csv"))
    if "TITLE" in df.columns:
        df["TITLE"] = df["TITLE"].fillna("").str.title()
    return df

def load_bow():
    return pd.read_csv(os.path.join(BASE_DIR, "courses_bows.csv"))

def load_sim():
    return pd.read_csv(os.path.join(BASE_DIR, "sim.csv"))


def add_new_ratings(ratings_df, new_courses):
    """
    Returns:
    - new_user_id (int)
    - updated_ratings_df (DataFrame)
    """
    if ratings_df.empty:
        new_user_id = 1
    else:
        new_user_id = ratings_df["user"].max() + 1

    new_ratings_df = pd.DataFrame({
        "user": [new_user_id] * len(new_courses),
        "item": new_courses,
        "rating": [3.0] * len(new_courses),
    })

    updated_ratings_df = pd.concat(
        [ratings_df, new_ratings_df],
        ignore_index=True
    )

    return new_user_id, updated_ratings_df


# ----------------------------
# Utilities
# ----------------------------
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(["doc_index", "doc_id"]).max().reset_index()
    idx_id_dict = grouped_df["doc_id"].to_dict()
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    return idx_id_dict, id_idx_dict

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected = all_courses - set(enrolled_ids)

    res = {}
    for e in enrolled_ids:
        for u in unselected:
            if e in id_idx_dict and u in id_idx_dict:
                s = sim_matrix[id_idx_dict[e]][id_idx_dict[u]]
                res[u] = max(res.get(u, 0), s)

    return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

# ----------------------------
# Training
# ----------------------------
def train(model_name, params=None):
    if params is None:
        params = {}

    ratings_df = load_ratings()
    bow_df = load_bow()

    if model_name == models[2]:  # Clustering
        kmeans = KMeans(n_clusters=params.get("n_clusters", 5), random_state=42)
        X = bow_df.drop(columns=["doc_id", "doc_index"])
        kmeans.fit(X)
        return {"model": kmeans}

    if model_name == models[5]:  # NMF
        user_item = ratings_df.pivot_table(index="user", columns="item", values="rating").fillna(0)
        nmf = NMF(n_components=params.get("n_components", 15), random_state=42)
        nmf.fit(user_item)
        return {"model": nmf}

    return None

# ----------------------------
# Prediction
# ----------------------------
def predict(model_name, user_ids, params=None):
    if params is None:
        params = {}

    sim_threshold = params.get("sim_threshold", 60) / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_sim().to_numpy()

    users, courses, scores = [], [], []

    ratings_df = load_ratings()

    for user_id in user_ids:
        if model_name == models[0]:  # Course Similarity
            enrolled = ratings_df[ratings_df["user"] == user_id]["item"].tolist()
            res = course_similarity_recommendations(
                idx_id_dict, id_idx_dict, enrolled, sim_matrix
            )
            for cid, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(cid)
                    scores.append(score)

    if not users:
        return pd.DataFrame(columns=["USER", "COURSE_ID", "SCORE"])

    return pd.DataFrame({
        "USER": users,
        "COURSE_ID": courses,
        "SCORE": scores
    })
