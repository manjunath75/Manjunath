import streamlit as st
import pandas as pd
import backend

st.set_page_config(page_title="Course Recommender", layout="wide")

# ----------------------------
# HEARTBEAT
# ----------------------------
st.title("📚 AI Course Recommender")

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data
def load_courses():
    return backend.load_courses()

courses_df = load_courses()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Configuration")

# 1. Model Selection
model_selection = st.sidebar.selectbox(
    "Choose Recommendation Model",
    backend.models
)

# 2. Dynamic Parameters
params = {}

# Check if "Clustering" is selected (Index 2 in the models list)
if model_selection == backend.models[2]: 
    # Show Cluster Slider ONLY for Clustering
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 10)
    params["n_clusters"] = n_clusters
else:
    # Show Similarity Slider for everything else
    sim_threshold = st.sidebar.slider("Similarity Threshold (%)", 0, 100, 30)
    params["sim_threshold"] = sim_threshold

# ----------------------------
# Course Selection
# ----------------------------
st.subheader("Select Courses You Liked")

selected_courses = st.multiselect(
    "Courses",
    options=courses_df["COURSE_ID"],
    format_func=lambda x: courses_df[courses_df["COURSE_ID"] == x]["TITLE"].values[0]
)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("🚀 Recommend Courses") and selected_courses:
    
    # 1. Load existing ratings
    base_ratings = backend.load_ratings()

    # 2. Add new user ratings (returns ID and MERGED dataframe)
    new_user_id, combined_ratings = backend.add_new_ratings(base_ratings, selected_courses)

    # 3. Monkey-patch backend to use the in-memory combined ratings
    backend.load_ratings = lambda: combined_ratings

    # 4. Predict
    res_df = backend.predict(model_selection, [new_user_id], params)

    if res_df.empty:
        st.warning("No recommendations found. Try lowering the threshold or changing the model.")
