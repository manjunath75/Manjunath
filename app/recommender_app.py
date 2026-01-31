import streamlit as st
import pandas as pd
import backend

st.set_page_config(page_title="Course Recommender", layout="wide")

# ----------------------------
# HEARTBEAT (prevents spinner)
# ----------------------------
st.title("📚 AI Course Recommender")
st.write("App loaded successfully")

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

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Configuration")

# 1. Model Selection
model_selection = st.sidebar.selectbox(
    "Choose Recommendation Model",
    backend.models
)

# 2. Dynamic Parameters (The Fix)
params = {}

# Check if "Clustering" is selected
if model_selection == backend.models[2]: 
    # Show Cluster Slider ONLY for Clustering
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 10)
    params["n_clusters"] = n_clusters
else:
    # Show Similarity Slider for everything else
    sim_threshold = st.sidebar.slider("Similarity Threshold (%)", 0, 100, 30)
    params["sim_threshold"] = sim_threshold

model_selection = st.sidebar.selectbox(
    "Choose Recommendation Model",
    backend.models
)

sim_threshold = st.sidebar.slider(
    "Similarity Threshold (%)",
    0, 100, 60
)

params = {"sim_threshold": sim_threshold}

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
    # 1. Add this debug line
    st.write(f"DEBUG: Selected Model: {model_selection}")
    
    # 2. Load and merge ratings
    base_ratings = backend.load_ratings()
    new_user_id, combined_ratings = backend.add_new_ratings(base_ratings, selected_courses)
    
    # 3. Monkey-patch
    backend.load_ratings = lambda: combined_ratings

    # 4. Predict
    res_df = backend.predict(model_selection, [new_user_id], params)
    
    # 5. Add this debug line
    st.write(f"DEBUG: Found {len(res_df)} results.")

    if res_df.empty:
        st.warning("No recommendations found. Try lowering the Similarity Threshold.")
    else:
        res_df = backend.predict(model_selection, [new_user_id], params)
