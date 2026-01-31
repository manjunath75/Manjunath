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
# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("🚀 Recommend Courses") and selected_courses:
    # 1. Load the existing ratings first
    base_ratings = backend.load_ratings()

    # 2. Pass BOTH arguments to the function
    # Note: backend.add_new_ratings returns the ID and the ALREADY MERGED dataframe
    new_user_id, combined_ratings = backend.add_new_ratings(base_ratings, selected_courses)

    # 3. Monkey-patch using the combined dataframe returned by the backend
    backend.load_ratings = lambda: combined_ratings

    res_df = backend.predict(model_selection, [new_user_id], params)
