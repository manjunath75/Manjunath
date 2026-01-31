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
if st.sidebar.button("🚀 Recommend Courses") and selected_courses:
    new_user_id, new_ratings_df = backend.add_new_ratings(selected_courses)

    # Merge temp ratings (NO disk write)
    base_ratings = backend.load_ratings()
    combined = pd.concat([base_ratings, new_ratings_df], ignore_index=True)

    # Monkey-patch for prediction only
    backend.load_ratings = lambda: combined

    res_df = backend.predict(model_selection, [new_user_id], params)

    if res_df.empty:
        st.warning("No recommendations found.")
    else:
        res_df = res_df.merge(
            courses_df,
            on="COURSE_ID",
            how="left"
        )[["TITLE", "DESCRIPTION", "SCORE"]]

        st.subheader("🎯 Recommended Courses")

        st.dataframe(
            res_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "TITLE": st.column_config.TextColumn("Course Title", width="medium"),
                "DESCRIPTION": st.column_config.TextColumn("Description", width="large"),
                "SCORE": st.column_config.NumberColumn("Score", format="%.3f")
            }
        )
