import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu

import profile_based_matching as pbm

st.set_page_config(
    page_title="Intelligent Hybrid Recommendation System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>




/* Main background */
.stApp{
    background: linear-gradient(
        135deg,
        #0F172A,
        #1E293B,
        #111827
    );
}

/* Main title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
    margin-top:20px;
}

/* Subtitle */

.subtitle{
    text-align:center;
    color:#CBD5E1;
    font-size:20px;
    margin-bottom:40px;
}

/* Metric Cards */

.metric-card{

    background:rgba(255,255,255,0.08);

    backdrop-filter: blur(16px);

    border-radius:20px;

    padding:25px;

    text-align:center;

    box-shadow:0 10px 30px rgba(0,0,0,0.35);

    transition:0.3s;
}

.metric-card:hover{

    transform:translateY(-6px);

    box-shadow:0 15px 35px rgba(0,229,255,0.35);
}

/* User Card */

.profile-card{

    background:#1E293B;

    border-radius:18px;

    padding:30px;

    color:white;

    box-shadow:0 8px 20px rgba(0,0,0,0.4);

}

.recommend-card{

    background:#111827;

    border-left:6px solid #00E5FF;

    border-radius:15px;

    padding:20px;

    margin-bottom:20px;

    color:white;

}

div.stButton>button{

    width:100%;

    border-radius:15px;

    height:60px;

    background:#00E5FF;

    color:black;

    font-size:18px;

    font-weight:bold;

}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backend():
    pbm.initialize_model()
    results, new_acceptance_rate = pbm.evaluate_system()
    return results, new_acceptance_rate

results_df, new_acceptance_rate = load_backend()


with st.sidebar:

    selected = option_menu(
        menu_title="Navigation",

        options=[
            "Dashboard",
            "Recommendations",
            "Analytics",
            "About"
        ],

        icons=[
            "speedometer2",
            "people-fill",
            "bar-chart-fill",
            "info-circle-fill"
        ],

        menu_icon="robot",

        default_index=0,

        styles={
            "container":{
                "padding":"8px",
                "background-color":"#111827",
                "border-radius":"15px"
            },

            "icon":{
                "color":"#00E5FF",
                "font-size":"18px"
            },

            "nav-link":{
                "font-size":"16px",
                "text-align":"left",
                "margin":"5px",
                "--hover-color":"#1E293B",
            },

            "nav-link-selected":{
                "background-color":"#00E5FF",
                "color":"black",
            }
        }
    )

if selected == "Dashboard":
    st.markdown("""
    <div class="main-title">
    🤝 Intelligent Hybrid Recommendation System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="subtitle">
    AI Powered Recommendation Engine using NLP, MBTI & Adaptive Learning
    </div>
    """, unsafe_allow_html=True)

    total_users = len(pbm.df)
    total_matches = len(results_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h1>👥</h1>
            <h2>{total_users}</h2>
            <p>Total Users</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h1>🤝</h1>
            <h2>{total_matches}</h2>
            <p>Total Matches</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h1>📈</h1>
            <h2>{pbm.initial_acceptance_rate:.2f}%</h2>
            <p>Before Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h1>🚀</h1>
            <h2>{new_acceptance_rate:.2f}%</h2>
            <p>After Learning</p>
        </div>
        """, unsafe_allow_html=True)    

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("""
    <h2 style='color:white;text-align:center;'>
    👤 Select User Profile
    </h2>
    """, unsafe_allow_html=True)


    user_ids = pbm.df["user_id"].tolist()

    selected_user = st.selectbox(
        "Choose User ID",
        user_ids
    )


    selected_profile = pbm.df[
        pbm.df["user_id"] == selected_user
    ].iloc[0]


    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="profile-card">

    <h2>👤 {selected_profile['name']}</h2>

    <hr>

    <b>🆔 User ID :</b> {selected_profile['user_id']}<br><br>

    <b>💼 Profession :</b> {selected_profile['profession']}<br><br>

    <b>🎓 Education :</b> {selected_profile['education']}<br><br>

    <b>📍 City :</b> {selected_profile['city']}<br><br>

    <b>🎭 MBTI :</b> {selected_profile['mbti']}<br><br>

    <b>🎂 Age :</b> {selected_profile['age']}<br><br>

    <b>⚧ Gender :</b> {selected_profile['gender']}<br><br>

    <b>⭐ Experience :</b> {selected_profile['experience']} Years<br><br>

    <b>🛠 Skills</b><br>

    {selected_profile['skills']}<br><br>

    <b>📄 Professional Summary</b><br>

    {selected_profile['professional_summary']}<br><br>

    <b>🙋 About Me</b><br>

    {selected_profile['about_me']}

    </div>
    """, unsafe_allow_html=True)


    generate = st.button(
        "🚀 Generate AI Recommendations",
        use_container_width=True
    )

    if generate:

        with st.spinner("Finding the best matches using AI..."):

            recommendations = pbm.recommend_users_after_learning(
                selected_user,
                top_n=5
            )

        st.success("Top 5 recommendations generated successfully!")


    if generate:

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <h2 style='color:white;text-align:center;'>
        🎯 Top AI Recommendations
        </h2>
        """, unsafe_allow_html=True)

        for _, row in recommendations.iterrows():

            score = row["Compatibility Score"]

            if score >= 85:
                color = "#22C55E"      # Green
            elif score >= 70:
                color = "#F59E0B"      # Orange
            else:
                color = "#EF4444"      # Red

            st.markdown(f"""
            <div style="
                background:#1E293B;
                padding:20px;
                border-radius:18px;
                margin-bottom:20px;
                border-left:8px solid {color};
                box-shadow:0 8px 20px rgba(0,0,0,0.3);
            ">

            <h3 style="color:white;">
            👤 {row['Name']}
            </h3>

            <b style="color:white;">💼 Profession:</b>
            <span style="color:#CBD5E1;">{row['Profession']}</span><br>

            <b style="color:white;">📍 City:</b>
            <span style="color:#CBD5E1;">{row['City']}</span><br>

            <b style="color:white;">🎭 MBTI:</b>
            <span style="color:#CBD5E1;">{row['MBTI']}</span><br><br>

            <b style="color:white;">Compatibility Score</b>

            <div style="
                background:#334155;
                border-radius:12px;
                overflow:hidden;
                height:18px;
                margin-top:8px;
            ">

            <div style="
                width:{score}%;
                background:{color};
                height:18px;
            "></div>

            </div>

            <br>

            <h2 style="color:{color};">
            ⭐ {score:.2f}%
            </h2>

            </div>

            """, unsafe_allow_html=True)


elif selected == "Recommendations":

    st.title("🎯 AI Recommendations")

    user = st.selectbox(
        "Select User",
        pbm.df["user_id"]
    )

    if st.button("Generate"):

        rec = pbm.recommend_users_after_learning(user)

        st.dataframe(rec)


elif selected == "Analytics":

    st.markdown("""
    <h1 style='text-align:center;color:white;'>
    📊 AI Analytics Dashboard
    </h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)


    with col1:
        st.metric(
            "📈 Initial Acceptance Rate",
            f"{pbm.initial_acceptance_rate:.2f}%"
        )

    with col2:
        st.metric(
            "🚀 Acceptance Rate After Learning",
            f"{new_acceptance_rate:.2f}%",
            delta=f"{new_acceptance_rate - pbm.initial_acceptance_rate:.2f}%"
        )

    st.divider()


    


    # ---------------- Gender ----------------

    with col1:

        gender = pbm.df["gender"].value_counts().reset_index()
        gender.columns = ["Gender", "Count"]

        fig = px.pie(
            gender,
            values="Count",
            names="Gender",
            hole=0.6,
            title="Gender Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- MBTI ----------------

    with col2:

        mbti = pbm.df["mbti"].value_counts().reset_index()
        mbti.columns = ["MBTI", "Count"]

        fig = px.bar(
            mbti,
            x="MBTI",
            y="Count",
            title="MBTI Personality Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    age = px.histogram(
        pbm.df,
        x="age",
        nbins=12,
        title="Age Distribution"
    )

    age.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(age, use_container_width=True)


elif selected == "About":

    st.title("ℹ️ About Project")

    st.markdown("""

### Intelligent Hybrid Recommendation System

This project recommends compatible users using

- NLP
- TF-IDF
- Cosine Similarity
- MBTI Personality Matching
- Location Matching
- Adaptive Learning

### Technologies

- Python

- Streamlit

- Machine Learning

- NLP

- Pandas

- Scikit-learn

- Matplotlib

- Seaborn

""")