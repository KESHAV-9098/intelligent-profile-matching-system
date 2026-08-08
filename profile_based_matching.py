# Importing required libraries

import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

import random


# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


TEXT_WEIGHT = 0.50
MBTI_WEIGHT = 0.30
LOCATION_WEIGHT = 0.20

TEXT_WEIGHT_NEW = None
MBTI_WEIGHT_NEW = None
LOCATION_WEIGHT_NEW = None

df = None
cosine_sim = None

initial_acceptance_rate = None



def clean_text(text):
    text = text.lower()
        
    # Remove punctuation and numbers
    text = ''.join(char for char in text if char.isalpha() or char.isspace())

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# MBTI compatibility dict:
def mbti_match_score(mbti1, mbti2):
    if mbti1 == mbti2:
        return 90

    elif mbti1[0] == mbti2[0]:
        return 75

    else:
        return 60


# Location score function:

def location_score(city1, city2):

    if city1 == city2:
        return 100

    else:
        return 50


# Hybrid recommendation system function:

def recommend_users(user_id, top_n=5):

    # Get selected user's index
    user_index = df[df['user_id'] == user_id].index[0]

    recommendations = []

    for i in range(len(df)):

        # Skip same user
        if i == user_index:
            continue


        # Text Similarity

        text_score = cosine_sim[user_index][i] * 100


        # MBTI Score

        mbti_score = mbti_match_score(
            df.loc[user_index, 'mbti'],
            df.loc[i, 'mbti']
        )


        # Location Score

        location = location_score(
            df.loc[user_index, 'city'],
            df.loc[i, 'city']
        )


        # Final Hybrid Score

        final_score = (
            TEXT_WEIGHT * text_score +
            MBTI_WEIGHT * mbti_score +
            LOCATION_WEIGHT * location
        )

        recommendations.append({

            "User ID": df.loc[i, "user_id"],
            "Name": df.loc[i, "name"],
            "Profession": df.loc[i, "profession"],
            "City": df.loc[i, "city"],
            "MBTI": df.loc[i, "mbti"],

            "Text Similarity (%)": round(text_score, 2),
            "MBTI Score": mbti_score,
            "Location Score": location,
            "Compatibility Score": round(final_score, 2)

        })

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        by="Compatibility Score",
        ascending=False
    )

    return recommendations.head(top_n)


# Hybrid recommendation system function after adaptive learning:

def recommend_users_after_learning(user_id, top_n=5):

    # Get selected user's index
    user_index = df[df['user_id'] == user_id].index[0]

    recommendations = []

    for i in range(len(df)):

        # Skip same user
        if i == user_index:
            continue


        # Text Similarity

        text_score = cosine_sim[user_index][i] * 100


        # MBTI Score

        mbti_score = mbti_match_score(
            df.loc[user_index, 'mbti'],
            df.loc[i, 'mbti']
        )


        # Location Score

        location = location_score(
            df.loc[user_index, 'city'],
            df.loc[i, 'city']
        )


        # Final Hybrid Score

        final_score = (
            TEXT_WEIGHT_NEW * text_score +
            MBTI_WEIGHT_NEW * mbti_score +
            LOCATION_WEIGHT_NEW * location
        )

        recommendations.append({

            "User ID": df.loc[i, "user_id"],
            "Name": df.loc[i, "name"],
            "Profession": df.loc[i, "profession"],
            "City": df.loc[i, "city"],
            "MBTI": df.loc[i, "mbti"],

            "Text Similarity (%)": round(text_score, 2),
            "MBTI Score": mbti_score,
            "Location Score": location,
            "Compatibility Score": round(final_score, 2)

        })

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        by="Compatibility Score",
        ascending=False
    )

    return recommendations.head(top_n)


def initialize_model():

    global df
    global cosine_sim
    global TEXT_WEIGHT_NEW
    global MBTI_WEIGHT_NEW
    global LOCATION_WEIGHT_NEW
    global initial_acceptance_rate

    # Uploading the users.csv dataset

    df = pd.read_csv("users.csv")


    # Dataset shape:
    print("Shape of Dataset:",df.shape)

    # List of column names:
    print("\nColumns:")
    print(df.columns.tolist())

    # Dataset Information:
    print("\nInfo of dataset:")
    df.info()

    # Checking missing values:
    missing_values = df.isnull().sum()
    print("missing_values:")
    print(missing_values)


    # Age distribution:
    plt.figure(figsize=(8,5))
    plt.hist(df['age'], bins=10)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Users")
    plt.grid(True)
    # plt.show()

    # Gender distribution:
    plt.figure(figsize=(6,4))
    df['gender'].value_counts().plot(kind='bar')

    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    # plt.show()

    # Combine Professional Summary and About Me
    df['combined_text'] = (
        df['professional_summary'] + " " +
        df['about_me']
    )

    # Display sample
    df[['user_id', 'combined_text']].head()


    # Clean the combined text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)


    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

    # Cosine Similarity Matrix:

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    plt.figure(figsize=(12,10))

    sns.heatmap(
        cosine_sim,
        cmap="Blues",
        cbar=True
    )

    plt.title("User Similarity Matrix")
    plt.xlabel("Users")
    plt.ylabel("Users")
    # plt.show()


    result = recommend_users("U012")

    plt.figure(figsize=(10,5))

    bars = plt.bar(
        result["Name"],
        result["Compatibility Score"]
    )

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.1f}",
            ha="center"
        )

    plt.title("Top 5 Recommended Users")
    plt.xlabel("Recommended Users")
    plt.ylabel("Compatibility Score")
    plt.ylim(0,100)
    plt.xticks(rotation=20)
    # plt.show()

    # Generating initial recommendation data:

    recommendation_data = []

    user_ids = df["user_id"].tolist()

    for user in user_ids:

        recommendations = recommend_users(user, top_n=20)

        for _, row in recommendations.iterrows():

            recommendation_data.append({
                "user_id": user,
                "matched_user_id": row["User ID"],
                "text_score": row["Text Similarity (%)"],
                "mbti_score": row["MBTI Score"],
                "location_score": row["Location Score"],
                "compatibility_score": row["Compatibility Score"]
            })

    recommendation_df = pd.DataFrame(recommendation_data)

    print(recommendation_df.shape)

    # Creating feedback dataset:

    random.seed(42)

    feedback = []

    for _, row in recommendation_df.iterrows():

        score = row["compatibility_score"]

        if score >= 95:
            action = 1

        elif score >= 80:
            action = 1 if random.random() < 0.80 else 0

        elif score >= 60:
            action = 1 if random.random() < 0.60 else 0

        elif score >= 40:
            action = 1 if random.random() < 0.50 else 0

        else:
            action = 0

        feedback.append(action)

    recommendation_df["action"] = feedback

    recommendation_df.head()

    feedback_df = recommendation_df[
        [
            "user_id",
            "matched_user_id",
            "action"
        ]
    ]

    # Saving feedback dataset to CSV:

    feedback_df.to_csv("feedback.csv", index=False)

    print("feedback.csv created successfully.")

    # Initial acceptance rate:

    initial_acceptance_rate = (
        recommendation_df["action"].mean() * 100
    )

    print(
        f"Initial Acceptance Rate: {initial_acceptance_rate:.2f}%"
    )

    # feedback distribution:

    plt.figure(figsize=(6,4))

    recommendation_df["action"].value_counts().plot(
        kind="bar"
    )

    plt.xticks([0,1],["Reject","Accept"])

    plt.xlabel("Feedback")

    plt.ylabel("Count")

    plt.title("Feedback Distribution")

    # plt.show()

    # Calculating average scores for accepted recommendations:

    accepted = recommendation_df[recommendation_df["action"] == 1]

    avg_text = accepted["text_score"].mean()
    avg_mbti = accepted["mbti_score"].mean()
    avg_location = accepted["location_score"].mean()

    print(avg_text)
    print(avg_mbti)
    print(avg_location)

    # New weights

    total = avg_text + avg_mbti + avg_location

    TEXT_WEIGHT_NEW = avg_text / total
    MBTI_WEIGHT_NEW = avg_mbti / total
    LOCATION_WEIGHT_NEW = avg_location / total

    print(TEXT_WEIGHT_NEW)
    print(MBTI_WEIGHT_NEW)
    print(LOCATION_WEIGHT_NEW)

    return df,cosine_sim


def evaluate_system():

    # Testing the new recommendation function:

    updated_recommendation_data = []

    user_ids = df["user_id"].tolist()

    for user in user_ids:

        recommendations = recommend_users_after_learning(user, top_n=20)

        for _, row in recommendations.iterrows():

            updated_recommendation_data.append({
                "user_id": user,
                "matched_user_id": row["User ID"],
                "text_score": row["Text Similarity (%)"],
                "mbti_score": row["MBTI Score"],
                "location_score": row["Location Score"],
                "compatibility_score": row["Compatibility Score"]
            })

    updated_recommendation_df = pd.DataFrame(updated_recommendation_data)

    print(updated_recommendation_df.shape)


    feedback_after = []

    for _, row in updated_recommendation_df.iterrows():

        score = row["compatibility_score"]

        if score >= 95:
            action = 1

        elif score >= 80:
            action = 1 if random.random() < 0.90 else 0

        elif score >= 60:
            action = 1 if random.random() < 0.70 else 0

        elif score >= 40:
            action = 1 if random.random() < 0.50 else 0

        else:
            action = 0

        feedback_after.append(action)

    updated_recommendation_df["action"] = feedback_after

    new_acceptance_rate = updated_recommendation_df["action"].mean() * 100

    print(f"Initial Acceptance Rate : {initial_acceptance_rate:.2f}%")
    print(f"New Acceptance Rate     : {new_acceptance_rate:.2f}%")

    # Visualizing the acceptance rate before and after learning:

    comparison = pd.DataFrame({
    "Stage": ["Before Learning", "After Learning"],
    "Acceptance Rate": [
        initial_acceptance_rate,
        new_acceptance_rate
    ]
})

    plt.figure(figsize=(6,5))

    bars = plt.bar(
        comparison["Stage"],
        comparison["Acceptance Rate"]
    )

    plt.title("Acceptance Rate Before vs After Adaptive Learning")
    plt.xlabel("Stage")
    plt.ylabel("Acceptance Rate (%)")
    plt.ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    # plt.show()

    updated_recommendation_df.to_csv(
    "final_recommendations.csv",
    index=False
)

    return updated_recommendation_df, new_acceptance_rate

def main():

    initialize_model()

    evaluate_system()               


if __name__ == "__main__":
    main()