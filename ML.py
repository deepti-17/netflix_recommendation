# from flask import Flask, render_template, request, jsonify
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MultiLabelBinarizer
# import pandas as pd
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Load the cleaned dataset
# df = pd.read_csv('cleaned_movies_10000.csv')

# # Replace NaN with an empty string
# df['genres'] = df['genres'].replace(np.nan, '')

# # Convert genres to a list of integers
# df['genres'] = df['genres'].apply(lambda x: list(map(int, str(x).split(','))) if x else [])

# # Step 2: One-hot encode genres
# mlb = MultiLabelBinarizer()
# genre_matrix = mlb.fit_transform(df['genres'])
# genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# # Step 3: Compute cosine similarity between movies based on genres
# cosine_sim = cosine_similarity(genre_df, genre_df)

# def get_content_based_recommendations_by_name(movie_name, cosine_sim=cosine_sim):
#     # Find the movie by name
#     idx = df[df['title'].str.lower() == movie_name.lower()].index
#     if idx.empty:
#         return "Movie not found in the dataset."

#     idx = idx[0]  # Get the index of the movie
#     similarity_scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores for the movie
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
#     similarity_scores = similarity_scores[1:11]  # Get top 10 similar movies

#     recommended_movie_indices = [x[0] for x in similarity_scores]
#     recommended_movie_titles = df['title'].iloc[recommended_movie_indices].tolist()

#     return recommended_movie_titles  # Returning movie titles


# # Route to display the frontend (index.html)
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to handle movie recommendation requests (via AJAX)
# @app.route('/recommend', methods=['POST'])
# def recommend():
#     movie_name = request.json['movie_name']  # Get movie name from frontend
#     recommended_movies = get_content_based_recommendations_by_name(movie_name)
#     return jsonify({'recommended_movies': recommended_movies})

# if __name__ == '__main__':
#     app.run(debug=True)

# # Example usage:
# movie_name = input("Enter the name of the movie: ")
# recommended_movies = get_content_based_recommendations_by_name(movie_name)
# print("Recommended Movies based on Content-Based Filtering:", recommended_movies)

















from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import requests

# Initialize Flask app
app = Flask(__name__)

# Load the cleaned dataset
df = pd.read_csv('cleaned_movies_10000.csv')

# Replace NaN with an empty string
df['genres'] = df['genres'].replace(np.nan, '')

# Convert genres to a list of integers
df['genres'] = df['genres'].apply(lambda x: list(map(int, str(x).split(','))) if x else [])

# Step 2: One-hot encode genres
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Step 3: Compute cosine similarity between movies based on genres
cosine_sim = cosine_similarity(genre_df, genre_df)

# TMDb API key and base URL
API_KEY = '93f05d46b0f5f9b2e7dfe9393a21dcf0'
BASE_URL = 'https://api.themoviedb.org/3'

# Function to fetch the movie poster URL from TMDb
def get_movie_poster(movie_title):
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(search_url)
    data = response.json()

    if data['results']:
        # Return the first movie's poster path
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None  # Return None if no poster is found

def get_content_based_recommendations_by_name(movie_name, cosine_sim=cosine_sim):
    # Find the movie by name
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if idx.empty:
        return "Movie not found in the dataset."

    idx = idx[0]  # Get the index of the movie
    similarity_scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores for the movie
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    similarity_scores = similarity_scores[1:11]  # Get top 10 similar movies

    recommended_movie_indices = [x[0] for x in similarity_scores]
    recommended_movie_titles = df['title'].iloc[recommended_movie_indices].tolist()
    
    # Fetch movie posters
    recommended_movie_posters = []
    for title in recommended_movie_titles:
        poster_url = get_movie_poster(title)
        recommended_movie_posters.append(poster_url)

    return list(zip(recommended_movie_titles, recommended_movie_posters))  # Return both titles and posters

# Route to display the frontend (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle movie recommendation requests (via AJAX)
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.json['movie_name']  # Get movie name from frontend
    recommended_movies = get_content_based_recommendations_by_name(movie_name)
    return jsonify({'recommended_movies': recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)

