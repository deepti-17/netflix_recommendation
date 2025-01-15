import requests
import pandas as pd
import time

# Your TMDb API key
api_key = '93f05d46b0f5f9b2e7dfe9393a21dcf0'
base_url = "https://api.themoviedb.org/3/movie/popular"

# Create an empty list to store movie data
movies_data = []

# Number of movies to fetch
total_movies = 10000
movies_per_page = 20
pages_to_fetch = total_movies // movies_per_page

# Loop through pages to collect movie data
for page in range(1, pages_to_fetch + 1):
    response = requests.get(base_url, params={'api_key': api_key, 'page': page})
    
    if response.status_code == 200:
        data = response.json()

        for movie in data['results']:
            movie_info = {
                'movieId': movie['id'],
                'title': movie['title'],
                'overview': movie['overview'],
                'genres': ', '.join([str(genre) for genre in movie['genre_ids']]),  # Genre IDs as a string
                'rating': movie['vote_average'],
            }
            movies_data.append(movie_info)

        print(f"Page {page}/{pages_to_fetch} fetched successfully.")
    else:
        print(f"Error fetching page {page}, status code: {response.status_code}")
        break

    # Delay to avoid hitting rate limits
    time.sleep(1)  # 1-second delay between requests

# Convert to DataFrame
df = pd.DataFrame(movies_data)

# Display the first few rows
print(df.head())

# Optionally, save the data to a CSV file
df.to_csv('movies_10000_dataset.csv', index=False)
