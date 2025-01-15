import pandas as pd

# Load the dataset
df = pd.read_csv('movies_10000_dataset.csv')

# Data preprocessing steps
df = df.dropna(subset=['movieId', 'title', 'rating'])  # Drop rows with missing key columns
df = df[['movieId', 'title', 'genres', 'rating']]  # Keep only relevant columns

# Save the cleaned data
df.to_csv('cleaned_movies_10000.csv', index=False)

print("Data Preprocessing Done!")
print(df.head())

