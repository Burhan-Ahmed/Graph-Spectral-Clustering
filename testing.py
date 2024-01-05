import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('songs.csv')  # Replace 'songs.csv' with your file path

# Assuming 'x' contains the features for computation
# Replace this with your specific column names or feature selection
x = data[['artist_familiarity', 'artist_ hotttnesss', 'artist_num_songs','release','duration','energy',
          'pitches','timbre','loudness','danceability']]  # Replace column names accordingly

# Function to compute Euclidean distance matrix
def euclidean_distance_matrix(x):
    n = x.shape[1] 
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Compute Euclidean distance between points i and j
            distance = np.linalg.norm(x.iloc[i] - x.iloc[j])
            adjacency_matrix[i][j] = distance
            #adjacency_matrix[j][i] = distance  # Since it's a symmetric matrix

    return adjacency_matrix

# Compute the adjacency matrix
adjacency_matrix = euclidean_distance_matrix(x)
# You can use this adjacency matrix for further processing or analysis

print(adjacency_matrix)
print(adjacency_matrix.shape)