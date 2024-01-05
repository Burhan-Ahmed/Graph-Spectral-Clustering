import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform

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

# Calculate degree matrix
degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

print(degree_matrix)
print(degree_matrix.shape)

# Calculate Laplacian matrix
laplacian_matrix = degree_matrix - adjacency_matrix

# Display or use the Laplacian matrix
print("Laplacian matrix:")
print(laplacian_matrix)
print(laplacian_matrix.shape)
# Calculate degree matrix
degree_matrix_sqrt_inv = np.diag(1 / np.sqrt(np.sum(adjacency_matrix, axis=1)))

# Calculate normalized Laplacian matrix
laplacian_matrix = degree_matrix_sqrt_inv @ laplacian_matrix @ degree_matrix_sqrt_inv

# Display or use the normalized Laplacian matrix
print("Normalized Laplacian matrix:")
print(laplacian_matrix)


is_symmetric = np.allclose(laplacian_matrix, laplacian_matrix.T)
print("Is Laplacian matrix symmetric:", is_symmetric)

# Compute eigenvalues of the Laplacian matrix
eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

# Sort the eigenvalues in ascending order
sorted_indices = np.argsort(eigenvalues)
sorted_eigenvalues = eigenvalues[sorted_indices]

# Display the eigenvalues
print("Eigenvalues:")
print(sorted_eigenvalues)

# Check the properties
smallest_eigenvalue = sorted_eigenvalues[0]
second_smallest_eigenvalue = sorted_eigenvalues[1]

# Check if smallest eigenvalue is close to 0
print(f"Smallest eigenvalue: {smallest_eigenvalue}")
print(f"Second smallest eigenvalue: {second_smallest_eigenvalue}")

if smallest_eigenvalue >= 0:
    print("Smallest eigenvalue is greater than or equal to 0.")
else:
    print("Smallest eigenvalue is NOT greater than or equal to 0.")

if second_smallest_eigenvalue > 0:
    print("Second smallest eigenvalue is greater than 0.")
else:
    print("Second smallest eigenvalue is NOT greater than 0.")

if np.all(sorted_eigenvalues > 0):
    print("All eigenvalues are positive.")
else:
    print("NOT all eigenvalues are positive.")

# Find the index of the smallest eigenvalue
index_of_min_eigenvalue = sorted_indices[0]

# Retrieve the corresponding eigenvector
corresponding_eigenvector = eigenvectors[:, index_of_min_eigenvalue]

# Display the corresponding eigenvector
print("Corresponding Eigenvector for the smallest eigenvalue:")
print(corresponding_eigenvector)

# Calculate the left-hand side of the eigenvalue equation: L * v
lhs = np.dot(laplacian_matrix, corresponding_eigenvector)

# Calculate the right-hand side of the eigenvalue equation: lambda * v
rhs = smallest_eigenvalue * corresponding_eigenvector

# Check if both sides of the equation are approximately equal
verification_result = np.allclose(lhs, rhs)

# Display the verification result
if verification_result:
    print("The corresponding eigenvector satisfies the eigenvalue equation.")
else:
    print("The corresponding eigenvector does not satisfy the eigenvalue equation.")


# Assuming 'laplacian_matrix' is the unnormalized Laplacian matrix calculated earlier

# Calculate the degree matrix
degree_vector = np.sum(adjacency_matrix, axis=1)  # Calculate degree vector
degree_matrix_sqrt_inv = np.diag(1.0 / np.sqrt(degree_vector))  # D^{-1/2}

# Compute normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}
normalized_laplacian = np.dot(np.dot(degree_matrix_sqrt_inv, laplacian_matrix), degree_matrix_sqrt_inv)

# Calculate eigenvectors of the normalized Laplacian
#laplacian_matrix
eigenvalues_norm, eigenvectors_norm = np.linalg.eig(normalized_laplacian)

# Find the index of the smallest eigenvalue of the normalized Laplacian
index_min_eigenvalue_norm = np.argmin(eigenvalues_norm)
smallest_eigenvector_norm = eigenvectors_norm[:, index_min_eigenvalue_norm]

# Retrieve the corresponding eigenvector of the unnormalized Laplacian
corresponding_eigenvector_unnorm = eigenvectors[:, index_of_min_eigenvalue]

# Calculate D^{-1/2}v
d_sqrt_inv_times_v = np.dot(degree_matrix_sqrt_inv, corresponding_eigenvector_unnorm)

# Check if the eigenvector relationship holds
is_proportional = np.allclose(smallest_eigenvector_norm, d_sqrt_inv_times_v)

# Display the verification result
if is_proportional:
    print("The eigenvector relationship holds: D^{-1/2}v is proportional to the smallest eigenvector of the normalized Laplacian.")
else:
    print("The eigenvector relationship does not hold.")