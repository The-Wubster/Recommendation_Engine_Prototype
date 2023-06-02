# 21560579
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

movies = pd.read_csv('./movies.csv')
u_mat = np.loadtxt('./user_mat.csv', delimiter=',')
v_mat = np.loadtxt('./movie_mat.csv', delimiter=',')
u_bias = np.loadtxt('./user_bias.csv', delimiter=',')
v_bias = np.loadtxt('./movie_bias.csv', delimiter=',')
genome_data = np.loadtxt('./genome_mat_small.csv', delimiter=',')

# One-hot encoding the genres column:
# Splitting the genres for each movie into a list:
for idx, val_ in tqdm(enumerate(movies['genres'])):
    movies['genres'][idx] = val_.split('|')

# Encoding the data as integers:
column_as_list = movies['genres'].tolist()
unique_values = set(val_ for sublist in column_as_list for val_ in sublist)
value_to_index = {val__: idx for idx, val__ in enumerate(unique_values)}

# Creating an empty array to store encoded data:
encoded_matrix = np.zeros((len(column_as_list), len(unique_values)), dtype=int)

# Performing One-Hot encoding:
for idx, val_ in tqdm(enumerate(column_as_list)):
    for val__ in val_:
        idx_ = value_to_index[val__]
        encoded_matrix[idx, idx_] = 1

# Plotting 2D representation of 10 ratings to look at ability of model:
# Completing PCA Analysis:
latent_dim = 2
pca_genre = PCA(n_components=latent_dim)
genre_reduced = pca_genre.fit_transform(encoded_matrix)

# Movie Indices to extract:
indices = np.array([0, 2355, 7355, 6743, 7324, 8151, 3131, 3132, 6958, 6959])
plot_array = np.zeros((10, 2))
count = 0
for idx in indices:
    plot_array[count, 0] = genre_reduced[idx, 0]
    plot_array[count, 1] = genre_reduced[idx, 1]
    count += 1

# Plotting the 2D Genre Embedding
fig, ax = plt.subplots()
ax.scatter(plot_array[0:3, 0], plot_array[0:3, 1], color="dodgerblue", label="Toy Story Movies")
ax.scatter(plot_array[3:6, 0], plot_array[3:6, 1], color="hotpink", label="Ironman Movies")
ax.scatter(plot_array[6:10, 0], plot_array[6:10, 1], color="green", label="Revenge of the Nerds Movies")
ax.set_title("2D Embedding of Genre Data")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend()
plt.show()

# Adding a dummy user with a rating for Iron Man
user_rating = 5.0
user_bias = 0
movie_id = 1210#6743     # Movie Id based on row in dataframe (Iron Man)
new_user_vec = (user_rating + user_bias + v_bias[movie_id]) * v_mat[movie_id]
print(np.shape(new_user_vec))

# Adding Genome Data to be Reduced using PCA:
"""genome_data = pd.read_csv('./genome-scores.csv')
movie_ids = genome_data['movieId'].to_numpy()
new_genome = genome_data[genome_data['movieId'].isin(movie_ids)]
np.savetxt('genome_mat_small.csv', new_genome, delimiter=',')"""

# Remapping indices of the movies:
"""original_movie_ids = np.array(genome_data['movieId'].unique())
internal_movie_ids = np.linspace(0,len(original_movie_ids) - 1, len(original_movie_ids),dtype="int")"""

# Re-indexing user and movie ids:
"""movie_remap = {original_movie_ids[i]: internal_movie_ids[i] for i in range(len(original_movie_ids))}
tqdm(genome_data.replace({"movieId": movie_remap},inplace=True))"""
#print(np.max(genome_data['movieId']))

# Creating Matrix to store tag data:
genome_mat = np.zeros((len(np.unique(genome_data[:, 0])), len(np.unique(genome_data[:, 1]))))
#genome_data = genome_data.to_numpy()

# Populating Matrix:
count = 0
new_val = 1
for row in tqdm(genome_data):
    if int(row[0]) > new_val:
        new_val = int(row[0])
        count += 1
        
    genome_mat[count, (int(row[1]) - 1)] = row[2]

print(genome_mat[1:10, :])
# Saving the resulting matrix:
np.savetxt('genome_mat.csv', genome_mat, delimiter=',')

# Completing PCA Analysis to reduce the genome data dimension to the same as that of the latent dimension:
latent_dim = 20
pca_genome = PCA(n_components=latent_dim)
genome_reduced = pca_genome.fit_transform(genome_mat)

# Getting user predictions for all movies and outputting the top 20:
new_ratings = []
rated_ids = []

# Calculating ratings using only new user data and their previously rated movie:
"""for v_id, val_ in enumerate(v_bias):
    new_ratings.append((np.dot(new_user_vec, v_mat[v_id:v_id+1,:].T) + val_* 0.05) + 2.5) # 2.5 Added to normalise the values.
    rated_ids.append(v_id)"""

# Calculating ratings using new user data, their previously rated movie and one-hot encoded genre data:
for v_id, val_ in enumerate(v_bias):
    new_ratings.append((np.dot(new_user_vec, (v_mat[v_id:v_id+1,:] + encoded_matrix[v_id:v_id+1,:]).T) + val_ * 0.05) + 2.5) # 2.5 Added to normalise the values.
    rated_ids.append(v_id)

"""# Calculating ratings using new user data, their previously rated movie, one-hot encoded genre data and the reduced dimension genome data:
for v_id, val_ in enumerate(v_bias):
    new_ratings.append((np.dot(new_user_vec, (v_mat[v_id:v_id+1,:] + encoded_matrix[v_id:v_id+1,:] + genome_reduced[v_id:v_id+1,:]).T) + val_* 0.05) + 2.5) # 2.5 Added to normalise the values.
    rated_ids.append(v_id)"""
movie_names = movies['title']
movie_names_rated = []

# Outputting the results
for idx, val_ in enumerate(rated_ids):
    movie_names_rated.append(movie_names[val_])

rated_df = pd.DataFrame(list(zip(movie_names_rated, new_ratings)), columns=['Movie', 'Rating']).sort_values(by=['Rating'], ascending=False)
print(rated_df.head(200))
