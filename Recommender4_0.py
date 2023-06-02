from enum import unique
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.linalg as lag
from tqdm import tqdm

movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
power_laws_user = np.array(ratings['userId'].value_counts)
ratings.drop('timestamp', inplace=True, axis=1)

original_user_ids = np.array(ratings['userId'].unique()) 
original_movie_ids = np.array(ratings['movieId'].unique()) 
internal_user_ids = np.linspace(0,len(original_user_ids) - 1, len(original_user_ids),dtype="int") 
internal_movie_ids = np.linspace(0,len(original_movie_ids) - 1, len(original_movie_ids),dtype="int") 

ratings1 = ratings.to_numpy()

# Re-indexing user and movie ids:
user_remap = {original_user_ids[i]: internal_user_ids[i] for i in range(len(original_user_ids))}
movie_remap = {original_movie_ids[i]: internal_movie_ids[i] for i in range(len(original_movie_ids))}

for idx, row in tqdm(enumerate(ratings1)):
    ratings1[idx][0] = user_remap.get(row[0], row[0])  
    ratings1[idx][1] = movie_remap.get(row[1], row[1]) 

user_freq = []

# Save reindexed array:
with open('ratings1.npy', 'wb') as f:
    np.save(f,ratings1)

# Loading the reindexed array:
with open('ratings1.npy', 'rb') as f:
    ratings = np.load(f)

# Creating Datastructures for users:
start_indices_u = np.where(np.roll(ratings[:,0],1)!=ratings[:,0])[0]
movie_tuples = list(zip(ratings[:,1], ratings[:,0]))
user_len = len(start_indices_u)
end_indices_u = []

# Getting number of times each user rated movies and creating end index datastructure for users:
for i,freq in enumerate(start_indices_u):
    if i > 0:
        user_freq.append(freq - start_indices_u[i-1])
        end_indices_u.append(freq)
    if i == (user_len-1):
        user_freq.append(user_len - freq)
        end_indices_u.append(len(movie_tuples))

# Creting Datastructures for movies:
ratings_m_sorted = ratings[ratings[:, 1].argsort()]
start_indices_v = np.where(np.roll(ratings_m_sorted[:, 1],1)!=ratings_m_sorted[:, 1])[0]
user_tuples = list(zip(ratings_m_sorted[:, 0], ratings_m_sorted[:, 2]))

movie_len = len(start_indices_v)
movie_freq = []
end_indices_v = []

# Getting number of times each movie was rated and creating end index datastructure for movies:
for i,freq in enumerate(start_indices_v):
    if i > 0:
        movie_freq.append(freq - start_indices_v[i-1])
        end_indices_v.append(freq)
    if i == (movie_len-1):
        movie_freq.append(movie_len - freq)
        end_indices_v.append(len(user_tuples))

#Plotting Power Laws:
user_freq, unique_user_freq = np.array(np.unique(user_freq, return_counts=True))
movie_freq, unique_movie_freq = np.array(np.unique(movie_freq, return_counts=True))
plt.scatter(user_freq, unique_user_freq, label="Movie IDs")
plt.scatter(movie_freq, unique_movie_freq, label="User IDs")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Power Laws Plot for Movies Dataset")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.show()

#ratings['rating'].plot(kind='hist', logy=True, title="Distribution of Ratings for Movies Dataset", xlabel="Rating", ylabel="Frequency")
#plt.show()

# Creating variables for MLE calculations:
latent_dim = 20
n_users = len(np.unique(ratings[:,0]))
n_movies = len(np.unique(ratings[:,1]))
u_mat = np.reshape(np.random.normal(0, (2.5/np.sqrt(latent_dim)), n_users * latent_dim),(n_users,latent_dim))
v_mat = np.reshape(np.random.normal(0, (2.5/np.sqrt(latent_dim)), n_movies * latent_dim),(n_movies,latent_dim))
u_bias = np.zeros(n_users)
v_bias = np.zeros(n_movies)
alph = 0.015
lamb = 0.3
tauu = np.sqrt(latent_dim)

# Funtion to calculate cholesky decomposition of a matrix and return the lower triangular matrix as well as its inverse
def calc_cholesky(input_matrix):
    lower_chol = lag.cholesky(input_matrix, lower=True)     #lower=true
    inverse_mat = np.linalg.inv(lower_chol)
    result = np.dot(inverse_mat.T, inverse_mat)
    return result

# Function to collect a rating:
def capture_rating(id_of_user, id_of_movie):
    start_index = start_indices_u[id_of_user]
    end_index = end_indices_u[id_of_user]
    selected_tuples = movie_tuples[start_index:end_index]
    mov_id, rating = [val for val in selected_tuples if val[0] == id_of_movie]
    return rating

# Function to collect tuples which contain all movies a user rated:
def capture_user_movies(id_of_user):
    start_index = start_indices_u[id_of_user]
    end_index = end_indices_u[id_of_user]
    result = movie_tuples[start_index:end_index]
    return result

# Function to collect tuples which contain all users that rated a movie:
def capture_movie_users(id_of_movie):
    start_index = start_indices_v[id_of_movie]
    end_index = end_indices_v[id_of_movie]
    result = user_tuples[start_index:end_index]
    return result

# Function to update the user bias:
def update_user_bias():
  for user_id in range(n_users):
      numer = np.float64(0)
      denom = np.float64(alph)
      for mov,rating in capture_user_movies(user_id):
          mov = int(mov)
          numer += lamb * (rating - 1 * (np.dot(u_mat[user_id:user_id+1,:], v_mat[mov:mov+1,:].T) + v_bias[mov]))
          denom += lamb
      u_bias[user_id] = numer / denom

# Function to update the movie bias:
def update_movie_bias():
  for movie_id in range(n_movies):
      numer = 0
      denom = alph
      for user,rating in capture_movie_users(movie_id):
          user = int(user)
          numer += lamb * (rating -1* (np.dot(u_mat[user:user+1,:], v_mat[movie_id:movie_id+1,:].T) + u_bias[user]))
          denom += lamb
      v_bias[movie_id] = numer / denom

# Function to update user vectors:
def update_user():
    for u_id, val_ in enumerate(u_bias):
        new_bias = np.reshape(np.zeros(latent_dim),(1,latent_dim))
        temp_matrix = np.zeros((latent_dim, latent_dim))
        temp_vec = np.zeros((latent_dim, 1))
        for mov,rating in capture_user_movies(u_id):
            mov = int(mov)
            temp_matrix += lamb * np.dot(v_mat[mov:mov+1,:].T, v_mat[mov:mov+1,:])
            temp_vec += lamb * (v_mat[mov:mov+1,:].T * (rating - val_ - v_bias[mov]))

        inverted_matrix = calc_cholesky(temp_matrix + tauu * np.eye(latent_dim))
        new_bias = np.dot(inverted_matrix, temp_vec).T

        u_mat[u_id] = new_bias

# Function to update movie vectors:
def update_movie():
    for v_id, val_ in enumerate(v_bias):
        new_bias = np.reshape(np.zeros(latent_dim),(1,latent_dim))
        temp_matrix = np.zeros((latent_dim, latent_dim))
        temp_vec = np.zeros((latent_dim, 1))
        for user_,rating in capture_movie_users(v_id):
            user_ = int(user_)
            temp_matrix += lamb * np.dot(u_mat[user_:user_+1,:].T, u_mat[user_:user_+1,:])
            temp_vec += lamb * (u_mat[user_:user_+1,:].T * (rating - u_bias[user_] - val_))

        inverted_matrix = calc_cholesky(temp_matrix + tauu * np.eye(latent_dim))
        new_bias = np.dot(inverted_matrix, temp_vec).T

        v_mat[v_id] = new_bias

# Function to calculate loss and RMSE:
def calculate_loss():
    loss = 0
    rmse_pred = 0
    rmse_actual = 0
    rmse_val = 0
    rmse_counter = 0
    for user_id in range(n_users):
      for mov,rating in capture_user_movies(user_id):
          mov = int(mov)
          fact_1 = -0.5 * lamb *  (rating - (np.dot(u_mat[user_id:user_id+1,:], v_mat[mov:mov+1,:].T) + u_bias[user_id] + v_bias[mov]))**2
          fact_2 = -0.5 * tauu * np.dot(u_mat[user_id:user_id+1,:], u_mat[user_id:user_id+1,:].T)
          fact_3 = -0.5 * tauu * np.dot(v_mat[mov:mov+1,:], v_mat[mov:mov+1,:].T)
          loss += -1 * (fact_1 + fact_2 + fact_3)
          rmse_pred = np.dot(u_mat[user_id:user_id+1,:], v_mat[mov:mov+1,:].T) + u_bias[user_id] + v_bias[mov]
          rmse_actual = rating
          rmse_val += (rmse_actual - rmse_pred)**2
          rmse_counter += 1
    
    rmse_val = np.sqrt((1/rmse_counter) * rmse_val)
    print("The current RMSE is: " + str(rmse_val))
    print("The current loss is: " + str(loss))
    return loss

# Main Loop for updating values and calculating the loss:
num_iterations = 50
iteration_plot = np.linspace(1,num_iterations + 1, num_iterations)
loss_plot = []
for iteration in tqdm(range(num_iterations)):
    print("Current Iteration: " + str(iteration))
    update_user_bias()
    update_user()
    update_movie_bias()
    update_movie()
    loss_plot.append(calculate_loss())

plt.scatter(iteration_plot, loss_plot)
plt.xlabel("Iteration Number")
plt.ylabel("Magnitude of Loss")
plt.title("Plot Showing Loss as a Function of Iterations:")
plt.yscale("log")
plt.show()

# Saving user and movie details for model:
np.savetxt('user_mat.csv', u_mat, delimiter=',')
np.savetxt('movie_mat.csv', v_mat, delimiter=',')
np.savetxt('user_bias.csv', u_bias, delimiter=',')
np.savetxt('movie_bias.csv', v_bias, delimiter=',')

with open('user_mat.npy', 'wb') as f:
    np.save(f,u_mat)

with open('movie_mat.npy', 'wb') as f:
    np.save(f,v_mat)

with open('user_bias.npy', 'wb') as f:
    np.save(f,u_bias)

with open('movie_bias.npy', 'wb') as f:
    np.save(f,v_bias)

"""u_mat = np.loadtxt('user_mat.csv', delimiter=',')
v_mat = np.loadtxt('movie_mat.csv', delimiter=',')
u_bias = np.loadtxt('user_bias.csv', delimiter=',')
v_bias = np.loadtxt('movie_bias.csv', delimiter=',')"""

