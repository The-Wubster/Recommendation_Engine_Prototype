# 21560579
from enum import unique
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.linalg as lag
from tqdm import tqdm
from sklearn.decomposition import PCA

# Loading the required datasets:
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
power_laws_user = np.array(ratings['userId'].value_counts)
movies_25 = pd.read_csv('./ml-25m/movies.csv')
ratings_25 = pd.read_csv('./ml-25m/ratings.csv')

# Flag to output data distribution plots:
plot_flag = 0
plot_power =0

if plot_flag == 1:
    # Plotting Several Distributions of the data:
    fig, ax = plt.subplots(3, 2)
    fig.suptitle('Plots Showing Distribution of Data in 25 Million and \n 100 000 Ratings Datasets Respectively')
    fig.subplots_adjust(hspace=0.6)
    fig.plots_adjust(wspace=0.3)
    # Plotting the counts of the number of movies a user rated for the 25 mil dataset:
    ax[0,0].scatter(np.array(ratings_25['userId'].value_counts().value_counts().index), np.array(ratings_25['userId'].value_counts().value_counts()), s=4, color='c')
    ax[0,0].set_title('Distribution of User IDs')
    ax[0,0].set_xlabel('Number of Movies a User Rated')
    ax[0,0].set_ylabel('Value Count')
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')

    # Plotting the counts of the number of times a movie was rated for the 25 mil dataset
    ax[1,0].scatter(np.array(ratings_25['movieId'].value_counts().value_counts().index), np.array(ratings_25['movieId'].value_counts().value_counts()), s=4, color='coral')
    ax[1,0].set_title("Distribution of Movie IDs")
    ax[1,0].set_xlabel('Number of Times a Movie was Rated')
    ax[1,0].set_ylabel('Value Count')
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')

    # Plotting the rating distributions for the 25 mil dataset:
    ax[2,0].bar(np.array(ratings_25['rating'].value_counts().index), np.array(ratings_25['rating'].value_counts()), color='orchid')
    ax[2,0].set_title("Distribution of Ratings")
    ax[2,0].set_xlabel('Number of Times a Rating was Given')
    ax[2,0].set_ylabel('Value Count')
    ax[2,0].set_xlim(0, 5)
    ax[2,0].set_yscale('log')

    # Plotting the counts of the number of movies a user rated for the 100k dataset:
    ax[0,1].scatter((ratings['userId'].value_counts().value_counts().index), np.array(ratings['userId'].value_counts().value_counts()), s=4, color='c')
    ax[0,1].set_title('Distribution of User IDs')
    ax[0,1].set_xlabel('Number of Movies a User Rated')
    ax[0,1].set_ylabel('Value Count')
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')

    # Plotting the counts of the number of times a movie was rated for the 100k dataset
    ax[1,1].scatter(np.array(ratings['movieId'].value_counts().value_counts().index), np.array(ratings['movieId'].value_counts().value_counts()), s=4, color='coral')
    ax[1,1].set_title("Distribution of Movie IDs")
    ax[1,1].set_xlabel('Number of Times a Movie was Rated')
    ax[1,1].set_ylabel('Value Count')
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')

    # Plotting the rating distributions for the 100k dataset:
    ax[2,1].bar(np.array(ratings['rating'].value_counts().index), np.array(ratings['rating'].value_counts()), color='orchid')
    ax[2,1].set_title("Distribution of Ratings")
    ax[2,1].set_xlabel('Number of Times a Rating was Given')
    ax[2,1].set_ylabel('Value Count')
    ax[2,1].set_xlim(0, 5)
    ax[2,1].set_yscale('log')

    plt.show()

    nbins = 10
    hist, bin_spec = np.histogram(ratings_25['userId'].value_counts(), nbins, density=True)
    a, b = min(bin_spec), max(bin_spec)
    dx = (b-a)/nbins

    figure, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2, figsize=(10, 5))
    figure.subplots_adjust(hspace=0.4)
    figure.suptitle('Plots Showing Distribution of Data in 25 Million and \n 100 000 Ratings Datasets Respectively')
    ax_1.bar(bin_spec[:-1], hist, width = dx, color="springgreen")
    ax_1.set_title("Number of Movies Rated by User Distribution")
    ax_1.set_xlabel("Number of Rated Movies")
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')

    hist1, bin_spec1 = np.histogram(ratings_25['movieId'].value_counts(), nbins, density=True)
    a, b = min(bin_spec1), max(bin_spec1)
    dx = (b-a)/nbins
    a, b = min(bin_spec1), max(bin_spec1)
    ax_3.bar(bin_spec1[:-1], hist1, width = dx, color="dodgerblue")
    ax_3.set_title("Number of Times Movie was Rated Distribution")
    ax_3.set_xlabel("Number of Ratings for Movie")
    ax_3.set_xscale('log')
    ax_3.set_yscale('log')

    hist2, bin_spec2 = np.histogram(ratings['userId'].value_counts(), nbins, density=True)
    a, b = min(bin_spec2), max(bin_spec2)
    dx = (b-a)/nbins
    ax_2.bar(bin_spec2[:-1], hist2, width = dx, color="springgreen")
    ax_2.set_title("Number of Movies Rated by User Distribution1")
    ax_2.set_xlabel("Number of Rated Movies")
    ax_2.set_xscale('log')
    ax_2.set_yscale('log')

    hist3, bin_spec3 = np.histogram(ratings['movieId'].value_counts(), nbins, density=True)
    a, b = min(bin_spec3), max(bin_spec3)
    dx = (b-a)/nbins
    a, b = min(bin_spec3), max(bin_spec3)
    ax_4.bar(bin_spec3[:-1], hist3, width = dx, color="dodgerblue")
    ax_4.set_title("Number of Times Movie was Rated Distribution1")
    ax_4.set_xlabel("Number of Ratings for Movie")
    ax_4.set_xscale('log')
    ax_4.set_yscale('log')

    plt.show()

original_user_ids = np.array(ratings['userId'].unique()) #
original_movie_ids = np.array(ratings['movieId'].unique()) #
internal_user_ids = np.linspace(0,len(original_user_ids) - 1, len(original_user_ids),dtype="int") #
internal_movie_ids = np.linspace(0,len(original_movie_ids) - 1, len(original_movie_ids),dtype="int") #
ratings.drop('timestamp', inplace=True, axis=1)
#ratings1 = ratings.to_numpy()

# Re-indexing user and movie ids:
user_remap = {original_user_ids[i]: internal_user_ids[i] for i in range(len(original_user_ids))}
movie_remap = {original_movie_ids[i]: internal_movie_ids[i] for i in range(len(original_movie_ids))}
tqdm(ratings.replace({"userId": user_remap,"movieId": movie_remap},inplace=True))

user_freq = []

# Save reindexed dataframe:
ratings.to_csv('ratings1.csv', index=False)
#np.savetxt("ratings1.csv", ratings1, delimiter=",")
ratings = pd.read_csv('./ratings1.csv')

# Creating Datastructures for users:
start_indices_u = np.where(np.roll(ratings['userId'],1)!=ratings['userId'])[0]
movie_tuples = list(zip(ratings['movieId'], ratings['rating']))
print()
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

# Creating Datastructures for movies:
ratings_m_sorted = ratings.sort_values(by=['movieId'])
start_indices_v = np.where(np.roll(ratings_m_sorted['movieId'],1)!=ratings_m_sorted['movieId'])[0]
user_tuples = list(zip(ratings_m_sorted['userId'], ratings_m_sorted['rating']))

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
if plot_power == 1:
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
    plt.show()

#ratings['rating'].plot(kind='hist', logy=True, title="Distribution of Ratings for Movies Dataset", xlabel="Rating", ylabel="Frequency")
#plt.show()

# Creating variables for MLE calculations:
latent_dim = 20
n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
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
          numer += lamb * (rating - 1 * (np.dot(u_mat[user_id:user_id+1,:], v_mat[mov:mov+1,:].T) + v_bias[mov]))
          denom += lamb
      u_bias[user_id] = numer / denom

# Function to update the movie bias:
def update_movie_bias():
  for movie_id in range(n_movies):
      numer = 0
      denom = alph
      for user,rating in capture_movie_users(movie_id):
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
    return loss, rmse_val

# Main Loop for updating values and calculating the loss:
num_iterations = 20
iteration_plot = np.linspace(1,num_iterations + 1, num_iterations)
loss_plot = []
rmse_plot = []
for iteration in tqdm(range(num_iterations)):
    print("Current Iteration: " + str(iteration))
    update_user_bias()
    update_user()
    update_movie_bias()
    update_movie()
    l_p, rmse_p = calculate_loss()
    loss_plot.append(l_p)
    rmse_plot.append(rmse_p)

# Plotting Loss Values
plt.scatter(iteration_plot, loss_plot, color='gold', s=18)
plt.xlabel("Iteration Number")
plt.ylabel("Magnitude of Loss")
plt.title("Plot Showing Loss as a Function of Iterations:")
plt.yscale("log")
plt.show()

# Plotting RMSE Values
plt.scatter(iteration_plot, rmse_plot, color='lawngreen', s=18)
plt.xlabel("Iteration Number")
plt.ylabel("Magnitude of RMSE")
plt.title("Plot Showing Root-Mean-Squared-Error (RMSE) as a Function of Iterations:")
plt.legend()
plt.show()

# Saving user and movie details for model:
np.savetxt('user_mat.csv', u_mat, delimiter=',')
np.savetxt('movie_mat.csv', v_mat, delimiter=',')
np.savetxt('user_bias.csv', u_bias, delimiter=',')
np.savetxt('movie_bias.csv', v_bias, delimiter=',')

u_mat = np.loadtxt('user_mat.csv', delimiter=',')
v_mat = np.loadtxt('movie_mat.csv', delimiter=',')
u_bias = np.loadtxt('user_bias.csv', delimiter=',')
v_bias = np.loadtxt('movie_bias.csv', delimiter=',')

# Plotting 2D representation of 10 ratings to look at ability of model:
# Completing PCA Analysis:
print(np.shape(v_mat))
latent_dim = 2
pca_v = PCA(n_components=latent_dim)
v_reduced = pca_v.fit_transform(v_mat)
print(np.shape(v_reduced))

# Movie Indices to extract:
indices = np.array([0, 2355, 7355, 6743, 7324, 8151, 3131, 3132, 6958, 6959])
plot_array = np.zeros((10, 2))
count = 0
for idx in indices:
    plot_array[count, 0] = v_reduced[idx, 0]
    plot_array[count, 1] = v_reduced[idx, 1]
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