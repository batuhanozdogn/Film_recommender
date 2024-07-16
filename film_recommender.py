import pandas as pd
from sklearn.metrics import pairwise_distances

movies = pd.read_csv("/Users/batuhanozdogan/Downloads/ml-latest-small/movies.csv")
ratings = pd.read_csv("/Users/batuhanozdogan/Downloads/ml-latest-small/ratings.csv")

df = pd.merge(movies, ratings, on="movieId")
df = df[["userId", "title", "rating"]]

pivot = df.pivot_table(index="title", columns="userId", values="rating", fill_value=0)
similarity_matrix = pairwise_distances(pivot, metric="correlation")
indexs = list(pivot.index)

film_index = indexs.index("Call Me by Your Name (2017)")
similarity_scores = similarity_matrix[film_index]
similar_movies_indices = similarity_scores.argsort()[1:6]

for i in similar_movies_indices:
    print(indexs[i])
