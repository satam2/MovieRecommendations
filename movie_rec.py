# importing resources
import numpy as np
import pandas as pd
import difflib # finds closest match in database to user inputted movie name
from sklearn.feature_extraction.text import TfidfVectorizer # turns text into values
from sklearn.metrics.pairwise import cosine_similarity # finds similarity score btwn movies

movie_data = pd.read_csv('C:/Users/hurri/ml_projs/movie_rec/movies.csv')
#movie_data.head()
#rows_and_col = movie_data.shape

# feature selection for recommendation algo
features = ['genres','keywords','overview','popularity','production_companies','cast'] # content-based

# replace missing values with null string
for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')

# combine selected features
combined_feats = movie_data['genres']+' '+movie_data['keywords']+' '+movie_data['overview']+' '+str(movie_data['popularity'])+' '+movie_data['production_companies']+' '+movie_data['cast']

vectorizer = TfidfVectorizer() # turning text data into feature vectors
feature_vectors = vectorizer.fit_transform(combined_feats)

# finding similarity score using cosine similarity
similarity = cosine_similarity(feature_vectors)

# getting user input
movie_name = input('Enter the name of your favorite movie: ')

list_of_movie_titles = movie_data['title'].tolist()
find_match = difflib.get_close_matches(movie_name,list_of_movie_titles)
best_match = find_match[0]

index_of_movie = movie_data[movie_data.title == best_match]['index'].values[0]

# getting similar movies
similarity_score  = list(enumerate(similarity[index_of_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1],reverse = True) # takes similarity value and sorts in descending order
sorted_similar_movies.remove(sorted_similar_movies[0])

# print recommendations
print('\nBecause you liked '+best_match+', here are some recommended movies for you: ')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movie_data[movie_data.index==index]['title'].values[0]
    if(i<20): # prints top 20 similar movies
        print(str(i)+'.', title_from_index)
        i+=1
