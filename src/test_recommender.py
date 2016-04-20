import pandas as pd
import numpy as np
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords




## need to create a dictionary, model --> maybe instance variables of the class?
def create_dictionary(documents):
    '''
    INPUT: text documents (array of strings)
    OUTPUT: dictionary object, corpus
    '''
    ## Vectorize and store recipe text
    stoplist = set(stopwords.words('english'))
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

def create_model(corpus):
    ## Apply Tfidf model
    tfidf = models.TfidfModel(corpus)
    ## prepare for similarity queries
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = 1018) # num_features is len of dictionary
    return tfidf, index

def vectorize_restaurant_menu(name, db, stopwords, dictionary):
    '''
    INPUT: restaurant name (STRING), database connection, stopwords (LIST), dictionary object
    OUTPUT: menu vector (ARRAY)
    '''
    stopset = set(stopwords)
    ## Get 1 restaurant menu
    cursor = db.restaurants.find_one({'name' : name})
    menu = cursor['menu']

    ## Vectorize and prep menu text
    menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
    menu_string = ' '.join(menu_list)

    menu_tokens = [word for word in menu_string.lower().split() if word not in stopset]  #add 'description',  'available' to stopwords?
    menu_vector = dictionary.doc2bow(menu_tokens) ## need to pass in or just exist in environ?
    return menu_vector

def get_recommendations(menu_vector, num):
    '''
    INPUT: menu vector (ARRAY), number (INT) of recommendations requested
    OUTPUT: dataframe/series(?) of recommended recipes
    '''
    sims = index[tfidf[menu_vector]]
    rec_indices = np.argsort(sims)[:-10:-1] # gets top 10
    print data.loc[rec_indices, 'title']

## THIS WORKS!!!!!!!!!!!!!!!!1!!1!!!

if __name__ == '__main__':

    ## Connect to database
    conn = MongoClient()
    db = conn.project

    ## Get Recipe data
    cursor = db.recipes.find({}, {'title': 1, 'ingredients': 1, '_id' : 0})
    data = pd.DataFrame(list(cursor))
    data['ingredients'] = data['ingredients'].apply(lambda x: " ".join(x))
    documents = data['ingredients'].values

    
