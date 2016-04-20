import pandas as pd
import numpy as np
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords


## need to create a shared stopwords set, dictionary, index, model --> maybe instance variables of the class?
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

def vectorize_restaurant_menu(name, db, dictionary):
    '''
    INPUT: restaurant name (STRING), database connection, stopwords (LIST), dictionary object
    OUTPUT: menu vector (ARRAY)
    '''
    stopset = set(stopwords.words('english'))
    stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
    ## Get 1 restaurant menu
    cursor = db.restaurants.find_one({'name' : name})
    menu = cursor['menu']

    ## Vectorize and prep menu text
    menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
    menu_string = " ".join(menu_list)

    menu_tokens = [word for word in menu_string.lower().split() if word not in stopset]
    menu_vector = dictionary.doc2bow(menu_tokens)
    return menu_vector

def get_recommendations(index, menu_vector, num, model, df):
    '''
    INPUT: menu vector (ARRAY), number (INT) of recommendations requested
    OUTPUT: dataframe/series(?) of recommended recipes
    '''
    sims = index[model[menu_vector]]
    rec_indices = np.argsort(sims)[:-num:-1] # gets top 10
    return df.loc[rec_indices, 'title']

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
    dictionary, corpus = create_dictionary(documents)
    tfidf, index = create_model(corpus)

    menu_vec = vectorize_restaurant_menu('Hogwash', db, dictionary)
    recs = get_recommendations(index, menu_vec, 5, tfidf, data)
    print recs
