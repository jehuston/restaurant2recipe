import pandas as pd
import numpy as np
import sys
import string
import time
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


## need to create a shared stopwords set, dictionary, index, model --> maybe instance variables of the class?
def prepare_documents(db):
    cursor = db.recipes.find({'title': { '$ne':'All Recipes' } }, {'title': 1, 'ingredients': 1, '_id' : 0})
    df = pd.DataFrame(list(cursor))
    df['ingredients'] = df['ingredients'].apply(lambda x: " ".join(x))
    documents = df['ingredients'].values
    return df, documents

def clean_text(documents):
    '''
    INPUT: array of strings
    OUTPUT: array of lists (ok?)
    '''
    stopset = set(stopwords.words('english'))
    stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
    wnl = WordNetLemmatizer()
    texts = []
    for doc in documents:
        words = doc.lower().split()
        tokens = []
        for word in words:
            if word not in stopset and not any(c.isdigit() for c in word): #filter stopwords and numbers
                token = wnl.lemmatize(word.strip(string.punctuation))
                tokens.append(token)
        texts.append(tokens)
    text_array = np.array(texts)
    return text_array

def initialize_model(pretrained_fp, model_fp):
    '''
    INPUT: filepath to pretrained word vectors
    OUTPUT: model saved to disk

    Only call this once on a machine that has the pretrained vectors downloaded (takes 3 min).
    Performance will be much faster loading saved model in future.
    '''
    model = models.Word2Vec.load_word2vec_format(pretrained_fp, binary=True)
    model.init_sims(replace=True)
    model.save(model_fp)

def load_model(filepath):
    '''
    INPUT: path to saved model
    OUTPUT: model
    '''
    model = models.Word2Vec.load(filepath, mmap='r')
    #print "loading saved model : ", time.time() - time2 #takes 26 seconds
    return model

def get_restaurant_menu(name, db):
    '''
    INPUT: restaurant name (STRING), database connection, stopwords (LIST), dictionary object
    OUTPUT: menu vector (ARRAY)
    '''
    stopset = set(stopwords.words('english'))
    stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
    ## Get 1 restaurant menu
    cursor = db.restaurants.find_one({'name' : name})
    menu = cursor['menu']

    ## Vectorize and prep menu text - Broken if menu field is empty! Only 16 of these though.
    menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
    menu_string = " ".join(menu_list)

    return menu_string


def create_doc_vectors(model, text_array): ## this is much more complicated than TfIDF
    '''
    INPUT: tokenized text documents (array of strings)
    OUTPUT: document vectors

    Translates words/tokens into word vectors, sums to create document vectors.
    '''
    doc_vectors = []
    unfound_words = 0
    for i in xrange(text_array.shape[0]):
        doc_vector = np.zeros((300,))
        for j in xrange(len(text_array[i])):
            try:
                doc_vector += model[text_array[i][j]]
            except KeyError: #if word not in word vectors, skip it
                unfound_words += 1
                continue
        doc_vectors.append(doc_vector)
    print "Num words not found: ", unfound_words
    return np.array(doc_vectors)

def create_index(doc_vectors):
    '''
    INPUT: array of document vectors
    OUTPUT: gensim index

    Makes a searchable index of document vectors and saves to disk.
    '''
    index = similarities.Similarity('/mnt/word2vec/index', doc_vectors, num_features = 300)
    return index

def get_recommendations(name, db, index, model, num, df):
    '''
    INPUT: index (), menu vector (ARRAY), number (INT) of recommendations requested
    OUTPUT: dataframe/series(?) of recommended recipes
    '''
    menu_string = get_restaurant_menu(name, db)
    menu_array = clean_text([menu_string])
    menu_vector = create_doc_vectors(model, menu_array)[0] #returns array length 1
    sims = index[menu_vector]
    rec_indices = np.argsort(sims)[:-(num + 1):-1] # gets top n
    return df.loc[rec_indices, 'title'], sims[rec_indices]


if __name__ == '__main__':

    restaurant_name = sys.argv[1]
    ## Connect to database
    conn = MongoClient()
    db = conn.project

    ## Get Recipe data
    data, documents = prepare_documents(db)
    recipe_array = clean_text(documents)

    ## first time only:
    #initialize_model('/home/ec2-user/vectors/GoogleNews-vectors-negative300.bin.gz')

    model = load_model('/mnt/word2vec/words')

    recipe_vectors = create_doc_vectors(model, recipe_array)
    recipe_index = create_index(recipe_vectors)

    ## translate menu into doc_vector
    # menu_string = get_restaurant_menu(restaurant_name, db)
    # menu_array = clean_text([menu_string])
    # menu_vector = create_doc_vectors(model, menu_array)[0]
    ## get recommendations
    results = get_recommendations(restaurant_name, db, recipe_index, model, 5, data)
    print results
