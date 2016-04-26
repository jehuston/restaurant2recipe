import pandas as pd
import numpy as np
import sys
import string
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


## need to create a shared stopwords set, dictionary, index, model --> maybe instance variables of the class?

def create_dictionary(documents):
    '''
    INPUT: text documents (array of strings)
    OUTPUT: gensim dictionary object, corpus
    '''
    ## Vectorize and store recipe text
    stoplist = set(stopwords.words('english'))
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts] ## convert to BOW
    return dictionary, corpus # could write to disk instead?

def use_word2vec(filename):
    model = models.Word2Vec.load_word2vec_format(filename, binary=True)
    return model

## translate recipe data into word2vec vectors. Store somewhere.

## vectorize restaurant menu and find most similar recipe vector.

def create_model(corpus, dict_size): ## model should be passed in.
    '''
    INPUT: Model class, corpus (ARRAY)
    OUTPUT: trained model, index (for similarity scoring)
    '''
    ## Apply model
    model = models.TfidfModel(corpus)
    ## prepare for similarity queries - unlikely to be memory constrained (< 100K docs) so won't write to disk
    index = similarities.SparseMatrixSimilarity(model[corpus], num_features = dict_size) # num_features is len of dictionary
    return model, index

def get_restaurant_menu(name, db, dictionary):
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
    ## Broken if menu field is empty! Only 16 of these though.
    menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
    menu_string = " ".join(menu_list)
    #
    # menu_tokens = [word for word in menu_string.lower().split() if word not in stopset]
    # menu_vector = dictionary.doc2bow(menu_tokens)
    return menu_string

def clean_text(documents):
    '''
    INPUT: array of strings
    OUTPUT: array of lists (ok?)
    '''
    stopset = set(stopwords.words('english'))
    stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
    wnl = WordNetLemmatizer()
    texts = [[wnl.lemmatize(word.strip(string.punctuation)) for word in\
                document.lower().replace("/", "").split()\
                if word not in stopset if not any(c.isdigit() for c in word)]\
            for document in documents]    ## this is hideous
    text_array = np.array(texts)
    return text_array

def get_recommendations(index, menu_vector, num, model, df):
    '''
    INPUT: index (), menu vector (ARRAY), number (INT) of recommendations requested
    OUTPUT: dataframe/series(?) of recommended recipes
    '''
    sims = index[model[menu_vector]] ## convert BOW to Tfidf
    rec_indices = np.argsort(sims)[:-num:-1] # gets top n
    return df.loc[rec_indices, 'title'], sims[rec_indices]

## THIS WORKS!!!!!!!!!!!!!!!!1!!1!!!

if __name__ == '__main__':

    restaurant_name = sys.argv[1]
    ## Connect to database
    conn = MongoClient()
    db = conn.project

    ## Get Recipe data
    cursor = db.recipes.find({}, {'title': 1, 'ingredients': 1, '_id' : 0})
    data = pd.DataFrame(list(cursor))
    data['ingredients'] = data['ingredients'].apply(lambda x: " ".join(x))
    documents = data['ingredients'].values
    #dictionary, corpus = create_dictionary(documents)
    #print documents[:5]

    recipe_array = clean_text(documents)
    #print texts_array[:2]

    # dict_size = 0
    # for i in dictionary.iterkeys():
    #     dict_size +=1
    # print dict_size
    #model = models.TfidfModel()
    #tfidf, index = create_model(corpus, dict_size)
    ## Need to do the above just once - when all recipes collected, can write to disk ##
    ## (see gensim docs)

    ## playing with Word2Vec
    model = use_word2vec('/home/ec2-user/mnt/GoogleNews-vectors-negative300.bin.gz') #loads but slowwwly
    print model[recipe_array[1]]
    #index = similarities.Similarity(model[recipe_array])
    print 'Done!'
    ## save index to disk for speed?

    #menu_string = get_restaurant_menu(restaurant_name, db)
    # recs, scores = get_recommendations(index, menu_vec, 5, tfidf, data)
    # print [result for result in zip(recs, scores)]
    #print recs
    #menu_array = clean_text([menu_string])
    #sims = index[model[menu_vector]] ## convert BOW to word vectors
    #rec_indices = np.argsort(sims)[:-5:-1] # gets top 5
    #print data.loc[rec_indices, 'title'], sims[rec_indices]
