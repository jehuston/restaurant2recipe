import pandas as pd
import numpy as np
import sys
import string
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


## need to create a shared stopwords set, dictionary, index, model --> maybe instance variables of the class?

def create_dictionary(documents):
    '''
    INPUT: text documents (array of strings)
    OUTPUT: gensim dictionary object, corpus
    '''
    ## Vectorize and store recipe text
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents] ## convert to BOW
    return dictionary, corpus # could write to disk instead?

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
    #port = PorterStemmer()
    # texts = [[wnl.lemmatize(word.strip(string.punctuation)) for word in\
    #             document.lower().replace("/", "").split()\
    #             if word not in stopset if not any(c.isdigit() for c in word)]\
    #         for document in documents]    ## this is hideous
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
    index = similarities.SparseMatrixSimilarity('models/recipe_index', doc_vectors, num_features = 300)
    return index

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

    # Get Recipe data
    cursor = db.recipes.find({'title': { '$ne':'All Recipes' } }, {'title': 1, 'ingredients': 1, '_id' : 0})
    data = pd.DataFrame(list(cursor))
    data['ingredients'] = data['ingredients'].apply(lambda x: " ".join(x))
    documents = data['ingredients'].values

    recipe_array = clean_text(documents)
    #print recipe_array
    #
    # # ## playing with Word2Vec
    ## AWS location
    #model = models.Word2Vec.load_word2vec_format('/home/ec2-user/vectors/GoogleNews-vectors-negative300.bin.gz', binary=True)
    ## local location
    model = models.Word2Vec.load_word2vec_format('word-vectors/GoogleNews-vectors-negative300.bin.gz', binary=True)
    model.init_sims(replace=True)
    #print model.similarity('potato', 'tomato')

    # first time:
    recipe_vectors = create_doc_vectors(model, recipe_array)
    recipe_index = create_index(recipe_vectors)

    ## subsequent:
    # recipe_index = load('models/recipe_index')

    ## translate menu into doc_vector
    # menu_string = get_restaurant_menu(restaurant_name, db)
    # menu_array = clean_text([menu_string])
    # #print menu_array[0]
    # menu_vector = create_doc_vectors(model, menu_array)[0]
    # #print menu_vector
    # sims = recipe_index[menu_vector]
    # rec_indices = np.argsort(sims)[:-6:-1] # gets top 5
    # print data.loc[rec_indices, 'title'], sims[rec_indices]

    # recs, scores = get_recommendations(index, menu_vec, 5, tfidf, data)
    # print [result for result in zip(recs, scores)]
    # print recs
