import pandas as pd
import numpy as np
import sys
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords


## need to create a shared stopwords set, dictionary, index, model --> maybe instance variables of the class?
class MyRecommender():
    '''
    A class that will build take in text documents, build a dictionary and index, and
    return recommendations from that index upon new input.
    '''
    def __init__(self):
        self.stopset = set(stopwords.words('english'))
        self.stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
        self.dictionary = None
        self.dictionary_len = 0
        self.index = None
        self.corpus = None
        self.df = None ## Need df to get recipe ids back?

    def _prepare_documents(self, db):
        cursor = db.recipes.find({}, {'rec_id': 1, 'title' : 1, 'ingredients': 1, '_id' : 0})
        self.df = pd.DataFrame(list(cursor))
        self.df['ingredients'] = self.df['ingredients'].apply(lambda x: " ".join(x))
        documents = self.df['ingredients'].values
        return documents


    def _create_dictionary(self, db):
        '''
        INPUT: text documents (array of strings)
        OUTPUT: gensim dictionary object, corpus
        '''
        ## Vectorize and store recipe text
        documents = self._prepare_documents(db)
        texts = [[word for word in document.lower().split() if word not in self.stopset] for document in documents]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts] ## convert to BOW

        for i in self.dictionary.iterkeys():
            self.dictionary_len +=1

    def _create_model(self): ## how to make extendable to other models?
        '''
        INPUT: Model class, corpus (ARRAY)
        OUTPUT: trained model, index (for similarity scoring)
        '''
        ## Apply model
        self.model = models.TfidfModel(self.corpus)
        ## prepare for similarity queries - unlikely to be memory constrained (< 100K docs) so won't write to disk
        self.index = similarities.SparseMatrixSimilarity(self.model[self.corpus], num_features = self.dictionary_len) # num_features is len of dictionary
        #return model, index

    def _vectorize_restaurant_menu(self, name, db):
        '''
        INPUT: restaurant name (STRING), database connection, stopwords (LIST), dictionary object
        OUTPUT: menu vector (ARRAY)
        '''
        ## Get 1 restaurant menu
        cursor = db.restaurants.find_one({'name' : name})
        menu = cursor['menu']

        ## Vectorize and prep menu text
        ## Broken if menu field is empty! Only 16 of these though.
        menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
        menu_string = " ".join(menu_list)

        menu_tokens = [word for word in menu_string.lower().split() if word not in self.stopset]
        menu_vector = self.dictionary.doc2bow(menu_tokens)
        return menu_vector

    def fit(self, db):
        '''
        INPUT:
        OUTPUT:
        '''
        self._create_dictionary(db)
        self._create_model()


    def get_recommendations(self, name, db, num):
        '''
        INPUT: index (), menu vector (ARRAY), number (INT) of recommendations requested
        OUTPUT: dataframe/series(?) of recommended recipes
        '''
        menu_vector = self._vectorize_restaurant_menu(name, db)
        sims = self.index[self.model[menu_vector]] ## convert BOW to Tfidf
        rec_indices = np.argsort(sims)[:-num:-1] # gets top n
        return self.df.loc[rec_indices, 'title'], sims[rec_indices]

## THIS WORKS!!!!!!!!!!!!!!!!1!!1!!!

if __name__ == '__main__':

    restaurant_name = sys.argv[1]
    ## Connect to database
    conn = MongoClient()
    db = conn.project

    recommender = MyRecommender()
    recommender.fit(db)
    ## Need to do the above just once - when all recipes collected, can write to disk ##
    ## (see gensim docs)

    recs, scores = recommender.get_recommendations(restaurant_name, db, 5)
    print [result for result in zip(recs, scores)]
    #print recs
