import pandas as pd
import numpy as np
import sys
import string
from pymongo import MongoClient
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class MyRecommender():
    '''
    A class that will build take in text documents, build a dictionary and index, and
    return recommendations from that index upon new input.
    '''
    def __init__(self, model):
        self.model = model
        self.stopset = set(stopwords.words('english'))
        self.stopset.update(['description', 'available']) ## add some words that appear a lot in menu data
        self.dictionary = None
        self.dictionary_len = 0
        self.index = None
        self.corpus = None
        self.df = None

    def _prepare_documents(self, db):
        '''
        INPUT: database connection
        OUTPUT: array of strings

        Given database connection, collect all recipe documents, join ingredients lists
        into strings, and return as array.
        '''
        cursor = db.recipes.find({}, {'rec_id': 1, 'title' : 1, 'ingredients': 1, '_id' : 0})
        self.df = pd.DataFrame(list(cursor))
        self.df['ingredients'] = self.df['ingredients'].apply(lambda x: " ".join(x))
        documents = self.df['ingredients'].values
        return documents

    def _clean_text(self, documents):
        '''
        INPUT: array of strings
        OUTPUT: array of lists (ok?)

        Given array of strings (recipes or restaurant menus), tokenize and return as array.
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

    def _create_doc_vectors(self, text_array): ## word2vec only
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
                    doc_vector += self.model[text_array[i][j]]
                except KeyError: #if word not in word vectors, skip it
                    unfound_words += 1
                    continue
            doc_vectors.append(doc_vector)
            #doc_vectors.append(np.divide(doc_vector, text_array.shape[0])) #to average?
        print "Num words not found: ", unfound_words
        return np.array(doc_vectors)

    def _vectorize_restaurant_menu(self, name, db):
        '''
        INPUT: restaurant name (STRING), database connection
        OUTPUT: menu vector (ARRAY)

        Given restaurant name that exists in database, return menu and vectorize to
        prepare for similarity query. -- needs if word2vec / else tfidf
        '''
        ## Get 1 restaurant menu
        cursor = db.restaurants.find_one({'name' : name})
        menu = cursor['menu']

        ## Vectorize and prep menu text
        ## Broken if menu field is empty! Only 16 of these though.
        menu_list = [" ".join(i) for i in zip(menu['items'], menu['descriptions'])]
        menu_string = " ".join(menu_list)

        menu_tokens = self._clean_text([menu_string])[0]

        if self.model.__init__.im_class == models.tfidfmodel.TfidfModel:
            menu_vector = self.dictionary.doc2bow(menu_tokens)
            menu_vector = self.model[menu_vector]
        else:
            menu_vector = self._create_doc_vectors(menu_tokens)[0]
        return menu_vector

    def fit(self, db):
        '''
        INPUT: connection to database with recipes, restaurants data
        OUTPUT: fit model, index

        Creates a dictionary and model for recommender system. Given database connection,
        find all recipe ingredient lists, vectorize, build corpus and dictionary,
        fit model and create index.
        '''
        documents = self._prepare_documents(db)
        texts = self._clean_text(documents)

        if self.model.__init__.im_class == models.tfidfmodel.TfidfModel:
            ## Vectorize and store recipe text
            self.dictionary = corpora.Dictionary(texts)
            self.corpus = [self.dictionary.doc2bow(text) for text in texts] ## convert to BOW

            for i in self.dictionary.iterkeys():
                self.dictionary_len +=1

            self.model = self.model(self.corpus)
            ## prepare for similarity queries
            self.index = similarities.SparseMatrixSimilarity(self.model[self.corpus], num_features = self.dictionary_len)

        else: # word2vec
            self.model = models.Word2Vec.load('/mnt/word2vec/words', mmap='r')
            doc_vectors = self._create_doc_vectors(texts)
            self.index = similarities.Similarity('/mnt/word2vec/index', doc_vectors, num_features = 300)


    def get_recommendations(self, name, db, num):
        '''
        INPUT: index (), menu vector (ARRAY), number (INT) of recommendations requested
        OUTPUT: dataframe/series(?) of recommended recipes

        Returns top n recommended recipes based on cosine similiarity to restaurant menu.
        '''
        menu_vector = self._vectorize_restaurant_menu(name, db)
        sims = self.index[menu_vector]
        rec_indices = np.argsort(sims)[:-(num+1):-1] # gets top n
        return self.df.loc[rec_indices, 'rec_id'], sims[rec_indices]


if __name__ == '__main__':

    restaurant_name = sys.argv[1]
    ## Connect to database
    conn = MongoClient()
    db = conn.project

    #model = models.TfidfModel
    model = models.Word2Vec
    recommender = MyRecommender(model)
    recommender.fit(db)


    #recs, scores = recommender.get_recommendations(restaurant_name, db, 5)
    #print [result for result in zip(recs, scores)]
    #print recs
