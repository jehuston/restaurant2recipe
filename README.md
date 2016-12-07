# Restaurant2Recipe Recommender

Everyone's been there-- it's time to make dinner and you have no idea what you want. Restaurant2Recipe was built to help the next time you find yourself in this exact situation.

<img src = 'https://images.unsplash.com/photo-1424847651672-bf20a4b0982b?crop=entropy&fit=crop&fm=jpg&h=800&ixjsv=2.1.0&ixlib=rb-0.3.5&q=80&w=1375'>

Restaurant2Recipe is a content-based recommendation system that uses the menu of a favorite SF restaurant to suggest recipes you might like. Recommendations are generated by first analyzing the text of the restaurant menus and recipe ingredient lists. The tool then determines the most similar recipes from the Restaurant2Recipe database, and returns results including pictures and links to the original recipes.

<img src = '/my_app/static/images/home_page.png'>

### Process
I first collected menu data from restaurants across San Francisco as well as recipe information from across the web. The recipe ingredient lists became the corpus for the recommender system, while the restaurant menu items and descriptions are stored and analyzed on an ad-hoc basis. This will allow me to extend the search utility in the future to include taking in new menus not already in my database.

The first step in almost any natural language processing (NLP) task is to translate your text into a numerical form, a process known as word embedding. I initially intended to use word2vec as a word embedding technique. I tested the system using both pretrained word2vec word vectors and tf-idf (term frequency-inverse document frequency) as word embedding techniques. Ultimately, I found better and more stable results in this instance using tf-idf (more on this below).  

Once the text data has been vectorized, I next calculate cosine similarity for the queried restaurant against all recipe document vectors. I return the top most similar recipes.

<img src = '/my_app/static/images/results_page.png'>

### Evaluation
One limitation of recommender systems is they are notoriously difficult to validate. As there is no target to predict, I can't calculate an accuracy score. In production, one could perhaps A/B test a recommender system (or several) and see which one resulted in more click-throughs, purchases, etc. Absent that information, I'm left with an admittedly anecdotal smell test (or perhaps a taste test). Basically, enter a Mexican restaurant-- do I see tacos, enchiladas, or nachos as suggested recipes? How about an Italian restaurant, or a fish restaurant? There's a pierogi restaurant in SF-- try entering 'Stuffed' into the recommender and note the plethora of potato recipes you receive.

I was initially surprised to receive what appeared to be better results with tf-idf compared to word2vec, because word2vec is a more sophisticated and seems to be the hot new technique in NLP. I have a few theories why this might be the case.
* Due to data limitations-- you need a <em>lot</em> of data to train word2vec models-- I relied on pre-trained word vectors. While the word vectors were trained on GoogleNews and presumably quite extensive, they were still inevitably missing some words, perhaps some very specialized food-related words. This could affect the ability of the resulting document vectors to well represent the documents they came from.

* While I could easily transform a word into a vector, what I really needed was a vector representation of a document. There are multiple ways to do this, and there's even a doc2vec class in gensim to accomodate this need. The simplest solution is to summ up the component word vectors in each document to create a document vector. I also tried averaging the word vectors into the document vector but the resulting recommendations did not appear to be significantly different. While aggregating word vectors is mathematically valid, I suspect the aggregation might have been skewing the results. 

* At its heart, this is a cross-domain recommender, as I am trying to go from restaurants to recipes instead of movies to movies or books to books as many recommendation systems do. While I'm assuming that restaurant menus and recipe ingredient lists are similar enough to base recommendations on, it's possible they are less similar in the type of information they contain and thus less helpful in content-based recommendation. It would certainly be interesting, given different data, to try different approaches such as collaborative filtering.

### Next Steps
Future features I'd like to add include:
* Test alternate word embedding techniques
* Experiment with dimensionality reduction
* Upgrade input functionality to allow for novel menu entry
* Add more recipe data sources

### Technology
Restaurant2Recipe was created in Python, and uses the following libraries:
* gensim
* nltk
* BeautifulSoup
* requests
* pandas
* numpy

The web app uses the Flask framework and is supported by a mongoDB database, hosted on an AWS EC2 instance.

### Credits
Restaurant2Recipe is powered by the Food2Fork recipe API and menu data from Yelp.
This project was completed to fulfill the capstone requirement for Galvanize Data Science Immersive in April 2016.
