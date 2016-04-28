from flask import Flask, render_template, request, url_for
from pymongo import MongoClient
from recommender import MyRecommender

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title = 'Recipe Recommender')

@app.route('/how-it-works')
def methodology():
    return render_template('howto_index.html')



@app.route('/submit', methods = ['GET', 'POST'])
def make_submission():
    ## enter a restaurant name from db to start
    return render_template('submit.html', title = 'Recipe Recommender- Submission')


@app.route('/results', methods = ['GET', 'POST'])
def result():
    ## compute recommendations
    restaurant_name = str(request.form['input_text'])
    cursor = db.restaurants.find({'name' : restaurant_name})
    if cursor.count() == 0:
        return render_template('notfound.html')
    recs, scores = recommender.get_recommendations(restaurant_name, db, 6)
    recipes = []
    for recc in recs:
        recipe = db.recipes.find({'rec_id' : recc}, {'_id' :0, 'ingredients':0})
        for r in recipe:
            recipes.append(r)
    #print recipes

    ## get full info for recipes from db
    return render_template('results.html', recs = recipes) ## recs is list of recommendations

if __name__ == '__main__':
    ## connect to db
    conn = MongoClient()
    db = conn.project

    ## grab indexed recipe vectors instead of instantiating?

    recommender = MyRecommender()
    recommender.fit(db) ## OR do I pickle this?? or write to disk with gensim??

    ## run app
    app.run(host = '0.0.0.0', port = 8000, debug = True)
