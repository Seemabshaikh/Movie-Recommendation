from flask import Flask, render_template
import pickle
import pandas as pd

from gensim.models import KeyedVectors

# load the model
model = KeyedVectors.load_word2vec_format('Word2vec.txt', binary=False)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        movie_title = request.form['title']
        try:
            recommendations = model.wv.most_similar(movie_title)[:10]
            recommendations = [r[0] for r in recommendations]
        except KeyError:
            recommendations = []
        return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)
    else:
        return render_template('recommendations.html')

if __name__ == '__main__':
    app.run(debug=True)
