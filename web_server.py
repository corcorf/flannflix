"""
export FLASK_APP=web_Server.py
export FLASK_DEBUG=True
flask run
"""

from flask import Flask
from flask import render_template, request
import recommender

app = Flask('movie recommender')


@app.route('/') # Decorator (adds functionality to a function)
def hello():
    return render_template('main_page.html')


@app.route('/recommend')
def run_recommender():
    n_recommendations = 5
    movies = [request.args[f'movie_{n+1}'] for n in range(5) if request.args[f'movie_{n+1}'] != ""]
    n_choices = len(movies)
    # try:
    #     n = int(request.args['movie_1'])
    # except ValueError:
    #     n = 3
    context = dict(title="Your recommendations:")
    if n_choices > 0:
        result = recommender.get_recommendations(movies, n_recommendations)
    else:
        result = ["Parasite"]

    context['movies'] = result
    return render_template('recommendation.html',
                            **context
                        )
