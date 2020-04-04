"""
Flask app for Movie Recommender page
"""

import logging
from flask import Flask
from flask import render_template, request
import recommender

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

APP = Flask('movie recommender')
MOVIES, RATINGS, RATING_MATRIX, MODEL = recommender.prep_for_recommendations(
    fill_value='median'
)


@APP.route('/')
def hello():
    """
    Main Page
    Includes dialog boxes for user to input movie choices
    """
    return render_template('main_page.html')


@APP.route('/recommend')
def run_recommender():
    """
    Recommender page
    Presents user with recommedations based on inputs
    Also displays which films in the db the user's inputs were matched to
    """
    n_recommendations = 5
    selections = [request.args[f'movie_{n+1}'] for n in range(5)
                  if request.args[f'movie_{n+1}'] != ""]
    n_selections = len(selections)

    context = dict(title="Your recommendations:")

    if n_selections > 0:
        result = recommender.get_recommendations(
            selections, RATING_MATRIX, MODEL,
            n_recommendations
        )
        result, interpreted_choices = result
        interpretation_message = "Based on our interpretation of your choices"
    else:
        result = ["Parasite"]
        interpreted_choices = []
        interpretation_message = ""

    context['movies'] = result
    context['interpreted_choices'] = interpreted_choices
    context['interpretation_message'] = interpretation_message
    context['user_input'] = selections
    context['inputs_to_interpretations'] = list(zip(selections,
                                                    interpreted_choices))

    return render_template('recommendation.html', **context)


@APP.route('/recommendation_frequencies')
def show_rec_freqs():
    """
    Display an frequency distribution of the recommended movies
    """
    return render_template('recommendation_frequencies.html')
