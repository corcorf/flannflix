"""
Create a chart showing movie recommendation frequencies and save as
an Altair JSON for displaying on a webpage
"""
import pandas as pd
import altair as alt
from sql_tables import connect_to_db, read_tables
from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB


def create_frequency_chart(engine,
                           input_filename="test_results_12032020_2140.csv"):
    """
    Create an Altair chart showing movie recommendation frequencies
    __________
    Parameters:
        engine (sqlalchemy connection engine): link to postgres database
                                               containing the movie ratings
        input_filename (string): path to a csv with columns 'iteration',
                                 'user_selections', 'interpretations'
                                 and 'recommendations', like that created by
                                 the 'results_tester' script.

    Returns:
        Altair interactive chart object
    """
    results = pd.read_csv(input_filename, index_col=0)
    total_recs = results['recommendations'].value_counts().sum()
    data = read_tables(engine,
                       table_names=['ratings', 'movies'])

    ratings = data['ratings']
    movies = data['movies']

    mean_ratings = ratings.merge(movies, on="movieId").groupby("title")\
                          .agg({"rating": "mean", "userId": "count",
                                "movieId": "first", "genres": "first"})
    mean_ratings = mean_ratings.rename(columns={"rating": "Average rating",
                                                "userId": "# ratings",
                                                "genres": "Genres"})

    source = results.groupby('recommendations')['iteration'].count()\
                    .sort_values(ascending=False).reset_index()
    source = source.reset_index().rename(columns={'index': '#',
                                                  'iteration': 'Count',
                                                  'recommendations': 'Movie'})
    source['Frequency (%)'] = source['Count'] / total_recs * 100
    source = source.join(mean_ratings, on="Movie")
    source["Year"] = source['Movie'].str.extract(r"((?<=\()[0-9]+(?=\)\s*$))")\
                                        .astype(int)
    source["Decade"] = source['Year'].round(-1)

    chart = alt.Chart(source, background="#ffffff66")\
               .mark_circle(size=60)\
               .encode(
                   x='#',
                   y='Frequency (%)',
                   color=alt.Color('Decade:O',
                                   scale=alt.Scale(scheme='darkred')),
                   tooltip=['Movie', "Frequency (%)", "Average rating",
                            "# ratings", "Genres"]
               ).interactive()
    return chart


if __name__ == "__main__":
    CONN_STRING = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    ENGINE, SESSION = connect_to_db(CONN_STRING, verbose=False)
    CHART = create_frequency_chart(ENGINE)
    CHART_FILENAME = "static/recommendation_frequencies.json"
    CHART.save(CHART_FILENAME)
