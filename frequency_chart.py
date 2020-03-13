import pandas as pd
from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
from sql_tables import connect_to_db, read_tables, Rating, Movie, Tag, Link

conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
engine, session = connect_to_db(conn_string, verbose=False)

results = pd.read_csv("test_results_12032020_2140.csv", index_col=0)
total_recs = results['recommendations'].value_counts().sum()
data = read_tables(engine, table_names = ['ratings', 'tags', 'links', 'movies'])

ratings = data['ratings']
movies = data['movies']
tags = data['tags']
links = data['links']

mean_ratings = ratings.merge(movies, on="movieId").groupby("title").agg({"rating":"mean", "userId":"count",
                                                          "movieId":"first", "genres":"first"})
mean_ratings = mean_ratings.rename(columns={"rating":"Average rating",
                                            "userId":"# ratings", "genres":"Genres"})

source = results.groupby('recommendations')['iteration'].count().sort_values(ascending=False).reset_index()
source = source.reset_index().rename(columns={'index':'#', 'iteration':'Count','recommendations':'Movie'})
source['Frequency (%)'] = source['Count'] / total_recs
source = source.join(mean_ratings, on="Movie")

# source.merge(links, on="movieId")

import altair as alt

chart = alt.Chart(source).mark_circle(size=60).encode(
            x='#',
            y='Frequency (%)',
        #     color='Origin',
            tooltip=['Movie', "Frequency (%)", "Average rating", "# ratings", "Genres"]
        ).interactive()

chart.save("recommendation_frequencies.json")
