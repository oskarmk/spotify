import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# define path of data
path = '/home/oscar/newsletter/spotify/data2ndpost100'

# define the list of columns that we care about
cols_of_interest = ['danceability',
                    'energy',
                    'loudness',
                    'speechiness',
                    'acousticness',
                    'instrumentalness',
                    'liveness',
                    'valence',
                    'tempo',
                    'duration_ms',
                    'song_title',
                    'popularity',
                    'decade']

# read individual features
features_50 = pd.read_csv(path + '/1950/track_features_1950s.csv', usecols=cols_of_interest)
features_60 = pd.read_csv(path + '/1960/track_features_1960s.csv', usecols=cols_of_interest)
features_70 = pd.read_csv(path + '/1970/track_features_1970s.csv', usecols=cols_of_interest)
features_80 = pd.read_csv(path + '/1980/track_features_1980s.csv', usecols=cols_of_interest)
features_90 = pd.read_csv(path + '/1990/track_features_1990s.csv', usecols=cols_of_interest)
features_00 = pd.read_csv(path + '/2000/track_features_2000s.csv', usecols=cols_of_interest)
features_10 = pd.read_csv(path + '/2010/track_features_2010s.csv', usecols=cols_of_interest)
features_20 = pd.read_csv(path + '/2020/track_features_2020s.csv', usecols=cols_of_interest)

# concatenate features
features = pd.concat([features_50,
                      features_60,
                      features_70,
                      features_80,
                      features_90,
                      features_00,
                      features_10,
                      features_20])

features['duration_ms'] = features['duration_ms'] / 60000

features = features.rename(columns={'energy' : 'Energy',
                                    'loudness': 'Loudness [dB]',
                                    'duration_ms': 'Duration [Min]'})

####
# Plot
# fig = px.scatter(features,
#                  x= 'valence',
#                  y ='tempo',
#                  color='decade',
#                  size='popularity',
#                  marginal_x='histogram')
#                  #marginal_y='histogram',
#                  #facet_col='Country') # pa ver tol rollo de marginal https://plotly.com/python/marginal-plots/


# fig.update_layout(
#     #plot_bgcolor='rgba(0,0,0,0)',

#     font = dict(
#         size = 30
#     ),

#     title = dict(
#         text= '',
#         #x = 0.5,
#         #y = 0.8
#     ),
#     xaxis = dict(
#         #title = 'Energy',
#         #gridcolor='rgba(0,0,0,0)',
#         #showline=True,
#         #linewidth=1,
#         #linecolor='rgba(0,0,0,0.2)',
#         #tickangle = 45
#     ),
#     #xaxis_title=r'$Grand Slam$',
#     yaxis = dict(
#         #title = '',
#         #tickmode = 'array',
#         #tickvals= list(range(0, 101, 10)),
#         #range = [0, 100],
#         #gridcolor='rgba(0,0,0,0.2)'
#     )
# )

# fig.show()
####

### Violin Plots

# fig = px.violin(features, y = 'valence', color='decade')
# fig.show()

###

### Histogram

# fig = px.histogram(features, x="valence", color="Country")
# fig.show()

###

### Correlation Matrix

# features_corr = features.corr()

# fig = px.imshow(features_corr, color_continuous_scale='RdBu_r', text_auto=True)

# fig.update_layout(
#     #plot_bgcolor='rgba(0,0,0,0)',

#     font = dict(
#         size = 20
#     ),

#     title = dict(
#         text= '',
#         #x = 0.5,
#         #y = 0.8
#     ),
#     xaxis = dict(
#         #title = 'Energy',
#         #gridcolor='rgba(0,0,0,0)',
#         #showline=True,
#         #linewidth=1,
#         #linecolor='rgba(0,0,0,0.2)',
#         #tickangle = 45
#     ),
#     #xaxis_title=r'$Grand Slam$',
#     yaxis = dict(
#         #title = '',
#         #tickmode = 'array',
#         #tickvals= list(range(0, 101, 10)),
#         #range = [0, 100],
#         #gridcolor='rgba(0,0,0,0.2)'
#     )
# )
# fig.show()

###

### K-Means Analysis:
# from sklearn.cluster import KMeans

# df_analysis = features[['danceability', 'valence']]
# X = np.array(df_analysis.values.tolist())

# centroids = [[0.25, 0.25],
#              [0.5, 0.50],
#              [0.75, 0.75]]

# kmeans = KMeans(n_clusters=3, init=centroids, random_state=0).fit(X)
# # compute cluster centers and predict cluster index for each sample
# y_km = kmeans.fit_predict(X)

# centers = kmeans.cluster_centers_
# print(X[y_km == 1, 0])
# print(X[y_km == 1, 1])

# import plotly.graph_objs as go

# fig = go.Figure()

# # plot cluster centers
# fig.add_trace(go.Scatter(
#     x = centers.T[0],
#     y = centers.T[1],
#     name = 'cluster_centers',
#     mode = 'markers',
#     marker = dict(
#         size=20
#     )
# ))

# # cluster 1
# fig.add_trace(go.Scatter(
#     x = X[y_km == 0, 0],
#     y = X[y_km == 0, 1],
#     name = 'cluster1',
#     mode = 'markers',
#     marker = dict(
#         size=10
#     )
# ))

# # cluster 2
# fig.add_trace(go.Scatter(
#     x = X[y_km == 1, 0],
#     y = X[y_km == 1, 1],
#     name = 'cluster2',
#     mode = 'markers',
#     marker = dict(
#         size=10
#     )
# ))

# # cluster 3
# fig.add_trace(go.Scatter(
#     x = X[y_km == 2, 0],
#     y = X[y_km == 2, 1],
#     name = 'cluster3',
#     mode = 'markers',
#     marker = dict(
#         size=10
#     )
# ))

# fig.show()
###

### Affinity propagation
# from sklearn.cluster import AffinityPropagation

# df_analysis = features[['danceability', 'valence']]
# X = np.array(df_analysis.values.tolist())

# clustering = AffinityPropagation(random_state=0).fit(X)
# cluster_centers_indices = clustering.cluster_centers_indices_
# labels = clustering.labels_

# n_clusters_ = len(cluster_centers_indices)

# import matplotlib.pyplot as plt

# plt.close("all")
# plt.figure(1)
# plt.clf()

# colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))

# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.scatter(
#         X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
#     )
#     plt.scatter(
#         cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
#     )
#     for x in X[class_members]:
#         plt.plot(
#             [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
#         )

# plt.title("Estimated number of clusters: %d" % n_clusters_)
# plt.show()



### QUITAR TODOS LOS REPETIDOS SONGS, PORQUE SI ESTAN EN TODOS LADOS, NO TIENE SENTIDO

# Mean analyisis:
features_mean = pd.concat([pd.DataFrame(features_50.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_60.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_70.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_80.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_90.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_00.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_10.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_20.mean(axis=0, numeric_only=True)).transpose()])

features_mean['decade'] = ['50s', '60s', '70s', '80s', '90s', '2000s', '2010s', '2020s']


df = features_mean
fig = px.scatter(df, x='valence', y='danceability', color='decade',
                 size = 'popularity', text = 'decade')
fig.update_traces(textposition="top center")

fig.update_layout(
    font = dict(
        size = 30
    ),

    xaxis = dict(
        title = 'Valence',
        range = [0, 1]
    ),

    yaxis = dict(
        title = 'Danceability',
        range = [0,1]
    )
)

fig.show()

### Side bar chart

features_mean_2 = features_mean.sort_values('acousticness', ascending=True)

fig = go.Figure(go.Bar(
            x=features_mean_2['acousticness'],
            y=features_mean_2['Country'],
            #color=features_mean_2['Country'],
            orientation='h'))

fig.update_layout(
    font = dict(
        size = 30
    ),

    xaxis = dict(
        title = 'Acousticness',
        #range = [0, 1]
    ),

    yaxis = dict(
        title = '',
        #range = [0,1]
    )
)

fig.show()

###

### World map

# fig = px.choropleth(features_mean, locations="Country",
#                     color="valence",
#                     locationmode = 'country names',
#                     #hover_name="location",
#                     #animation_frame="date",
#                     title = "Covid Cases plotted using Plotly",
#                     color_continuous_scale=px.colors.sequential.PuRd)

# fig.show()

###

### 3-D Plot

# df = features_select
# fig = px.scatter_3d(df, x='Energy', y='Loudness [dB]', z='valence',
#               color='Country')
# fig.show()

###