import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# define path of data
path = '/home/oscar/newsletter/spotify/data'

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
                    'popularity']

# read individual features
features_BR = pd.read_csv(path + '/BR/track_features_BR.csv', usecols=cols_of_interest)
features_CA = pd.read_csv(path + '/CA/track_features_CA.csv', usecols=cols_of_interest)
features_CN = pd.read_csv(path + '/CN/track_features_CN.csv', usecols=cols_of_interest)
features_DE = pd.read_csv(path + '/DE/track_features_DE.csv', usecols=cols_of_interest)
features_ES = pd.read_csv(path + '/ES/track_features_ES.csv', usecols=cols_of_interest)
features_FI = pd.read_csv(path + '/FI/track_features_FI.csv', usecols=cols_of_interest)
features_FR = pd.read_csv(path + '/FR/track_features_FR.csv', usecols=cols_of_interest)
features_IN = pd.read_csv(path + '/IN/track_features_IN.csv', usecols=cols_of_interest)
features_IT = pd.read_csv(path + '/IT/track_features_IT.csv', usecols=cols_of_interest)
features_JP = pd.read_csv(path + '/JP/track_features_JP.csv', usecols=cols_of_interest)
features_KE = pd.read_csv(path + '/KE/track_features_KE.csv', usecols=cols_of_interest)
features_KO = pd.read_csv(path + '/KO/track_features_KO.csv', usecols=cols_of_interest)
features_MX = pd.read_csv(path + '/MX/track_features_MX.csv', usecols=cols_of_interest)
features_PH = pd.read_csv(path + '/PH/track_features_PH.csv', usecols=cols_of_interest)
features_PT = pd.read_csv(path + '/PT/track_features_PT.csv', usecols=cols_of_interest)
features_RU = pd.read_csv(path + '/RU/track_features_RU.csv', usecols=cols_of_interest)
features_SA = pd.read_csv(path + '/SA/track_features_SA.csv', usecols=cols_of_interest)
features_TU = pd.read_csv(path + '/TU/track_features_TU.csv', usecols=cols_of_interest)
features_UK = pd.read_csv(path + '/UK/track_features_UK.csv', usecols=cols_of_interest)
features_US = pd.read_csv(path + '/US/track_features_US.csv', usecols=cols_of_interest)

# concatenate features
features = pd.concat([features_BR,
                      features_CA,
                      features_CN,
                      features_DE,
                      features_ES,
                      features_FI,
                      features_FR,
                      features_IN,
                      features_IT,
                      features_JP,
                      features_KO,
                      features_KE,
                      features_MX,
                      features_PH,
                      features_PT,
                      features_RU,
                      features_SA,
                      features_TU,
                      features_UK,
                      features_US])

country_list = (['Brazil'] * 100 + ['Canada'] * 100 + ['China'] * 100 + ['Germany'] * 100 +
                ['Spain'] * 100 + ['Finland'] * 100 + ['France'] * 100 + ['Japan'] * 100 +
                ['India'] * 100 + ['Italy'] * 100 + ['Kenia'] * 100 + ['Korea'] * 100 +
                ['Mexico'] * 100 + ['Philippines'] * 100 + ['Portugal'] * 100 + ['Russia'] * 100 +
                ['South Africa'] * 100 + ['Turkey'] * 100 + ['UK'] * 100 + ['USA'] * 100)

features['Country'] = country_list
features['duration_ms'] = features['duration_ms'] / 60000

features = features.rename(columns={'energy' : 'Energy',
                                    'loudness': 'Loudness [dB]',
                                    'duration_ms': 'Duration [Min]'})

features_select = features[features['Country'].isin(['China', 'Kenia', 'Germany', 'Japan', 'Mexico'])]

####
# Plot
# fig = px.scatter(features_select,
#                  x= 'valence',
#                  y ='tempo',
#                  color='Country',
#                  size='popularity',
#                  marginal_x='histogram',
#                  #marginal_y='histogram',
#                  facet_col='Country') # pa ver tol rollo de marginal https://plotly.com/python/marginal-plots/


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

# fig = px.violin(features_select, y = 'danceability', color='Country')
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
features_mean = pd.concat([pd.DataFrame(features_BR.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_CA.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_CN.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_DE.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_ES.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_FI.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_FR.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_IN.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_IT.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_JP.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_KO.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_KE.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_MX.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_PH.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_PT.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_RU.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_SA.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_TU.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_UK.mean(axis=0, numeric_only=True)).transpose(),
                           pd.DataFrame(features_US.mean(axis=0, numeric_only=True)).transpose()])

features_mean['Country'] = ['Brazil','Canada', 'China','Germany', 'Spain', 'Finland', 'France',
                            'India', 'Italy', 'Japan', 'Korea', 'Kenia', 'Mexico', 'Portugal',
                            'Philippines', 'Russia', 'South Africa', 'Turkey', 'UK', 'US']


# df = features_mean
# fig = px.scatter(df, x='valence', y='danceability', color='Country',
#                  size = 'popularity', text = 'Country')
# fig.update_traces(textposition="top center")

# fig.update_layout(
#     font = dict(
#         size = 30
#     ),

#     xaxis = dict(
#         title = 'Valence',
#         range = [0, 1]
#     ),

#     yaxis = dict(
#         title = 'Danceability',
#         range = [0,1]
#     )
# )

# fig.show()

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