# import modules
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
from lyricsgenius import Genius
import os.path


def get_country_data(playlist_link: str):

    '''This function retrieves the lyrics and song features of the songs included
       in the desired playlist.
       Input: Spotify link of the playlist
       Output: .txt files with song lyrics
               .csv file with the song features'''

    ### Setup:
    # These are extracted from: https://developer.spotify.com/dashboard/applications/ed4d5bfd956442c2a1c093d3a3942aa2
    user_id = '4ag73vuzx8ldp7by0442uic3g' # my personal account user_id --> lo que utilizo pa meterme
    client_id = 'ed4d5bfd956442c2a1c093d3a3942aa2'
    client_secret = 'cdcc72470e684ec19527cb283f01baa5'

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    # Genius Token
    genius = Genius('QRLI3NSZJwB8lD22xq3hEN923Dw_U7Uk1BoYAMfnQgq5ejtGBhZLAxbBeGmKqag1',
                    timeout=20)

    tracks = sp.playlist_tracks(playlist_link)

    saving_csv_name = 'track_features_SA.csv'

    # Initialize pandas dataframe to populate with track features
    track_feature_pd = pd.DataFrame(columns=['danceability',
                                            'energy',
                                            'key',
                                            'loudness',
                                            'mode',
                                            'speechiness',
                                            'acousticness',
                                            'instrumentalness',
                                            'liveness',
                                            'valence',
                                            'tempo',
                                            'type',
                                            'id',
                                            'uri',
                                            'track_href',
                                            'analysis_url',
                                            'duration_ms',
                                            'time_signature',
                                            'song_title',
                                            'popularity']
                                            )

    # Loop over all tracks in the tracks['items'] list
    for track in tracks['items']:
        '''Get track data'''
        track_name = track['track']['name']
        track_popularity = track['track']['popularity']
        track_uri = track['track']['external_urls']['spotify']
        track_country = saving_csv_name[-6:-4]       
        track_features = sp.audio_features(track_uri)
        track_features[0]['song_title'] = track_name # add song name to feature list
        track_features[0]['popularity'] = track_popularity # add song popularity to feature list
        track_features[0]['country'] = track_country # add song popularity to feature list

        #album_name = track['track']['album']['name']

        '''Create DT with the songs features'''
        track_feature_pd = pd.concat([track_feature_pd, pd.DataFrame(track_features)])

        ### Could be interesting for a future analysis
        #track_audio_analysis = sp.audio_analysis(track['track']['id'])
        ###

        '''Get artist data'''
        artist_name = track['track']['artists'][0]['name']
        artist_id = track['track']['artists'][0]['id']
        artist = sp.artist(artist_id) # another fucking dictionary with 1000 infos

        '''Get lyrics'''
        try:
            song = genius.search_song(track_name, artist_name)


            if song == None:
                pass
            else:
                lyrics = song.lyrics

                if '/' in track_name:
                    continue

                else:

                    text_file = open(f'{track_name}.txt', 'w')
                    text_file.write(lyrics)
                    text_file.close()


        except requests.exceptions.Timeout:
            print('Timeout occurred')

    # adjust name of .csv file to be named how you want
    track_feature_pd.to_csv(saving_csv_name)


get_country_data('https://open.spotify.com/playlist/37i9dQZEVXbKqiTGXuCOsB?si=2523166715984c94')