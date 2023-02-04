# import modules
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
from lyricsgenius import Genius
import os.path

class SpotifyLyricsScraper():

    def __init__(self, user_id: str, client_id: str, client_secret: str, genius_id: str, playlist_link: str):
        self.user_id = user_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.genius_id = genius_id
        self.playlist_link = playlist_link

    def setup_connections(self):

        '''This function sets up the connection to Spotify & Genius API'''

        # Spotify
        self.client_credentials_manager = SpotifyClientCredentials(client_id=self.client_id,
                                                                   client_secret=self.client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)

        # Genius                                                    
        self.genius = Genius(self.genius_id)

    def get_country_data(self):


        '''This function retrieves the lyrics and song features of the songs included
           in the desired playlist.
           Input: Spotify link of the playlist
           Output: .txt files with song lyrics
           .csv file with the song features'''

        tracks = self.sp.playlist_tracks(self.playlist_link)

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
            track_features = self.sp.audio_features(track_uri)
            track_features[0]['song_title'] = track_name # add song name to feature list
            track_features[0]['popularity'] = track_popularity # add song popularity to feature list

            album_name = track['track']['album']['name']

            '''Create DT with the songs features'''
            track_feature_pd = pd.concat([track_feature_pd, pd.DataFrame(track_features)])

            ### Could be interesting for a future analysis
            #track_audio_analysis = sp.audio_analysis(track['track']['id'])
            ###

            '''Get artist data'''
            artist_name = track['track']['artists'][0]['name']
            artist_id = track['track']['artists'][0]['id']
            artist = self.sp.artist(artist_id) # another fucking dictionary with 1000 infos
            
            '''Get lyrics'''
            song = self.genius.search_song(track_name, artist_name)

            if song == None:
                pass
            else:
                lyrics = song.lyrics

                text_file = open(f'{track_name}.txt', 'w')
                text_file.write(lyrics)
                text_file.close()

        # adjust name of .csv file to be named how you want
        track_feature_pd.to_csv('track_features_US.csv')

test = SpotifyLyricsScraper(user_id = 'YOUR_SPOTIFY_USER_ID',
     client_id = 'YOUR_CLIENT_ID,
     client_secret = 'YOUR_CLIENT_SECRET',
     genius_id = 'YOUR_GENIUS_ID',
     playlist_link = 'https://open.spotify.com/playlist/5ABHKGoOzxkaa28ttQV9sE?si=3056f1cb35584b48')

test.setup_connections()
test.get_country_data()
