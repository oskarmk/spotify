# File to extract the lyrics from the top hits throughout history

# import modules
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import requests
from lyricsgenius import Genius
import os.path