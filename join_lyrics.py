import os
import tqdm

country_path = '/home/oscar/newsletter/spotify/data/ES/lyrics'

paths = os.listdir(country_path)

full: str = ''

for file in tqdm.tqdm(paths):
    print(f'Joining {file}')
    
    with open(os.path.join(country_path, file)) as f:
        content = f.read()

        full = full + '\n' + content

        f.close()
    
text_file = open('/home/oscar/newsletter/spotify/data/ES/full_lyrics.txt', 'w')
text_file.write(full)
text_file.close()