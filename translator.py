from googletrans import Translator
import tqdm
import os

translator = Translator()

country_path = '/home/oscar/newsletter/spotify/data/UK/lyrics'

paths = os.listdir(country_path)

for file in tqdm.tqdm(paths):
    print(f'Translating {file}')
    
    with open(os.path.join(country_path, file)) as f:
        content = f.read()

        print(type(content))
        
        if content:
        
            translation = translator.translate(str(content), dest='en')

            # write to different file
            text_file = open(f'/home/oscar/newsletter/spotify/data/TU/lyrics_tr/{file}_tr.txt', 'w')
            text_file.write(translation.text)
            text_file.close()

            f.close()

        else:
            pass


# mi dick in your ass feels very good.