from googletrans import Translator
import tqdm
import os

translator = Translator()

country_path = 'data/UK/lyrics'

paths = os.listdir(country_path)

for file in tqdm.tqdm(paths):
    print(f'Translating {file}')
    
    with open(os.path.join(country_path, file)) as f:
        content = f.read()

        print(type(content))
        
        if content:
        
            translation = translator.translate(str(content), dest='en')

            # write to different file
            text_file = open(f'data/UK/translated_lyrics/{file}_tr.txt', 'w')
            text_file.write(translation.text)
            text_file.close()

            f.close()
        
        else:
            pass
