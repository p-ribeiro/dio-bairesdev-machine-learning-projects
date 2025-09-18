#instalar a biblioteca gTTS !pip install gTTS

from gtts import gTTS
import os

def text_to_audiofile(text: str, language: str = "en", filename: str = "gtts.wav"):

    local_directory = os.path.dirname(__file__)
    filepath = os.path.join(local_directory, filename)

    gtts_object = gTTS(text = text, 
                  lang = language,
                  slow = False)

    gtts_object.save(filepath)


if __name__ == "__main__":
    text = "Hello, welcome to the world of machine learning. This is a text to speech conversion example using gTTS library."
    text_to_audiofile(text)