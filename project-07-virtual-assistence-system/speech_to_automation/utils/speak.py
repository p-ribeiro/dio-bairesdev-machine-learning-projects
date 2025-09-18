import os
from gtts import gTTS
import playsound

lang_dict = {
    'pt-BR': 'pt',
    'en-US': 'en'
}
    

def speak(text, lang: str ='en-US'):
    tts = gTTS(text=text, lang=lang_dict[lang])
    filename = "voice.mp3"
    try:
        os.remove(filename)
    except OSError:
        pass
    tts.save(filename)
    try:
        playsound.playsound(filename)
    except Exception as e:
        print(f"Error playing sound: {str(e).replace('\n',' ')}")
        print(f"Text that was attempted to be played: {text}")