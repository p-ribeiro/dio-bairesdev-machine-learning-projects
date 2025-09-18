from ctypes.wintypes import LANGID
from ..utils.speak import speak
from ..utils.get_audio import get_audio
import wikipedia


lang_dict = {
    'pt-BR': 'pt',
    'en-US': 'en'
}
def search_wikipedia(query: str, lang: str = 'en-US'):
    wikipedia.set_lang(lang_dict[lang])
    result = wikipedia.summary(query, sentences=3)
    speak("According to wikipedia")
    print(result)
    speak(result)