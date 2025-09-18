import pyjokes

from ..utils.speak import speak

lang_dict = {
    'pt-BR': 'pt',
    'en-US': 'en'
}


def tell_joke(lang: str = 'en-US'):
    joke = pyjokes.get_joke()  # type: ignore
    speak(joke)  