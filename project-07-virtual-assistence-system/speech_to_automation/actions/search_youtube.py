from ..utils.speak import speak
from ..utils.language import set_translation
import webbrowser

def search_youtube(keyword: str, lang: str = 'en-US'):
    _ = set_translation(lang)
    url = f"https://www.youtube.com/results?search_query={keyword}"
    webbrowser.get().open(url)
    speak(_("Here is what I have found on youtube."), lang=lang)