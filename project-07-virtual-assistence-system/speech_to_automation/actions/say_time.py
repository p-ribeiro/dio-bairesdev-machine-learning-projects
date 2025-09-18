from datetime import datetime

from ..utils.speak import speak

def say_time(lang: str = 'en-US'):
    strTime = datetime.today().strftime("%H:%M")
    print(strTime)
    speak(strTime, lang=lang)