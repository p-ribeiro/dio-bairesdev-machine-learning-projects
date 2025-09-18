import gettext
import os

import scipy as sp

from .utils.speak import speak
from .utils.get_audio import get_audio
from .utils.language import set_translation

from .actions.search_youtube import search_youtube
from .actions.search_wikipedia import search_wikipedia
from .actions.tell_joke import tell_joke
from .actions.emptry_recycle_bin import empty_recycle_bin
from .actions.say_time import say_time


## Translation setup

system_language = 'pt-BR'

_ = set_translation(system_language)

#function to respond to commands
def respond(text):
    print(f'{_("Audio text")}: {text}')
    
    if 'youtube' in text:
        speak(_("What do you want to search for?"), lang=system_language)
        keyword = get_audio(lang = system_language)
        if keyword!= '':
            search_youtube(keyword)
    
    
    elif _('search').lower() in text.lower():
        speak(_("What do you want to search for?"), lang=system_language)
        query = get_audio(lang = system_language)
        if query !='':
            search_wikipedia(query)
            
    elif _('joke') in text:
        tell_joke(lang=system_language)
        
    elif _('what time') in text:
        say_time(lang=system_language)
        
    elif _('exit').lower() in text.lower():
        speak(_("Goodbye, till next time"), lang=system_language)
        exit()
    else:
        speak(_("Sorry, I can't perform that action yet."), lang=system_language)

if __name__ == "__main__":
    while True:
        print(_("I am listening..."))
        text = get_audio(lang=system_language)
        respond(text)