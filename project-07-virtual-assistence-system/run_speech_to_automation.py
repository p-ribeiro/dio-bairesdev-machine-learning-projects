import argparse
import sys
import os
from pathlib import Path

# module imports
from speech_to_automation.utils.speak import speak
from speech_to_automation.utils.get_audio import get_audio
from speech_to_automation.utils.language import set_translation
from speech_to_automation.utils.language import valid_languages

from speech_to_automation.actions.say_time import say_time
from speech_to_automation.actions.search_youtube import search_youtube
from speech_to_automation.actions.search_wikipedia import search_wikipedia
from speech_to_automation.actions.tell_joke import tell_joke

_ = lambda s: s  # default to identity function

def listen(lang: str) -> str:
    print(_("I am listening..."))
    text = get_audio(lang=lang)
    return text

    

def main(args: list[str]):
    
    # set translation language
    if len(args) > 0:
        system_language = args[0]
        if system_language not in valid_languages():
            print(f"Unsupported language '{system_language}'. Falling back to English ('en-US').")
            system_language = 'en-US'
    else:
        system_language = 'en-US'
    
    global _
    _ = set_translation(system_language)
    
    while True:
        text = listen(lang=system_language)
        print(f'{_("Audio text")}: {text}')
        
        if 'youtube' in text:
            speak(_("What do you want to search for?"), lang=system_language)
            keyword = get_audio(lang=system_language)
            if keyword != '':
                search_youtube(keyword)
        
        elif _('search').lower() in text.lower():
            speak(_("What do you want to search for?"), lang=system_language)
            query = get_audio(lang=system_language)
            if query != '':
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
    main(args=sys.argv[1:])