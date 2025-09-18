from pathlib import Path
import os
import gettext


def valid_languages():
    return ['en-US', 'pt-BR']

curr_dir = Path(__file__).resolve().parent
locales_dir = Path.joinpath(curr_dir.parent, 'locales')

def set_translation(system_language: str):
    if system_language == 'en-US':
        translator = lambda s: s
    else:
        try:
            translation = gettext.translation('messages', localedir=locales_dir, languages=[system_language])
            translation.install()
            translator = translation.gettext

        except FileNotFoundError:
            translator = lambda s: s
            print(f"Translation file not found for language '{system_language}'. Falling back to English.")
    
    return translator

