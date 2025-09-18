import winshell

from ..utils.speak import speak

def empty_recycle_bin():
    winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=True)
    speak("Recycle bin emptied")