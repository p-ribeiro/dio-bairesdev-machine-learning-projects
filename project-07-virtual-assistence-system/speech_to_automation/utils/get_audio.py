import speech_recognition as sr
from .speak import speak
from .language import set_translation


def get_audio(lang: str = 'en-US'):
    _ = set_translation(lang)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        # wait for a second to let the recognizer adjust the
        # energy threshold based on the surrounding noise level
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio, language=lang)             #type: ignore
            print(said)
        except sr.UnknownValueError:
            speak(_("Sorry, I did not get that."), lang=lang)
        except sr.RequestError:
            speak(_("Sorry, the service is not available"), lang=lang)
    return said.lower()