import speech_recognition as sr
import pyttsx3
import os

speech = sr.Recognizer()

try:
    engine = pyttsx3.init()
except ImportError:
    print("Requested Driver not found")
except RuntimeError:
    print("Driver fails to initialize")

voices = engine.getProperty("voices")

for voice in voices:
    print(voice.id)

engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)

# engine.say("Hello boss, my name is Zend, your Artificial Intelligence")
# engine.runAndWait()


def speak_text_cmd(command):
    engine.say(command)
    print(command)
    engine.runAndWait()


def read_voice_cmd():
    voice_text = ''
    print("listening...")
    with sr.Microphone() as source:
        audio = speech.listen(source)

    try:
        voice_text = speech.recognize_google(audio)
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("Network error", str(e))

    return voice_text


if __name__ == '__main__':
    speak_text_cmd("Hello Sir, This is your Artificial Intelligence named Zend")

while True:
    voice_note = read_voice_cmd()
    print(voice_note)

    if "hello" in voice_note:
        speak_text_cmd("Hi!")
        continue
    elif 'open' in voice_note:
        os.system("explorer C:\\ {}".format(voice_note.replace("open", '')))
        continue