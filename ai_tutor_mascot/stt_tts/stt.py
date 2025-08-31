import speech_recognition as sr
from pydub import AudioSegment
import io

def listen_and_transcribe():
    """
    Listens for user speech, saves it as an audio file, and transcribes it to text.
    Returns the transcribed text or None if an error occurs.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, phrase_time_limit=5)

    try:
        print("Transcribing...")
        # Use Google's Web Speech API for transcription
        text = r.recognize_google(audio)
        print(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # This block allows you to test the function directly
    user_input = listen_and_transcribe()
    if user_input:
        print(f"Transcription successful: {user_input}")
    else:
        print("Transcription failed.")
