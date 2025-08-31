import pyttsx3


def speak(text, emotion="neutral"):
    """
    Converts text to speech using pyttsx3 and plays it directly.
    """
    try:
        engine = pyttsx3.init()
        # You can get a list of available voices with engine.getProperty('voices')
        # And set a specific voice with engine.setProperty('voice', voice_id)

        # The emotion parameter is not directly supported by pyttsx3,
        # but you could use it to adjust rate or pitch to simulate emotion.
        # For example, speaking faster for "happy" emotion:
        if emotion == "happy":
            engine.setProperty('rate', 200)  # words per minute
        elif emotion == "sad":
            engine.setProperty('rate', 100)
        else:
            engine.setProperty('rate', 150)

        print(f"Speaking: '{text}' with emotion: {emotion}")
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"An error occurred during TTS or playback with pyttsx3: {e}")


if __name__ == '__main__':
    # This block allows you to test the function directly
    sample_text_happy = "Hello, I am your new AI tutor. I am happy to help!"
    speak(sample_text_happy, "happy")

    sample_text_neutral = "Please state your question."
    speak(sample_text_neutral, "neutral")

    sample_text_sad = "I am sorry, but I do not have a solution for this problem right now."
    speak(sample_text_sad, "sad")
