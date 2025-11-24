from textblob import TextBlob

def detect_emotion(text):
    """Returns the emotion label based on sentiment polarity."""
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.2:
        return "😊 Emotion Detected: Happy"
    elif polarity < -0.2:
        return "😔 Emotion Detected: Sad"
    else:
        return "😐 Emotion Detected: Neutral"


def main():
    print("AI Emotion Detection from Text")
    print("-" * 40)

    while True:
        text = input("Enter a sentence (or type 'exit' to quit): ").strip()

        if text.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        if not text:
            print("⚠️ Please enter some text.\n")
            continue

        result = detect_emotion(text)
        print(result + "\n")


if __name__ == "__main__":
    main()
