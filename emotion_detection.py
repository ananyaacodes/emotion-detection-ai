from textblob import TextBlob

print("AI Emotion Detection from Text")
print("-" * 40)

while True:
    text = input("Enter a sentence (or type 'exit' to quit): ")
    if text.lower() == 'exit':
        break

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        print("😊 Emotion Detected: Happy\n")
    elif polarity < -0.2:
        print("😔 Emotion Detected: Sad\n")
    else:
        print("😐 Emotion Detected: Neutral\n")
