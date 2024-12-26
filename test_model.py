import joblib

# Load the trained model
model = joblib.load('sentiment_analysis_model.pkl')

# Test sentences
test_sentences = [
    "Pretty decent.",
    "Not good at all.",
    "Absolutely fantastic!",
    "This is amazing!",
    "Worst ever!"
]

# Predict sentiments
predictions = model.predict(test_sentences)

for text, prediction in zip(test_sentences, predictions):
    print(f"'{text}' => Sentiment: {prediction}")
