from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_pattern(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        classification = "Positive"
    elif polarity < -0.1:
        classification = "Negative"
    else:
        classification = "Neutral"

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "classification": classification
    }


def analyze_lexicon(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        classification = "Positive"
    elif compound <= -0.05:
        classification = "Negative"
    else:
        classification = "Neutral"

    return {
        "compound": round(compound, 4),
        "positive": round(scores['pos'], 4),
        "negative": round(scores['neg'], 4),
        "neutral": round(scores['neu'], 4),
        "classification": classification
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False,
                            "message": "No data received."}), 400

        text = data.get("text", "")
        if not isinstance(text, str):
            return jsonify({"success": False,
                            "message": "Invalid text format."}), 400

        text = text.strip()
        if not text:
            return jsonify({"success": False,
                            "message": "Please enter some text."}), 400

        cleaned = clean_text(text)
        if not cleaned:
            return jsonify({"success": False,
                            "message": "Text contains no analyzable content."}), 400

        pattern_result = analyze_pattern(cleaned)
        lexicon_result = analyze_lexicon(cleaned)

        words = text.split()
        sentences = len(re.findall(r'[.!?]+', text))
        if sentences == 0:
            sentences = 1
        char_count = len(text)
        avg_word_length = round(char_count / len(words), 2) if words else 0

        return jsonify({
            "success": True,
            "pattern": pattern_result,
            "lexicon": lexicon_result,
            "total_words": len(words),
            "total_sentences": sentences,
            "total_characters": char_count,
            "avg_word_length": avg_word_length
        })

    except Exception as error:
        print("Error:", error)
        return jsonify({"success": False,
                        "message": "An error occurred while processing."}), 500


if __name__ == "__main__":
    app.run(debug=True)