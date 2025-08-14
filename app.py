import uuid
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# =========================
# Load Models
# =========================

# Transformer-based sentiment analysis
sentiment_model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=sentiment_model_id,
    truncation=True
)

# Instruction-tuned model for explanation and rephrasing
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # Change to flan-t5-xl if you have GPU
    truncation=True,
    max_length=256
)

# In-memory ticket store
tickets = {}


# =========================
# Helper Functions
# =========================

def classify_sentiment(text: str) -> str:
    """Classify sentiment into Positive, Negative, Neutral."""
    result = sentiment_analyzer(text)[0]["label"].lower()
    if "pos" in result:
        return "Positive"
    elif "neg" in result:
        return "Negative"
    return "Neutral"


def explain_review(text: str, sentiment: str) -> str:
    """Generate meaningful explanation of sentiment."""
    prompt = (
        f"Explain in 3-4 sentences why the following review expresses {sentiment.lower()} sentiment. "
        f"Do NOT repeat the review verbatim, provide reasoning and insights instead.\n\n"
        f"Review: {text}"
    )
    return llm(prompt)[0]["generated_text"].strip()


def rephrase_text(text: str) -> str:
    """Rephrase text in a neutral, professional tone."""
    prompt = (
        "Rephrase the following review in a professional, neutral, brand-friendly tone. "
        "Do NOT copy the original words; change sentence structure and wording.\n\n"
        f"Review: {text}"
    )
    return llm(prompt)[0]["generated_text"].strip()


# =========================
# Routes
# =========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze review sentiment and give explanation."""
    data = request.json
    review = data.get('review', '').strip()
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    sentiment = classify_sentiment(review)
    explanation = explain_review(review, sentiment)

    return jsonify({'sentiment': sentiment, 'explanation': explanation})


@app.route('/rephrase', methods=['POST'])
def rephrase():
    """Rephrase any provided text."""
    data = request.json
    text = data.get('review', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    rephrased = rephrase_text(text)
    return jsonify({'rephrased': rephrased})


@app.route('/create_ticket', methods=['POST'])
def create_ticket():
    """Store ticket in memory."""
    data = request.json
    ticket_id = str(uuid.uuid4())[:8]
    ticket = {
        'review': data.get('review', ''),
        'sentiment': data.get('sentiment', ''),
        'explanation': data.get('explanation', ''),
        'rephrased': data.get('rephrased', '')
    }
    tickets[ticket_id] = ticket
    return jsonify({'ticket_id': ticket_id, 'ticket': ticket})


if __name__ == "__main__":
    app.run(debug=True)
