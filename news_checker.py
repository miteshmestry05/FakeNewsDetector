from flask import Flask, request, render_template
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained AI model for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze news credibility
def analyze_news(text):
    result = classifier(text)
    return result[0]['label']  # Returns "POSITIVE" (trusted) or "NEGATIVE" (fake)

# Define routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_text = request.form['news_text']
        result = analyze_news(news_text)
        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

