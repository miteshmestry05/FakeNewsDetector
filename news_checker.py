from flask import Flask, request, render_template
from transformers import pipeline
import requests
import torch  # Ensure PyTorch is imported

# Initialize Flask app
app = Flask(__name__)

# Load AI model for misinformation detection
classifier = pipeline("text-classification", model="roberta-base-openai-detector", framework="pt")

# Function to analyze news credibility
def analyze_news(text):
    result = classifier(text)
    return result[0]['label']  # Returns "POSITIVE" (trusted) or "NEGATIVE" (fake)

# Function to fetch fact-check results from NewsAPI
def fact_check(news_text):
    api_key = "YOUR_NEWSAPI_KEY"  # Replace with your actual API key
    url = f"https://newsapi.org/v2/everything?q={news_text}&apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        if "articles" in data and data["articles"]:
            return f"Source: {data['articles'][0]['title']} ({data['articles'][0]['source']['name']})"
        else:
            return "No verified news sources found for this query."
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching fact-check: {e}"

# Define routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news_text = request.form['news_text']
        ai_result = analyze_news(news_text)
        fact_check_result = fact_check(news_text)
        return render_template("index.html", result=ai_result, fact=fact_check_result)

    return render_template("index.html", result='', fact='')  # Reset results on page reload

if __name__ == '__main__':
    app.run(debug=True)