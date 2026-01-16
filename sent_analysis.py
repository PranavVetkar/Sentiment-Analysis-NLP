import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

class CryptoSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        score = self.sia.polarity_scores(text)['compound']
        
        if score >= 0.05:
            sentiment = "🚀 POSITIVE"
        elif score <= -0.05:
            sentiment = "📉 NEGATIVE"
        else:
            sentiment = "😐 NEUTRAL"
            
        return sentiment, score

if __name__ == "__main__":
    analyzer = CryptoSentimentAnalyzer()
    
    news_headlines = [
        "SEC approves first Bitcoin Spot ETF in historic move",
        "Major crypto exchange hacked, millions of USDT stolen",
        "Bitcoin price remains steady at 96k amid low volatility",
        "Elon Musk tweets a rocket emoji regarding Dogecoin"
    ]

    print("--- Crypto Sentiment Report ---")
    for headline in news_headlines:
        label, val = analyzer.analyze_text(headline)
        print(f"[{label} | Score: {val:+.2f}] -> {headline}")