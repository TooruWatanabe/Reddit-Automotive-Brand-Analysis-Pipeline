import os
from dotenv import load_dotenv
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 初回のみ
nltk.download("vader_lexicon")

# ===== 1. Reddit API認証 =====
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ===== 2. ブランド検出 =====
brand_dict = {"Tesla": ["tesla"]}

def detect_brand(text, brand_dict):
    text = text.lower()
    for brand, keywords in brand_dict.items():
        for kw in keywords:
            if kw in text:
                return brand
    return None

# ===== 3. 投稿収集 =====
texts = []
target_subreddits = ["cars", "whatcarshouldIbuy", "askcarsales"]

for sub in target_subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=200):
        brand = detect_brand(post.title + " " + post.selftext, brand_dict)
        if brand == "Tesla":
            texts.append(post.title + " " + post.selftext)

# ===== 4. 感情分析 =====
analyzer = SentimentIntensityAnalyzer()
results = [analyzer.polarity_scores(text) for text in texts]

pos = sum(1 for r in results if r["compound"] > 0.05)
neg = sum(1 for r in results if r["compound"] < -0.05)
neu = sum(1 for r in results if -0.05 <= r["compound"] <= 0.05)

print("✅ Tesla感情分析結果")
print(f"ポジティブ: {pos} 件")
print(f"ネガティブ: {neg} 件")
print(f"ニュートラル: {neu} 件")
