import os
from dotenv import load_dotenv
import praw
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

# ===== 1. Reddit API認証 =====
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ===== 2. ブランド辞書 =====
brand_dict = {"Toyota": ["toyota"]}

def detect_brand(text, brand_dict):
    text = text.lower()
    for brand, keywords in brand_dict.items():
        for kw in keywords:
            if kw in text:
                return brand
    return None

# ===== 3. データ収集 =====
target_subreddits = ["cars", "whatcarshouldIbuy", "askcarsales"]
texts = []

for sub in target_subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=200):
        brand = detect_brand(post.title + " " + post.selftext, brand_dict)
        if brand == "Toyota":
            texts.append(post.title + " " + post.selftext)

# ===== 4. テキスト前処理 =====
text_data = " ".join(texts)

# ノイズ除去: 記号や一文字単語を削除
text_data = re.sub(r"\b[a-zA-Z]\b", " ", text_data)  

# ストップワード（独自追加）
custom_stopwords = {
    "toyota", "car", "cars", "t", "ve", "re", "ll", "im", "u", "dont", "doesnt", "cant"
}
stopwords = set(STOPWORDS).union(custom_stopwords)

# ===== 5. ワードクラウド生成 =====
wordcloud = WordCloud(
    width=1200, height=800,
    background_color="white",
    stopwords=stopwords,
    collocations=False
).generate(text_data)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Toyota WordCloud (Cleaned)", fontsize=20)
plt.show()
