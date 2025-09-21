import os
from dotenv import load_dotenv
import praw

# .env の読み込み
load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# Reddit API 認証
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# ここから実際の処理
subreddit = reddit.subreddit("cars")

for post in subreddit.hot(limit=10):
    print(post.title, "-", post.score)
