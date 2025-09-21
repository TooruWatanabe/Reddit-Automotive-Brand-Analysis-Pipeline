import os
from dotenv import load_dotenv
import praw
import pandas as pd

# ===== 1. Reddit API認証 =====
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ===== 2. ブランド辞書（メーカー名のみ） =====
brand_dict = {
    "Toyota": ["toyota"],
    "Tesla": ["tesla"],
    "BMW": ["bmw"],
    "Honda": ["honda"],
    "Ford": ["ford"],
    "Mercedes": ["mercedes", "mercedes-benz", "benz"],
    "Volkswagen": ["volkswagen", "vw"],
    "Nissan": ["nissan"],
    "Hyundai": ["hyundai"],
    "Kia": ["kia"],
    "Chevrolet": ["chevrolet", "chevy"],
    "Volvo": ["volvo"],
    "GM": ["gm", "general motors"],
    "Stellantis": ["stellantis"],
    "Rivian": ["rivian"],
    "Chrysler": ["chrysler"],
    "Jeep": ["jeep"],
    "Dodge": ["dodge"],
    "Ram": ["ram"],
}

def detect_brand(text, brand_dict):
    text = text.lower()
    for brand, keywords in brand_dict.items():
        for kw in keywords:
            if kw in text:
                return brand
    return None

# ===== 3. 公平なSubredditリスト =====
target_subreddits = ["cars", "whatcarshouldIbuy", "askcarsales"]

posts = []
for sub in target_subreddits:
    subreddit = reddit.subreddit(sub)
    print(f"📥 Collecting from r/{sub} ...")
    for post in subreddit.hot(limit=200):  # 各Subredditから200件ずつ
        brand = detect_brand(post.title + " " + post.selftext, brand_dict)
        if brand:
            posts.append({"subreddit": sub, "brand": brand})

# ===== 4. 集計 =====
df = pd.DataFrame(posts)
counts = df["brand"].value_counts()

print("✅ ブランド言及件数（全Subreddit合算）")
print(counts)

print("\n📊 Subredditごとの件数内訳")
print(df.groupby(["subreddit", "brand"]).size())
