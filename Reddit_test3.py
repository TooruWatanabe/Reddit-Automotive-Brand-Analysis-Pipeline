import os
from dotenv import load_dotenv
import praw
import pandas as pd
import datetime

# ===== 1. Reddit API認証 =====
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ===== 2. ブランド辞書 =====
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

# ===== 3. 対象Subreddit =====
target_subreddits = ["cars", "whatcarshouldIbuy", "askcarsales"]

posts = []
for sub in target_subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=200):
        brand = detect_brand(post.title + " " + post.selftext, brand_dict)
        if brand:
            posts.append({"subreddit": sub, "brand": brand})

# ===== 4. 集計 =====
df = pd.DataFrame(posts)
counts = df["brand"].value_counts()
sub_counts = df.groupby(["subreddit", "brand"]).size()

# ===== 5. Markdownレポート出力 =====
today = datetime.date.today()
filename = f"reddit_brand_report_{today}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# 🚗 Reddit 自動車ブランド分析レポート\n\n")
    f.write(f"対象Subreddit: {', '.join(target_subreddits)}\n")
    f.write(f"収集日: {today}\n\n")

    f.write("## ブランド言及件数（合算ランキング）\n\n")
    f.write("| ブランド | 件数 |\n|----------|------|\n")
    for brand, count in counts.items():
        f.write(f"| {brand} | {count} |\n")
    f.write("\n")

    f.write("## Subredditごとの内訳\n\n")
    for sub in target_subreddits:
        f.write(f"### r/{sub}\n\n")
        subset = sub_counts[sub] if sub in sub_counts.index else None
        if subset is not None:
            f.write("| ブランド | 件数 |\n|----------|------|\n")
            for brand, count in subset.items():
                f.write(f"| {brand} | {count} |\n")
            f.write("\n")

print(f"✅ Markdownレポートを出力しました → {filename}")
