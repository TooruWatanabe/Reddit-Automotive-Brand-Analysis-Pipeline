import os
import re
import json
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from dotenv import load_dotenv
import praw

# ==== CrewAI ====
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# =========================
# 0. 環境準備
# =========================
load_dotenv()
Path("data").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

# NLTK 辞書
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Reddit 認証
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# 解析対象
TARGET_SUBS = ["cars", "whatcarshouldIbuy", "askcarsales"]
LIMIT_PER_SUB = 200

# メーカー辞書
BRANDS: Dict[str, List[str]] = {
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

CUSTOM_STOPWORDS = {
    "car", "cars",
    "t", "ve", "re", "ll", "im", "u",
    "dont", "doesnt", "cant", "didnt", "isnt", "wont",
}

# =========================
# 1. ツール内部関数
# =========================
def detect_brand(text: str) -> str | None:
    t = text.lower()
    for brand, kws in BRANDS.items():
        for kw in kws:
            if kw in t:
                return brand
    return None

def collect_posts() -> pd.DataFrame:
    rows = []
    for sub in TARGET_SUBS:
        for post in reddit.subreddit(sub).hot(limit=LIMIT_PER_SUB):
            title = post.title or ""
            body = post.selftext or ""
            brand = detect_brand(title + " " + body)
            rows.append({
                "subreddit": sub,
                "title": title,
                "selftext": body,
                "brand": brand,
                "score": getattr(post, "score", None),
                "upvote_ratio": getattr(post, "upvote_ratio", None),
                "created": datetime.datetime.fromtimestamp(post.created_utc),
                "url": post.url
            })
    df = pd.DataFrame(rows)
    df.to_csv("data/raw_posts.csv", index=False)
    return df

def brand_counts(df: pd.DataFrame) -> pd.DataFrame:
    counted = df.dropna(subset=["brand"])["brand"].value_counts().rename_axis("brand").reset_index(name="count")
    counted.to_csv("data/brand_counts.csv", index=False)
    return counted

def brand_texts(df: pd.DataFrame, brand: str) -> tuple[str, set]:
    texts = df.loc[df["brand"] == brand, ["title", "selftext"]].fillna("").agg(" ".join, axis=1).tolist()
    corpus = " ".join(texts)
    corpus = re.sub(r"\b[a-zA-Z]\b", " ", corpus)
    sw = set(STOPWORDS).union(CUSTOM_STOPWORDS, {brand.lower(), *BRANDS[brand]})
    return corpus, sw

def make_wordcloud(corpus: str, stopwords: set[str], brand: str) -> str:
    if not corpus.strip():
        return ""
    wc = WordCloud(
        width=1200, height=800,
        background_color="white",
        stopwords=stopwords,
        collocations=False
    ).generate(corpus)
    out = f"images/{brand.lower()}_wordcloud.png"
    plt.figure(figsize=(12,8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{brand} WordCloud (Cleaned)", fontsize=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def sentiments(df: pd.DataFrame, brand: str) -> dict:
    texts = df.loc[df["brand"] == brand, ["title", "selftext"]].fillna("").agg(" ".join, axis=1).tolist()
    if not texts:
        return {"pos":0,"neg":0,"neu":0,"n":0}
    analyzer = SentimentIntensityAnalyzer()
    pos = neg = neu = 0
    for t in texts:
        s = analyzer.polarity_scores(t)
        if s["compound"] > 0.05:
            pos += 1
        elif s["compound"] < -0.05:
            neg += 1
        else:
            neu += 1
    res = {"pos":pos, "neg":neg, "neu":neu, "n":len(texts)}
    with open(f"data/{brand.lower()}_sentiment.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res

def summarize_findings(counts: pd.DataFrame, senti_map: dict, notes_map: dict) -> str:
    md = []
    md.append("## 総括（ハイライト）\n")
    top3 = counts.head(3).to_dict("records")
    hl = ", ".join([f"{r['brand']}({r['count']})" for r in top3])
    md.append(f"- 言及件数トップ: {hl}\n")
    md.append("- 感情は brandごとにポジ/ネガ/ニュートラルを集計し、投稿本文＋タイトルベースでVADER判定。\n")

    md.append("\n## ブランド別 所見\n")
    for brand in counts["brand"].tolist():
        s = senti_map.get(brand, {"pos":0, "neg":0, "neu":0, "n":0})
        note = notes_map.get(brand, "")
        posr = f"{round(100*s['pos']/s['n'],1)}%" if s["n"] else "-"
        negr = f"{round(100*s['neg']/s['n'],1)}%" if s["n"] else "-"
        neur = f"{round(100*s['neu']/s['n'],1)}%" if s["n"] else "-"
        md.append(f"### {brand}\n- 言及件数: {counts.loc[counts['brand']==brand, 'count'].item()}\n"
                  f"- 感情: ポジ {s['pos']} ({posr}) / ネガ {s['neg']} ({negr}) / ニュートラル {s['neu']} ({neur})\n"
                  f"- メモ: {note}\n")
    return "\n".join(md)

# =========================
# 2. ツールクラス
# =========================

class CollectTool(BaseTool):
    name: str = "Reddit収集ツール"
    description: str = "Redditから投稿を収集しCSVに保存する"

    def _run(self, **kwargs) -> str:
        df = collect_posts()

        # 保存先フォルダを用意
        outpath = Path("data/raw_posts.csv")
        outpath.parent.mkdir(exist_ok=True)

        # CSVに保存
        df.to_csv(outpath, index=False, encoding="utf-8")

        return f"収集完了: {len(df)}件 ({outpath}) を生成しました"



class AnalysisTool(BaseTool):
    name: str = "Reddit分析ツール"
    description: str = "収集した投稿データを分析してブランドごとの件数、ワードクラウド、感情分析を行う"

    def _run(self, raw_posts_file: str = "data/raw_posts.csv", **kwargs) -> str:
        input_path = Path(raw_posts_file)
        if not input_path.exists():
            return f"❌ 必要なデータファイルが存在しません: {input_path}"

        df = pd.read_csv(input_path)

        # ブランド集計
        counts = df["brand"].value_counts().reset_index()
        counts.columns = ["brand", "count"]
        counts.to_csv("data/brand_counts.csv", index=False, encoding="utf-8")

        # 感情集計（例）
        sentiments = {"pos": 0, "neg": 0, "neu": 0}
        for text in df["text"].tolist():
            s = simple_sentiment(text)
            sentiments[s] += 1
        sentiments["n"] = len(df)
        with open("data/sentiments_all.json", "w", encoding="utf-8") as f:
            json.dump(sentiments, f, ensure_ascii=False, indent=2)

        # ワードクラウド
        for brand in counts["brand"].tolist():
            texts = df[df["brand"] == brand]["text"].tolist()
            if not texts:
                continue
            text_blob = " ".join(texts)
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
            wc.to_file(f"images/{brand.lower()}_wordcloud.png")

        return "分析完了: brand_counts.csv, sentiments_all.json, ワードクラウド画像を生成しました"



class ReportTool(BaseTool):
    name: str = "レポート生成ツール"
    description: str = "Markdownレポートを生成する"

    def _run(self, **kwargs) -> str:
        brand_counts_path = Path(kwargs.get("brand_counts", "data/brand_counts.csv"))
        sentiment_path = Path(kwargs.get("sentiment", "data/sentiments_all.json"))
        wc_dir = Path(kwargs.get("word_cloud_dir", "images"))

        # データ読み込み
        counts = pd.read_csv(brand_counts_path)
        with open(sentiment_path, encoding="utf-8") as f:
            senti_map = json.load(f)

        # ワードクラウド画像パス
        wc_map = {
            b: str(wc_dir / f"{b.lower()}_wordcloud.png")
            for b in counts["brand"].tolist()
            if (wc_dir / f"{b.lower()}_wordcloud.png").exists()
        }

        today = datetime.date.today()
        md_path = Path(f"data/reddit_auto_brand_report_{today}.md")

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# 🚗 Reddit 自動車ブランド分析レポート\n\n")
            f.write(f"- 収集日: {today}\n\n")

            # ブランド件数
            f.write("## ブランド言及件数\n\n")
            f.write(counts.to_markdown(index=False))
            f.write("\n\n")

            # 感情分析
            f.write("## 感情分析結果\n\n")
            f.write("| ブランド | ポジ | ネガ | ニュートラル | 件数 |\n|---|---:|---:|---:|---:|\n")
            for brand in counts["brand"].tolist():
                s = senti_map.get(brand, {"pos": 0, "neg": 0, "neu": 0, "n": 0})
                f.write(f"| {brand} | {s['pos']} | {s['neg']} | {s['neu']} | {s['n']} |\n")
            f.write("\n\n")

            # ワードクラウド
            f.write("## ワードクラウド\n\n")
            for brand, path in wc_map.items():
                f.write(f"### {brand}\n![{brand}]({path})\n\n")

        return str(md_path)



# =========================
# 3. エージェントとタスク
# =========================
collector = Agent(role="収集", goal="Reddit投稿を集める", backstory="公平なデータ収集担当")
analyzer = Agent(role="分析", goal="投稿を分析する", backstory="データ解析担当")
reporter = Agent(role="レポート", goal="Markdownを書く", backstory="考察をまとめる担当")

task1 = Task(
    description="Reddit投稿を収集",
    agent=collector,
    tools=[CollectTool()],
    expected_output="raw_posts.csv が生成されていること"
)

task2 = Task(
    description="ブランド別分析",
    agent=analyzer,
    tools=[AnalysisTool()],
    expected_output="brand_counts.csv やワードクラウド、sentiment.json が生成されていること"
)

task3 = Task(
    description="Markdownレポートを作成",
    agent=reporter,
    tools=[ReportTool()],   # ✅ BaseTool 継承済みインスタンス
    expected_output="ブランド分析に基づいたMarkdownレポートファイル"
)



# =========================
# 4. 実行
# =========================
def run():
    crew = Crew(agents=[collector, analyzer, reporter], tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    print("\n✅ 完了:", result)

if __name__ == "__main__":
    run()

