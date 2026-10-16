"""Fox8-23: build account features (features.csv) and labels (labels.csv) from the ndjson release."""

import argparse
import gzip
import json
import os
import numpy as np
import pandas as pd


def extract_user_features(user_obj: dict) -> dict:
    """Extract numerical profile features from a Twitter user object."""
    return {
        "followers_count":             user_obj.get("followers_count", 0) or 0,
        "friends_count":               user_obj.get("friends_count", 0) or 0,
        "statuses_count":              user_obj.get("statuses_count", 0) or 0,
        "favourites_count":            user_obj.get("favourites_count", 0) or 0,
        "listed_count":                user_obj.get("listed_count", 0) or 0,
        "verified":                    int(user_obj.get("verified", False) or False),
        "default_profile":             int(user_obj.get("default_profile", False) or False),
        "default_profile_image":       int(user_obj.get("default_profile_image", False) or False),
        "protected":                   int(user_obj.get("protected", False) or False),
        "geo_enabled":                 int(user_obj.get("geo_enabled", False) or False),
        "profile_use_background_image":int(user_obj.get("profile_use_background_image", False) or False),
        "has_extended_profile":        int(user_obj.get("has_extended_profile", False) or False),
        "contributors_enabled":        int(user_obj.get("contributors_enabled", False) or False),
        "is_translator":               int(user_obj.get("is_translator", False) or False),
        "is_translation_enabled":      int(user_obj.get("is_translation_enabled", False) or False),
        "profile_background_tile":     int(user_obj.get("profile_background_tile", False) or False),
        "description_len":             len(user_obj.get("description", "") or ""),
        "name_len":                    len(user_obj.get("name", "") or ""),
        "screen_name_len":             len(user_obj.get("screen_name", "") or ""),
        "has_location":                int(bool(user_obj.get("location", ""))),
        "has_url":                     int(bool(user_obj.get("url", ""))),
        "has_description":             int(bool(user_obj.get("description", ""))),
        # Ratio features
        "ff_ratio": (user_obj.get("followers_count", 0) or 0) /
                    max((user_obj.get("friends_count", 0) or 0) + 1, 1),
    }


def extract_tweet_features(tweets: list) -> dict:
    """Extract behavioural / textual features from a user's tweet list."""
    if not tweets:
        return {
            "n_tweets": 0, "retweet_ratio": 0.0, "reply_ratio": 0.0,
            "avg_tweet_len": 0.0, "avg_favorites": 0.0, "avg_retweets": 0.0,
            "avg_score": 0.0, "unique_lang_ratio": 0.0,
        }

    n = len(tweets)
    retweets = sum(1 for t in tweets if t.get("text", "").startswith("RT "))
    replies  = sum(1 for t in tweets if t.get("in_reply_to_status_id") is not None)
    texts    = [t.get("text", "") or "" for t in tweets]
    langs    = [t.get("lang", "und") for t in tweets]
    favs     = [t.get("favorite_count", 0) or 0 for t in tweets]
    rts      = [t.get("retweet_count", 0) or 0 for t in tweets]

    return {
        "n_tweets":          n,
        "retweet_ratio":     retweets / n,
        "reply_ratio":       replies / n,
        "avg_tweet_len":     float(np.mean([len(t) for t in texts])),
        "avg_favorites":     float(np.mean(favs)),
        "avg_retweets":      float(np.mean(rts)),
        "avg_score":         float(np.mean([f + r for f, r in zip(favs, rts)])),
        "unique_lang_ratio": len(set(langs)) / n,
    }


def process_fox8(ndjson_path: str, out_dir: str):
    """Read Fox8-23 ndjson and produce features.csv + labels.csv."""
    os.makedirs(out_dir, exist_ok=True)

    rows, labels_list = [], []
    print(f"Processing {ndjson_path} ...")

    opener = gzip.open if ndjson_path.endswith(".gz") else open
    with opener(ndjson_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("label") not in {"bot", "human"}:
                raise ValueError(f"Fox8-23 record {i} has missing/unknown label")
            label     = 1 if obj["label"] == "bot" else 0
            tweets    = obj.get("user_tweets", [])
            user_info = tweets[0]["user"] if tweets and "user" in tweets[0] else {}

            feat = {}
            feat["user_id"] = str(obj.get("user_id", i))
            feat.update(extract_user_features(user_info))
            feat.update(extract_tweet_features(tweets))
            rows.append(feat)
            labels_list.append(label)

            if (i + 1) % 200 == 0:
                print(f"  {i+1} records processed ...")

    df = pd.DataFrame(rows)
    labels_arr = np.array(labels_list, dtype=np.int64)
    if len(df) != 2280 or int(labels_arr.sum()) != 1140:
        raise ValueError(
            "Fox8-23 check: expected 2,280 records with "
            "1,140 bots and 1,140 humans; inspect input and JSON errors"
        )
    duplicate = df["user_id"].duplicated(keep="first").to_numpy()
    if duplicate.any():
        for uid in df.loc[duplicate, "user_id"].unique():
            if len(set(labels_arr[df["user_id"].eq(uid).to_numpy()])) != 1:
                raise ValueError(f"Fox8-23 user {uid} has conflicting labels")
        # One human account appears twice; keep its first record.
        print(f"  removing {int(duplicate.sum())} repeated user-id record(s)")
        df = df.loc[~duplicate].reset_index(drop=True)
        labels_arr = labels_arr[~duplicate]

    feat_path  = os.path.join(out_dir, "features.csv")
    label_path = os.path.join(out_dir, "labels.csv")
    df.to_csv(feat_path, index=False)
    pd.DataFrame({"label": labels_arr}).to_csv(label_path, index=False)

    print(f"\nDone! {len(df)} users.")
    print(f"  bot={labels_arr.sum()}, human={(labels_arr==0).sum()}")
    print(f"  Features ({df.shape[1]-1} dims) saved to {feat_path}")
    print(f"  Labels saved to {label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson",   required=True, help="Path to fox8_23_dataset.ndjson")
    parser.add_argument("--out_dir",  default="data/fox8_23",
                        help="Output directory for processed files")
    args = parser.parse_args()
    process_fox8(args.ndjson, args.out_dir)
