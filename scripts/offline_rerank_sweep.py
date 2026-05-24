import argparse
import itertools
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


GENERIC_RESTAURANT_TOKENS = {
    "restaurant", "restaurants", "food", "bars", "bar", "nightlife", "event",
    "events", "services", "planning", "venues", "shopping", "active", "life",
    "specialty", "traditional", "american", "new", "delivery", "caterers",
    "catering", "coffee", "tea",
}

DIASPORA_PROXY_TERMS = {
    "african", "senegalese", "ethiopian", "caribbean", "jamaican", "cajun",
    "creole", "southern", "soul", "bbq", "barbeque", "barbecue", "grill",
    "grilled", "smoke", "smoked", "spicy", "spice", "szechuan", "chinese",
    "indian", "pakistani", "thai", "vietnamese", "malaysian", "cambodian",
    "mexican", "tex", "tex-mex", "latin", "cuban", "brazilian", "seafood",
    "fish", "middle", "eastern", "mediterranean", "halal", "buffet",
    "rice", "noodles", "ramen", "tacos", "taqueria", "wings", "chicken",
}

ALCOHOL_HEAVY_TERMS = {
    "beer", "wine", "wines", "cocktail", "cocktails", "whiskey", "brewery",
    "breweries", "pub", "pubs", "bar", "bars", "nightlife", "sports bar",
}

PRETENTIOUS_OR_LIGHT_TERMS = {
    "french", "wine", "wines", "wine bars", "cocktail", "cocktail bars",
    "desserts", "gelato", "creperies", "bakery", "bakeries",
}


def parse_csv_list(value, cast=float):
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def split_semicolon(value):
    text = "" if value is None or pd.isna(value) else str(value)
    return [part.strip() for part in re.split(r";|\|", text) if part.strip()]


def tokenize_for_rerank(value):
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(value).lower())
        if token and token not in GENERIC_RESTAURANT_TOKENS
    ]


def load_user_items(path):
    user_items = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_items[int(parts[0])] = [int(item) for item in parts[1:]]
    return user_items


def load_profiles(dataset, n_avatars):
    persona_path = f"datasets/{dataset}/simulation/all_personas_like_modify.csv"
    history_path = f"datasets/{dataset}/raw_data/agg_top_25.csv"
    manifest_path = f"datasets/{dataset}/simulation/avatar_manifest.csv"

    personas = pd.read_csv(persona_path).iloc[:n_avatars].copy()

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
        if "user_id" in history.columns:
            history = history.set_index("user_id")
            mapping = {
                "restaurant_title_list": "history_titles",
                "restaurant_categories_list": "history_categories",
                "rating_list": "history_ratings",
            }
            for source, target in mapping.items():
                if source in history.columns:
                    personas[target] = personas.index.map(history[source])

    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        if "avatar_id" in manifest.columns:
            manifest = manifest.set_index("avatar_id")
            for column in ["ethnic_group", "pickiness", "selected_candidate_id", "selected_candidate_score"]:
                if column in manifest.columns:
                    personas[column] = personas.index.map(manifest[column])

    return personas


def item_text(item):
    return f"{item.get('title', '')} {item.get('genres', '')} {item.get('summary', '')}"


def build_profile(persona):
    weighted_profile = Counter()
    sources = [
        (split_semicolon(persona.get("frequent_categories", "")), 3.0),
        (split_semicolon(persona.get("history_categories", "")), 2.0),
        (split_semicolon(persona.get("liked_items", "")), 1.5),
        (split_semicolon(persona.get("history_titles", "")), 1.0),
        (split_semicolon(persona.get("taste", "")), 1.0),
    ]
    for source, weight in sources:
        for text in source:
            for token in tokenize_for_rerank(text):
                weighted_profile[token] += weight

    profile_categories = set()
    for text in split_semicolon(persona.get("frequent_categories", "")) + split_semicolon(persona.get("history_categories", "")):
        profile_categories.update(tokenize_for_rerank(text))

    disliked_tokens = set()
    for text in split_semicolon(persona.get("disliked_items", "")):
        disliked_tokens.update(tokenize_for_rerank(text))

    taste_text = " ".join(split_semicolon(persona.get("taste", ""))).lower()
    avoid_terms = set()
    if "avoid" in taste_text:
        avoid_terms = set(tokenize_for_rerank(taste_text.split("avoid", 1)[-1]))

    return {
        "weighted_profile": weighted_profile,
        "profile_categories": profile_categories,
        "disliked_tokens": disliked_tokens,
        "avoid_terms": avoid_terms,
        "profile_weight_sum": sum(weighted_profile.values()),
        "ethnic_group": str(persona.get("ethnic_group", "")).lower(),
        "price_sensitivity": str(persona.get("price_sensitivity", "medium")).lower(),
    }


def profile_semantic_score(profile, item):
    tokens = set(tokenize_for_rerank(item_text(item)))
    if not tokens or not profile["weighted_profile"]:
        return 0.0
    matched = sum(weight for token, weight in profile["weighted_profile"].items() if token in tokens)
    direct_overlap = matched / max(profile["profile_weight_sum"], 1.0)
    item_categories = set(tokenize_for_rerank(item.get("genres", "")))
    category_overlap = len(item_categories & profile["profile_categories"]) / max(min(len(profile["profile_categories"]), 10), 1)
    return min(1.0, 0.65 * direct_overlap * 4.0 + 0.35 * category_overlap)


def diaspora_proxy_score(profile, item):
    text = item_text(item).lower()
    tokens = set(tokenize_for_rerank(text))
    score = min(0.65, 0.09 * len(tokens & DIASPORA_PROXY_TERMS))
    ethnic_group = profile["ethnic_group"]
    if ethnic_group == "hausa":
        if {"halal", "grill", "grilled", "bbq", "barbeque", "rice", "indian", "middle", "eastern"} & tokens:
            score += 0.25
    elif ethnic_group == "igbo":
        if {"bbq", "barbeque", "seafood", "fish", "southern", "soul", "grill", "grilled", "indian", "mexican"} & tokens:
            score += 0.25
    elif ethnic_group == "yoruba":
        if {"spicy", "spice", "szechuan", "mexican", "indian", "seafood", "fish", "bbq", "barbeque", "thai"} & tokens:
            score += 0.25
    elif {"spicy", "spice", "bbq", "barbeque", "indian", "mexican", "thai", "african"} & tokens:
        score += 0.2
    if "price range 1" in text or "price range 2" in text:
        score += 0.08
    return min(1.0, score)


def price_match_score(profile, item):
    try:
        price = int(float(str(item.get("price", "2")).strip()))
    except (TypeError, ValueError):
        price = 2
    table = {
        "high": {1: 1.0, 2: 0.85, 3: 0.25, 4: 0.0},
        "medium": {1: 0.9, 2: 1.0, 3: 0.55, 4: 0.25},
        "low": {1: 0.55, 2: 0.85, 3: 0.9, 4: 0.75},
    }
    return table.get(profile["price_sensitivity"], table["medium"]).get(price, 0.5)


def social_proof_score(item):
    try:
        rating = float(item.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0.0
    try:
        reviews = float(item.get("review_count", 0))
    except (TypeError, ValueError):
        reviews = 0.0
    rating_score = max(0.0, min(1.0, (rating - 3.0) / 2.0))
    review_score = max(0.0, min(1.0, np.log1p(reviews) / np.log1p(3000)))
    return 0.65 * rating_score + 0.35 * review_score


def penalty_score(profile, item):
    tokens = set(tokenize_for_rerank(item_text(item)))
    penalty = 0.0
    if profile["disliked_tokens"] and tokens & profile["disliked_tokens"]:
        penalty += 0.35
    if profile["avoid_terms"] & tokens:
        penalty += 0.25
    if tokens & PRETENTIOUS_OR_LIGHT_TERMS:
        penalty += 0.15
    if profile["ethnic_group"] == "hausa" and tokens & ALCOHOL_HEAVY_TERMS:
        penalty += 0.3
    try:
        price = int(float(str(item.get("price", "2")).strip()))
    except (TypeError, ValueError):
        price = 2
    if profile["price_sensitivity"] == "high" and price >= 3:
        penalty += 0.35
    return min(1.0, penalty)


def attach_train_history_signals(profile, avatar_id, train_items, items):
    weighted_tokens = Counter()
    category_tokens = Counter()
    cities = Counter()
    for item_id in train_items.get(avatar_id, []):
        if item_id not in items.index:
            continue
        item = items.loc[item_id]
        for token in tokenize_for_rerank(item_text(item)):
            weighted_tokens[token] += 1.0
        for token in tokenize_for_rerank(item.get("genres", "")):
            category_tokens[token] += 1.0
        city = str(item.get("city", "")).strip().lower()
        if city:
            cities[city] += 1.0
    profile["train_weighted_profile"] = weighted_tokens
    profile["train_profile_weight_sum"] = sum(weighted_tokens.values())
    profile["train_categories"] = category_tokens
    profile["train_category_weight_sum"] = sum(category_tokens.values())
    profile["train_cities"] = cities
    profile["train_city_weight_sum"] = sum(cities.values())
    return profile


def train_history_score(profile, item):
    tokens = set(tokenize_for_rerank(item_text(item)))
    categories = set(tokenize_for_rerank(item.get("genres", "")))
    train_profile = profile.get("train_weighted_profile", Counter())
    direct_overlap = 0.0
    if train_profile and tokens:
        matched = sum(weight for token, weight in train_profile.items() if token in tokens)
        direct_overlap = min(1.0, 5.0 * matched / max(profile.get("train_profile_weight_sum", 1.0), 1.0))

    train_categories = profile.get("train_categories", Counter())
    category_overlap = 0.0
    if train_categories and categories:
        matched_categories = sum(weight for token, weight in train_categories.items() if token in categories)
        category_overlap = min(
            1.0,
            matched_categories / max(profile.get("train_category_weight_sum", 1.0) * 0.20, 1.0),
        )
    return min(1.0, 0.55 * direct_overlap + 0.45 * category_overlap)


def city_match_score(profile, item):
    cities = profile.get("train_cities", Counter())
    total = profile.get("train_city_weight_sum", 0.0)
    if not cities or total <= 0:
        return 0.0
    city = str(item.get("city", "")).strip().lower()
    return min(1.0, cities.get(city, 0.0) / total)


def precompute_catalog_components(profile, items):
    ids = np.array([int(item_id) for item_id in items.index], dtype=np.int64)
    id_to_pos = {int(item_id): pos for pos, item_id in enumerate(ids)}
    components = {
        "semantic": np.zeros(len(ids), dtype=float),
        "diaspora": np.zeros(len(ids), dtype=float),
        "history": np.zeros(len(ids), dtype=float),
        "city": np.zeros(len(ids), dtype=float),
        "price": np.zeros(len(ids), dtype=float),
        "social": np.zeros(len(ids), dtype=float),
        "penalty": np.zeros(len(ids), dtype=float),
    }
    for pos, item_id in enumerate(ids):
        item = items.loc[int(item_id)]
        components["semantic"][pos] = profile_semantic_score(profile, item)
        components["diaspora"][pos] = diaspora_proxy_score(profile, item)
        components["history"][pos] = train_history_score(profile, item)
        components["city"][pos] = city_match_score(profile, item)
        components["price"][pos] = price_match_score(profile, item)
        components["social"][pos] = social_proof_score(item)
        components["penalty"][pos] = penalty_score(profile, item)
    return {"ids": ids, "id_to_pos": id_to_pos, "components": components}


def rerank_one(ranking, profile, items, pool_size, weights):
    pool_size = min(pool_size, len(ranking))
    pool = list(ranking[:pool_size])
    tail = list(ranking[pool_size:])
    scored = []
    for local_rank, item_id in enumerate(pool):
        item = items.loc[item_id]
        cf_score = 1.0 - (local_rank / max(pool_size - 1, 1))
        score = (
            weights["cf"] * cf_score
            + weights["semantic"] * profile_semantic_score(profile, item)
            + weights["diaspora"] * diaspora_proxy_score(profile, item)
            + weights["price"] * price_match_score(profile, item)
            + weights["social"] * social_proof_score(item)
            - weights["penalty"] * penalty_score(profile, item)
        )
        scored.append((score, item_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [item_id for _score, item_id in scored] + tail


def semantic_retrieval_score(profile, item, weights):
    return (
        weights["semantic"] * profile_semantic_score(profile, item)
        + weights["diaspora"] * diaspora_proxy_score(profile, item)
        + weights.get("history", 0.0) * train_history_score(profile, item)
        + weights.get("city", 0.0) * city_match_score(profile, item)
        + weights["price"] * price_match_score(profile, item)
        + weights["social"] * social_proof_score(item)
        - weights["penalty"] * penalty_score(profile, item)
    )


def build_excluded_items(avatar_id, train_items, valid_items, test_items, split, allow_train):
    excluded = set()
    if not allow_train:
        excluded.update(train_items.get(avatar_id, []))
    if split == "test":
        excluded.update(valid_items.get(avatar_id, []))
    else:
        excluded.update(test_items.get(avatar_id, []))
    return excluded


def semantic_candidate_pool(profile, items, excluded, cf_pool, semantic_pool_size, weights, catalog_components=None):
    if semantic_pool_size <= 0:
        return []
    cf_pool = set(int(item_id) for item_id in cf_pool)
    if catalog_components is not None:
        ids = catalog_components["ids"]
        components = catalog_components["components"]
        scores = (
            weights["semantic"] * components["semantic"]
            + weights["diaspora"] * components["diaspora"]
            + weights.get("history", 0.0) * components["history"]
            + weights.get("city", 0.0) * components["city"]
            + weights["price"] * components["price"]
            + weights["social"] * components["social"]
            - weights["penalty"] * components["penalty"]
        )
        keep = np.array(
            [int(item_id) not in excluded and int(item_id) not in cf_pool and scores[pos] > 0 for pos, item_id in enumerate(ids)],
            dtype=bool,
        )
        candidate_ids = ids[keep]
        candidate_scores = scores[keep]
        order = np.lexsort((candidate_ids, -candidate_scores))
        return [int(item_id) for item_id in candidate_ids[order[:semantic_pool_size]]]

    scored = []
    for item_id in items.index:
        item_id = int(item_id)
        if item_id in excluded or item_id in cf_pool:
            continue
        item = items.loc[item_id]
        score = semantic_retrieval_score(profile, item, weights)
        if score > 0:
            scored.append((score, item_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [item_id for _score, item_id in scored[:semantic_pool_size]]


def precompute_components(
    ranking,
    profile,
    items,
    pool_size,
    semantic_pool_size=0,
    excluded=None,
    expansion_weights=None,
    cf_floor=0.05,
    catalog_components=None,
):
    pool_size = min(pool_size, len(ranking))
    cf_pool = list(ranking[:pool_size])
    if semantic_pool_size > 0:
        semantic_pool = semantic_candidate_pool(
            profile,
            items,
            excluded or set(),
            cf_pool,
            semantic_pool_size,
            expansion_weights,
            catalog_components=catalog_components,
        )
        pool = np.array(list(dict.fromkeys(cf_pool + semantic_pool)), dtype=np.int64)
    else:
        pool = np.array(cf_pool, dtype=np.int64)
    pool_set = set(int(item_id) for item_id in pool)
    tail = np.array([item_id for item_id in ranking if int(item_id) not in pool_set], dtype=np.int64)
    cf_rank_lookup = {int(item_id): local_rank for local_rank, item_id in enumerate(cf_pool)}
    components = {
        "cf": np.zeros(len(pool), dtype=float),
        "semantic": np.zeros(len(pool), dtype=float),
        "diaspora": np.zeros(len(pool), dtype=float),
        "history": np.zeros(len(pool), dtype=float),
        "city": np.zeros(len(pool), dtype=float),
        "price": np.zeros(len(pool), dtype=float),
        "social": np.zeros(len(pool), dtype=float),
        "penalty": np.zeros(len(pool), dtype=float),
    }
    for idx, item_id in enumerate(pool):
        component_pos = None
        if catalog_components is not None:
            component_pos = catalog_components["id_to_pos"].get(int(item_id))
        local_rank = cf_rank_lookup.get(int(item_id))
        if local_rank is None:
            components["cf"][idx] = cf_floor
        else:
            components["cf"][idx] = 1.0 - (local_rank / max(pool_size - 1, 1))
        if component_pos is not None:
            for name in ["semantic", "diaspora", "history", "city", "price", "social", "penalty"]:
                components[name][idx] = catalog_components["components"][name][component_pos]
        else:
            item = items.loc[int(item_id)]
            components["semantic"][idx] = profile_semantic_score(profile, item)
            components["diaspora"][idx] = diaspora_proxy_score(profile, item)
            components["history"][idx] = train_history_score(profile, item)
            components["city"][idx] = city_match_score(profile, item)
            components["price"][idx] = price_match_score(profile, item)
            components["social"][idx] = social_proof_score(item)
            components["penalty"][idx] = penalty_score(profile, item)
    return {"pool": pool, "tail": tail, "components": components}


def rerank_from_components(precomputed, weights):
    components = precomputed["components"]
    score = (
        weights["cf"] * components["cf"]
        + weights["semantic"] * components["semantic"]
        + weights["diaspora"] * components["diaspora"]
        + weights.get("history", 0.0) * components["history"]
        + weights.get("city", 0.0) * components["city"]
        + weights["price"] * components["price"]
        + weights["social"] * components["social"]
        - weights["penalty"] * components["penalty"]
    )
    pool = precomputed["pool"]
    # np.lexsort uses the last key as primary; this sorts by descending score, then item id.
    order = np.lexsort((pool, -score))
    return np.concatenate([pool[order], precomputed["tail"]])


def ndcg_at_k(top_items, gt, k):
    gt = set(gt)
    dcg = 0.0
    for rank, item_id in enumerate(top_items[:k], start=1):
        if item_id in gt:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(gt), k)
    if ideal_hits == 0:
        return np.nan
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_rankings(rankings, ground_truth, avatar_ids, k):
    hit_users = 0
    total_hits = 0
    total_gt = 0
    precisions = []
    recalls = []
    ndcgs = []
    ranks = []

    for row_idx, avatar_id in enumerate(avatar_ids):
        gt = set(ground_truth.get(avatar_id, []))
        if not gt:
            continue
        top_items = list(rankings[row_idx][:k])
        hits = len(set(top_items) & gt)
        total_hits += hits
        total_gt += len(gt)
        hit_users += int(hits > 0)
        precisions.append(hits / k)
        recalls.append(hits / len(gt))
        ndcgs.append(ndcg_at_k(top_items, gt, k))
        rank_lookup = {item_id: rank + 1 for rank, item_id in enumerate(rankings[row_idx])}
        for item_id in gt:
            if item_id in rank_lookup:
                ranks.append(rank_lookup[item_id])

    valid_ndcgs = [value for value in ndcgs if not np.isnan(value)]
    return {
        "hit_users_at_k": hit_users,
        "gt_items_at_k": total_hits,
        "total_gt_items": total_gt,
        "exposure_rate_at_k": hit_users / max(len(avatar_ids), 1),
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "micro_precision_at_k": total_hits / max(len(avatar_ids) * k, 1),
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "micro_recall_at_k": total_hits / max(total_gt, 1),
        "ndcg_at_k": float(np.mean(valid_ndcgs)) if valid_ndcgs else 0.0,
        "median_gt_rank": float(np.median(ranks)) if ranks else np.nan,
    }


def evaluate_proxy_utility(rankings, profiles, items, avatar_ids, k, catalog_components=None):
    """Estimate top-k persona quality without making LLM calls.

    This is intentionally not a replacement for the LLM simulation. It is a
    cheap guardrail so offline sweeps do not select a configuration that only
    pulls held-out IDs upward while lowering the food/persona fit that drives
    avatar satisfaction.
    """
    component_values = {
        "semantic": [],
        "diaspora": [],
        "history": [],
        "city": [],
        "price": [],
        "social": [],
        "penalty": [],
        "proxy_utility": [],
    }

    for row_idx, _avatar_id in enumerate(avatar_ids):
        profile = profiles[row_idx]
        for item_id in rankings[row_idx][:k]:
            component_pos = None
            if catalog_components is not None:
                component_pos = catalog_components[row_idx]["id_to_pos"].get(int(item_id))
            if component_pos is not None:
                row_components = catalog_components[row_idx]["components"]
                semantic = row_components["semantic"][component_pos]
                diaspora = row_components["diaspora"][component_pos]
                history = row_components["history"][component_pos]
                city = row_components["city"][component_pos]
                price = row_components["price"][component_pos]
                social = row_components["social"][component_pos]
                penalty = row_components["penalty"][component_pos]
            else:
                item = items.loc[int(item_id)]
                semantic = profile_semantic_score(profile, item)
                diaspora = diaspora_proxy_score(profile, item)
                history = train_history_score(profile, item)
                city = city_match_score(profile, item)
                price = price_match_score(profile, item)
                social = social_proof_score(item)
                penalty = penalty_score(profile, item)
            proxy_utility = (
                0.34 * semantic
                + 0.24 * diaspora
                + 0.14 * history
                + 0.10 * city
                + 0.08 * price
                + 0.05 * social
                - 0.20 * penalty
            )

            component_values["semantic"].append(semantic)
            component_values["diaspora"].append(diaspora)
            component_values["history"].append(history)
            component_values["city"].append(city)
            component_values["price"].append(price)
            component_values["social"].append(social)
            component_values["penalty"].append(penalty)
            component_values["proxy_utility"].append(proxy_utility)

    return {
        f"mean_{name}_at_k": float(np.mean(values)) if values else 0.0
        for name, values in component_values.items()
    }


def selection_score(metrics, args):
    if metrics["gt_items_at_k"] < args.min_gt_items_at_k:
        return -1.0
    return (
        args.objective_recall_weight * metrics["micro_recall_at_k"]
        + args.objective_ndcg_weight * metrics["ndcg_at_k"]
        + args.objective_utility_weight * metrics["mean_proxy_utility_at_k"]
        + args.objective_diaspora_weight * metrics["mean_diaspora_at_k"]
        - args.objective_penalty_weight * metrics["mean_penalty_at_k"]
    )


def main():
    parser = argparse.ArgumentParser(description="Offline sweep for hybrid reranker weights without LLM calls.")
    parser.add_argument("--dataset", default="yelp-kimi")
    parser.add_argument("--modeltype", default="LightGCN")
    parser.add_argument("--simulation_name", default="Test")
    parser.add_argument("--rankings_file", default=None)
    parser.add_argument("--n_avatars", type=int, default=50)
    parser.add_argument("--ground_truth_split", choices=["valid", "test"], default="valid")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--pool_sizes", default="100,200,500")
    parser.add_argument("--cf_weights", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--semantic_weights", default="0.10,0.20,0.30")
    parser.add_argument("--diaspora_weights", default="0.05,0.10,0.20")
    parser.add_argument("--history_weights", default="0.0")
    parser.add_argument("--city_weights", default="0.0")
    parser.add_argument("--price_weights", default="0.05,0.10")
    parser.add_argument("--social_weights", default="0.05")
    parser.add_argument("--penalty_weights", default="0.10,0.20,0.30")
    parser.add_argument("--semantic_candidate_expansion", action="store_true")
    parser.add_argument("--semantic_candidate_pool_sizes", default="0,100,200")
    parser.add_argument("--semantic_candidate_cf_floor", type=float, default=0.05)
    parser.add_argument("--allow_train_recommendations", action="store_true")
    parser.add_argument("--selection_objective", choices=["exposure", "balanced"], default="exposure",
                        help="exposure preserves the original best-config rule; balanced uses held-out recall plus persona utility.")
    parser.add_argument("--min_gt_items_at_k", type=int, default=0,
                        help="When using the balanced objective, reject configs with fewer exposed held-out items than this.")
    parser.add_argument("--objective_recall_weight", type=float, default=0.45)
    parser.add_argument("--objective_ndcg_weight", type=float, default=0.20)
    parser.add_argument("--objective_utility_weight", type=float, default=0.25)
    parser.add_argument("--objective_diaspora_weight", type=float, default=0.10)
    parser.add_argument("--objective_penalty_weight", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_best_ranking", action="store_true")
    args = parser.parse_args()

    rankings_file = args.rankings_file or (
        f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/rankings/full_rankings_{args.n_avatars}.npy"
    )
    output = args.output or (
        f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/rankings/offline_rerank_sweep.csv"
    )
    rankings = np.load(rankings_file)
    if len(rankings) < args.n_avatars:
        raise ValueError(f"Rankings file has {len(rankings)} rows, but n_avatars={args.n_avatars}")
    rankings = rankings[:args.n_avatars]
    avatar_ids = list(range(args.n_avatars))

    items = pd.read_csv(f"datasets/{args.dataset}/simulation/movie_detail.csv").set_index("movie_id", drop=False)
    personas = load_profiles(args.dataset, args.n_avatars)
    ground_truth = load_user_items(f"datasets/{args.dataset}/cf_data/{args.ground_truth_split}.txt")
    train_items = load_user_items(f"datasets/{args.dataset}/cf_data/train.txt")
    valid_items = load_user_items(f"datasets/{args.dataset}/cf_data/valid.txt")
    test_items = load_user_items(f"datasets/{args.dataset}/cf_data/test.txt")
    profiles = [
        attach_train_history_signals(build_profile(personas.iloc[avatar_id]), avatar_id, train_items, items)
        for avatar_id in avatar_ids
    ]
    print("Precomputing reusable persona-catalog features...", flush=True)
    catalog_components = [precompute_catalog_components(profile, items) for profile in profiles]
    excluded_items = [
        build_excluded_items(
            avatar_id,
            train_items,
            valid_items,
            test_items,
            args.ground_truth_split,
            args.allow_train_recommendations,
        )
        for avatar_id in avatar_ids
    ]

    base_metrics = {
        **evaluate_rankings(rankings, ground_truth, avatar_ids, args.top_k),
        **evaluate_proxy_utility(rankings, profiles, items, avatar_ids, args.top_k, catalog_components),
    }
    base_metrics["selection_score"] = selection_score(base_metrics, args)
    rows = [{
        "config": "base",
        "pool_size": 0,
        "cf": 1.0,
        "semantic": 0.0,
        "diaspora": 0.0,
        "history": 0.0,
        "city": 0.0,
        "price": 0.0,
        "social": 0.0,
        "penalty": 0.0,
        "semantic_candidate_pool_size": 0,
        "semantic_candidate_cf_floor": 0.0,
        **base_metrics,
    }]

    if args.selection_objective == "balanced":
        best_score = (
            base_metrics["selection_score"],
            base_metrics["gt_items_at_k"],
            base_metrics["ndcg_at_k"],
            base_metrics["mean_proxy_utility_at_k"],
        )
    else:
        best_score = (base_metrics["gt_items_at_k"], base_metrics["ndcg_at_k"], base_metrics["micro_precision_at_k"])
    best_rankings = rankings
    best_config = rows[0]

    pool_sizes = parse_csv_list(args.pool_sizes, int)
    semantic_candidate_pool_sizes = (
        parse_csv_list(args.semantic_candidate_pool_sizes, int)
        if args.semantic_candidate_expansion
        else [0]
    )
    component_cache = {}

    for pool_size in pool_sizes:
        grid = itertools.product(
            parse_csv_list(args.cf_weights),
            parse_csv_list(args.semantic_weights),
            parse_csv_list(args.diaspora_weights),
            parse_csv_list(args.history_weights),
            parse_csv_list(args.city_weights),
            parse_csv_list(args.price_weights),
            parse_csv_list(args.social_weights),
            parse_csv_list(args.penalty_weights),
            semantic_candidate_pool_sizes,
        )
        for cf, semantic, diaspora, history, city, price, social, penalty, semantic_pool_size in grid:
            weights = {
                "cf": cf,
                "semantic": semantic,
                "diaspora": diaspora,
                "history": history,
                "city": city,
                "price": price,
                "social": social,
                "penalty": penalty,
            }
            if semantic_pool_size > 0:
                cache_key = (
                    pool_size,
                    semantic_pool_size,
                    semantic,
                    diaspora,
                    history,
                    city,
                    price,
                    social,
                    penalty,
                    args.semantic_candidate_cf_floor,
                )
            else:
                cache_key = (pool_size, 0, args.semantic_candidate_cf_floor)
            if cache_key not in component_cache:
                print(
                    "Precomputing rerank components for "
                    f"pool_size={pool_size}, semantic_candidate_pool_size={semantic_pool_size}, "
                    f"weights={{semantic:{semantic}, diaspora:{diaspora}, history:{history}, city:{city}, "
                    f"price:{price}, social:{social}, penalty:{penalty}}}..."
                )
                component_cache[cache_key] = [
                    precompute_components(
                        rankings[row_idx],
                        profiles[row_idx],
                        items,
                        pool_size,
                        semantic_pool_size=semantic_pool_size,
                        excluded=excluded_items[row_idx],
                        expansion_weights=weights,
                        cf_floor=args.semantic_candidate_cf_floor,
                        catalog_components=catalog_components[row_idx],
                    )
                    for row_idx in range(args.n_avatars)
                ]
            reranked = np.array([
                rerank_from_components(component_cache[cache_key][row_idx], weights)
                for row_idx in range(args.n_avatars)
            ])
            metrics = {
                **evaluate_rankings(reranked, ground_truth, avatar_ids, args.top_k),
                **evaluate_proxy_utility(reranked, profiles, items, avatar_ids, args.top_k, catalog_components),
            }
            metrics["selection_score"] = selection_score(metrics, args)
            row = {
                "config": "hybrid",
                "pool_size": pool_size,
                "semantic_candidate_pool_size": semantic_pool_size,
                "semantic_candidate_cf_floor": args.semantic_candidate_cf_floor,
                **weights,
                **metrics,
            }
            rows.append(row)
            if args.selection_objective == "balanced":
                score = (
                    metrics["selection_score"],
                    metrics["gt_items_at_k"],
                    metrics["ndcg_at_k"],
                    metrics["mean_proxy_utility_at_k"],
                )
            else:
                score = (metrics["gt_items_at_k"], metrics["ndcg_at_k"], metrics["micro_precision_at_k"])
            if score > best_score:
                best_score = score
                best_rankings = reranked
                best_config = row

    if args.selection_objective == "balanced":
        results = pd.DataFrame(rows).sort_values(
            ["selection_score", "gt_items_at_k", "ndcg_at_k", "mean_proxy_utility_at_k"],
            ascending=[False, False, False, False],
        )
    else:
        results = pd.DataFrame(rows).sort_values(
            ["gt_items_at_k", "hit_users_at_k", "ndcg_at_k", "micro_precision_at_k"],
            ascending=[False, False, False, False],
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    results.to_csv(output, index=False)

    print(f"Wrote sweep results to {output}")
    print("\nBase metrics:")
    print(pd.Series(rows[0]).to_string())
    print("\nBest config:")
    print(pd.Series(best_config).to_string())
    print("\nTop 10 configs:")
    print(results.head(10).to_string(index=False))

    if args.save_best_ranking:
        best_path = output.replace(".csv", "_best_rankings.npy")
        np.save(best_path, best_rankings)
        print(f"\nSaved best rankings to {best_path}")


if __name__ == "__main__":
    main()
