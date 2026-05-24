from simulation.avatar import Avatar
from simulation.base.abstract_arena import abstract_arena
from termcolor import colored, cprint
import pandas as pd
import os
import os.path as op
import json

import time
import re
import numpy as np
import pickle
from collections import Counter

import simulation.vars as vars
from simulation.utils import *


def clamp_rating(value):
    try:
        rating = float(str(value).strip().strip(";"))
    except (TypeError, ValueError):
        return None
    if rating < 1.0 or rating > 5.0:
        return None
    return rating


def nanmean_or_nan(values):
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def strip_thinking_blocks(text):
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalize_title(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def is_placeholder_title(value):
    text = str(value or "").strip().lower()
    return (
        not text
        or "[" in text
        or "]" in text
        or "restaurant name" in text
        or "movie name" in text
        or "you choose" in text
        or "item name" in text
    )


def page_title_lookup(movie_detail, id_on_page):
    lookup = {}
    for item_id in id_on_page:
        title = str(movie_detail.loc[item_id].title).strip()
        lookup[normalize_title(title)] = (item_id, title)
    return lookup


def clean_yes_no(value):
    return str(value or "").strip().strip(";").lower()


ITEM_LABEL_PATTERN = r"(?:MOVIE|RESTAURANT|ITEM)"


def parse_page_response(response, movie_detail, id_on_page):
    """Parse avatar output without imputing missing ratings or hallucinated titles."""
    expected_lookup = page_title_lookup(movie_detail, id_on_page)
    text = strip_thinking_blocks(response)
    rating_rows = []
    align_rows = []
    errors = []

    rating_pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?{ITEM_LABEL_PATTERN}:\s*(.+?)\s*;\s*RATING:\s*([^;]+)\s*;\s*FEELING:\s*(.*?)\s*;?\s*$",
        re.IGNORECASE,
    )
    align_pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?{ITEM_LABEL_PATTERN}:\s*(.+?)\s*;\s*ALIGN:\s*([^;]+)\s*;\s*REASON:\s*(.*?)\s*;?\s*$",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not re.match(rf"^\s*(?:[-*]\s*)?{ITEM_LABEL_PATTERN}:", line, flags=re.IGNORECASE):
            continue

        rating_match = rating_pattern.match(line)
        if rating_match:
            raw_title, raw_rating, feeling = rating_match.groups()
            title_key = normalize_title(raw_title)
            rating = clamp_rating(raw_rating)
            if is_placeholder_title(raw_title):
                errors.append(f"placeholder rating title ignored: {raw_title}")
                continue
            if title_key not in expected_lookup:
                errors.append(f"off-page rating title ignored: {raw_title}")
                continue
            if rating is None:
                errors.append(f"invalid rating ignored for {raw_title}: {raw_rating}")
                continue
            item_id, canonical_title = expected_lookup[title_key]
            rating_rows.append(
                {
                    "item_id": item_id,
                    "title": canonical_title,
                    "rating": rating,
                    "feeling": str(feeling).strip().strip(";"),
                }
            )
            continue

        align_match = align_pattern.match(line)
        if align_match:
            raw_title, raw_align, reason = align_match.groups()
            title_key = normalize_title(raw_title)
            align_value = clean_yes_no(raw_align)
            if is_placeholder_title(raw_title):
                errors.append(f"placeholder align title ignored: {raw_title}")
                continue
            if title_key not in expected_lookup:
                errors.append(f"off-page align title ignored: {raw_title}")
                continue
            if align_value not in {"yes", "no"}:
                errors.append(f"invalid ALIGN value ignored for {raw_title}: {raw_align}")
                continue
            item_id, canonical_title = expected_lookup[title_key]
            align_rows.append(
                {
                    "item_id": item_id,
                    "title": canonical_title,
                    "align": align_value,
                    "reason": str(reason).strip().strip(";"),
                }
            )
            continue

        if "RATING:" in line.upper() or "ALIGN:" in line.upper():
            errors.append(f"unparseable structured line ignored: {line[:160]}")

    parsed_title_count = len({row["title"] for row in align_rows})
    expected_count = len(id_on_page)
    parse_failed = len(align_rows) == 0 and len(rating_rows) == 0
    parse_partial = (not parse_failed) and parsed_title_count < expected_count
    if parse_failed:
        errors.append("no valid on-page ALIGN or RATING lines were parsed")
    elif parse_partial:
        errors.append(f"partial parse: {parsed_title_count}/{expected_count} on-page ALIGN lines parsed")

    return {
        "rating_rows": rating_rows,
        "align_rows": align_rows,
        "parse_failed": parse_failed,
        "parse_partial": parse_partial,
        "errors": errors,
    }


def parse_forced_watch_response(response, movie_detail, forced_items_ids):
    """Parse validation WATCH decisions using restaurant-native labels."""
    expected_lookup = page_title_lookup(movie_detail, forced_items_ids)
    pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?{ITEM_LABEL_PATTERN}:\s*(.+?)\s*;\s*WATCH:\s*([^;]+)\s*;\s*REASON:\s*(.*?)\s*;?\s*$",
        re.IGNORECASE,
    )
    choices = []
    errors = []
    for raw_line in strip_thinking_blocks(response).splitlines():
        line = raw_line.strip()
        if not line or not re.match(rf"^\s*(?:[-*]\s*)?{ITEM_LABEL_PATTERN}:", line, flags=re.IGNORECASE):
            continue
        match = pattern.match(line)
        if not match:
            if "WATCH:" in line.upper():
                errors.append(f"unparseable validation line ignored: {line[:160]}")
            continue
        raw_title, raw_watch, reason = match.groups()
        title_key = normalize_title(raw_title)
        watch_value = clean_yes_no(raw_watch)
        if title_key not in expected_lookup:
            errors.append(f"off-page validation title ignored: {raw_title}")
            continue
        if watch_value not in {"yes", "no"}:
            errors.append(f"invalid WATCH value ignored for {raw_title}: {raw_watch}")
            continue
        item_id, canonical_title = expected_lookup[title_key]
        choices.append(
            {
                "item_id": item_id,
                "title": canonical_title,
                "watch": watch_value,
                "reason": str(reason).strip().strip(";"),
            }
        )
    return choices, errors


def parse_interview_rating(interview):
    if isinstance(interview, dict) and "interview" in interview:
        values = interview.get("interview") or []
        if values:
            first = str(values[0])
            match = re.search(r"(\d+(?:\.\d+)?)", first)
            if match:
                return max(1.0, min(10.0, float(match.group(1))))
    text = str(interview)
    match = re.search(r"RATING:\s*(\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"['\"]?\s*(\d+(?:\.\d+)?)\s*;\s*REASON", text)
    if not match:
        return None
    try:
        rating = float(match.group(1))
    except ValueError:
        return None
    return max(1.0, min(10.0, rating))


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


def tokenize_for_rerank(value):
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(value).lower())
        if token and token not in GENERIC_RESTAURANT_TOKENS
    ]


class Arena(abstract_arena):
    def __init__(self, args):
        super().__init__(args)
        
        self.max_pages = args.max_pages
        self.finished_num = 0

    def load_additional_info(self):
        
        self.user_profile_csv = pd.read_csv(f'datasets/{self.dataset}/raw_data/agg_top_25.csv')

        # return super().load_additional_info()
        self.add_advert = self.args.add_advert
        self.display_advert = self.args.display_advert
        if(self.add_advert):
            self.total_adverts, self.clicked_adverts = 0, 0
            advert_pool = pd.read_pickle(f'datasets/{self.dataset}/simulation/advertisement_review.pkl')
            advert_dict = {'all': {**advert_pool['pop_high_rating'], **advert_pool['pop_low_rating'], **advert_pool['unpop_high_rating'], **advert_pool['unpop_low_rating']}, 
                        'pop_high':advert_pool['pop_high_rating'], 'pop_low':advert_pool['pop_low_rating'], 'unpop_high':advert_pool['unpop_high_rating'], 'unpop_low':advert_pool['unpop_low_rating']}
            # print(self.args.advert_type)
            self.advert = advert_dict[self.args.advert_type]
            if "yelp" in str(self.dataset).lower():
                self.advert_word = "A sponsored restaurant you may want to try: "
            else:
                self.advert_word = "The best movie you should not miss in your life! "

    def initialize_all_avatars(self):
        """
        initialize avatars
        """
        super().initialize_all_avatars()
        # self.persona_df = pd.read_csv(f"datasets/{self.dataset}/simulation/all_personas_like_information_house.csv")
        persona_dir = getattr(self.args, "persona_dir", None)
        if persona_dir:
            persona_path = op.join(persona_dir, "all_personas_like_modify.csv")
            statistic_path = op.join(persona_dir, "user_statistic.csv")
        else:
            persona_path = f"datasets/{self.dataset}/simulation/all_personas_like_modify.csv"
            statistic_path = f"datasets/{self.dataset}/simulation/user_statistic.csv"
        self.persona_df = pd.read_csv(persona_path)
        self.user_statistic = pd.read_csv(statistic_path, index_col=0)
        if hasattr(self, "user_profile_csv") and self.user_profile_csv is not None:
            history_df = self.user_profile_csv.copy()
            if "user_id" in history_df.columns:
                history_df = history_df.set_index("user_id")
                history_columns = {
                    "restaurant_title_list": "history_titles",
                    "restaurant_categories_list": "history_categories",
                    "rating_list": "history_ratings",
                }
                for source_column, target_column in history_columns.items():
                    if source_column in history_df.columns:
                        self.persona_df[target_column] = self.persona_df.index.map(history_df[source_column])
        manifest_path = op.join(op.dirname(persona_path), "avatar_manifest.csv")
        if op.exists(manifest_path):
            manifest_df = pd.read_csv(manifest_path)
            if "avatar_id" in manifest_df.columns:
                manifest_df = manifest_df.set_index("avatar_id")
                for column in ["ethnic_group", "pickiness", "selected_candidate_id", "selected_candidate_score"]:
                    if column in manifest_df.columns:
                        self.persona_df[column] = self.persona_df.index.map(manifest_df[column])
        # @ avatars and evaluation indicators
        self.avatars = {}
        self.ratings = {}
        self.new_train_dict = {}
        self.exit_page = {}
        self.perf_per_page = {}
        self.expanded_perf_per_page = {}
        self.watch = {}
        self.n_likes = {}
        self.parse_valid_pages = {}
        self.parse_partial_pages = {}
        self.parse_failed_pages = {}
        self.parse_errors = {}
        self.remaining_users = list(range(self.n_avatars))

        for avatar_id in self.simulated_avatars_id:
            self.avatars[avatar_id] = Avatar(self.args, avatar_id, self.persona_df.loc[avatar_id], self.user_statistic.loc[avatar_id])
            self.new_train_dict[avatar_id] = self.data.train_user_list[avatar_id]
            self.ratings[avatar_id] = []
            self.n_likes[avatar_id] = []
            self.watch[avatar_id] = []
            self.exit_page[avatar_id] = 0
            self.perf_per_page[avatar_id] = []
            self.expanded_perf_per_page[avatar_id] = []
            self.parse_valid_pages[avatar_id] = []
            self.parse_partial_pages[avatar_id] = []
            self.parse_failed_pages[avatar_id] = []
            self.parse_errors[avatar_id] = {}

    def ground_truth_items(self, avatar_id):
        split = getattr(self.args, "ground_truth_split", "valid")
        if split == "test":
            return self.data.test_user_list[avatar_id]
        return self.data.valid_user_list[avatar_id]

    def natural_relevance_items(self, avatar_id):
        """Return the relevance set used for natural-simulation metrics.

        The strict setting is held-out only. The expanded setting is reported
        separately and treats persona/history positives as relevant too, which
        better reflects food-preference utility under sparse Yelp labels.
        """
        relevance = set(self.ground_truth_items(avatar_id))
        if getattr(self.args, "natural_relevance_scope", "heldout") == "heldout":
            return relevance

        title_to_ids = {}
        for item_id, title in zip(self.movie_detail["movie_id"], self.movie_detail["title"]):
            title_to_ids.setdefault(normalize_title(title), set()).add(int(item_id))

        avatar = self.avatars[avatar_id]
        positive_titles = set(getattr(avatar, "liked_items", [])) | set(getattr(avatar, "evidence_items", []))
        for title, rating in zip(getattr(avatar, "history_titles", []), getattr(avatar, "history_ratings", [])):
            if rating >= 4:
                positive_titles.add(title)

        for title in positive_titles:
            relevance.update(title_to_ids.get(normalize_title(title), set()))
        return relevance

    def maybe_inject_ground_truth(self, id_on_page, avatar_id, page_num):
        if not getattr(self.args, "inject_ground_truth", False):
            return id_on_page
        ground_truth = list(self.ground_truth_items(avatar_id))
        if not ground_truth or set(id_on_page) & set(ground_truth):
            return id_on_page
        candidate = ground_truth[(page_num - 1) % len(ground_truth)]
        if candidate in id_on_page:
            return id_on_page
        patched_page = list(id_on_page)
        if patched_page:
            patched_page[-1] = candidate
        return np.array(patched_page)

    def get_full_rankings(self, filename="full_rankings", batch_size=512):
        super().get_full_rankings(filename=filename, batch_size=batch_size)
        if getattr(self.args, "hybrid_rerank", False):
            self.apply_hybrid_rerank(filename=filename)

    def apply_hybrid_rerank(self, filename="full_rankings"):
        pool_size = max(int(getattr(self.args, "rerank_pool_size", 100)), self.items_per_page)
        expansion_enabled = getattr(self.args, "semantic_candidate_expansion", False)
        semantic_pool_size = max(int(getattr(self.args, "semantic_candidate_pool_size", 200)), 0)
        reranked_rows = []
        diagnostics = []
        before_hits, after_hits = 0, 0
        before_exposed, after_exposed = 0, 0
        candidate_pool_sizes = []

        for row_idx, avatar_id in enumerate(self.simulated_avatars_id):
            ranking = list(self.full_rankings[row_idx])
            pool = list(ranking[:pool_size])
            if expansion_enabled and semantic_pool_size > 0:
                semantic_pool = self.semantic_candidate_pool(avatar_id, pool, semantic_pool_size)
                candidate_pool = list(dict.fromkeys(pool + semantic_pool))
            else:
                semantic_pool = []
                candidate_pool = pool
            candidate_set = set(candidate_pool)
            tail = [item_id for item_id in ranking if item_id not in candidate_set]
            avatar = self.avatars[avatar_id]
            cf_rank_lookup = {item_id: local_rank for local_rank, item_id in enumerate(pool)}
            scored_pool = [
                (self.hybrid_rerank_score(avatar, item_id, cf_rank_lookup.get(item_id), pool_size), item_id)
                for item_id in candidate_pool
            ]
            scored_pool.sort(key=lambda x: (-x[0], x[1]))
            reranked_pool = [item_id for _, item_id in scored_pool]
            reranked = reranked_pool + tail
            reranked_rows.append(reranked)
            candidate_pool_sizes.append(len(candidate_pool))

            gt = set(self.ground_truth_items(avatar_id))
            before_top = set(ranking[:self.items_per_page * self.max_pages])
            after_top = set(reranked[:self.items_per_page * self.max_pages])
            before_overlap = before_top & gt
            after_overlap = after_top & gt
            before_hits += 1 if before_overlap else 0
            after_hits += 1 if after_overlap else 0
            before_exposed += len(before_overlap)
            after_exposed += len(after_overlap)
            diagnostics.append(
                f"{avatar_id}\tpool={len(candidate_pool)}\tsemantic_added={len(set(semantic_pool) - set(pool))}"
                f"\tbefore={sorted(before_overlap)}\tafter={sorted(after_overlap)}"
            )

        self.full_rankings = np.array(reranked_rows)
        np.save(
            self.storage_base_path + "/rankings/" + f"/{filename}_{self.n_avatars}_hybrid.npy",
            self.full_rankings,
        )
        diag_path = self.storage_base_path + "/rankings/hybrid_rerank_diagnostics.txt"
        with open(diag_path, "w") as f:
            f.write(f"pool_size={pool_size}\n")
            f.write(f"semantic_candidate_expansion={expansion_enabled}\n")
            f.write(f"semantic_candidate_pool_size={semantic_pool_size}\n")
            f.write(f"average_candidate_pool_size={np.mean(candidate_pool_sizes) if candidate_pool_sizes else 0}\n")
            f.write(f"top_window={self.items_per_page * self.max_pages}\n")
            f.write(f"before_hit_users={before_hits}\n")
            f.write(f"after_hit_users={after_hits}\n")
            f.write(f"before_exposed_items={before_exposed}\n")
            f.write(f"after_exposed_items={after_exposed}\n")
            f.write("\n".join(diagnostics))
        cprint(
            f"Hybrid rerank applied: GT exposure in simulation window {before_exposed} -> {after_exposed} items "
            f"({before_hits} -> {after_hits} users). "
            f"Semantic expansion={'on' if expansion_enabled else 'off'}.",
            color="cyan",
            attrs=["bold"],
        )

    def semantic_candidate_pool(self, avatar_id, cf_pool, semantic_pool_size):
        avatar = self.avatars[avatar_id]
        if not hasattr(avatar, "_rerank_profile"):
            avatar._rerank_profile = self.build_rerank_profile(avatar)

        excluded = self.rerank_excluded_items(avatar_id)
        cf_pool_set = set(cf_pool)
        scored = []
        for item_id in self.movie_detail["movie_id"]:
            item_id = int(item_id)
            if item_id in excluded or item_id in cf_pool_set:
                continue
            item = self.movie_detail.loc[item_id]
            score = self.semantic_retrieval_score(avatar, item)
            if score > 0:
                scored.append((score, item_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [item_id for _score, item_id in scored[:semantic_pool_size]]

    def rerank_excluded_items(self, avatar_id):
        split = getattr(self.args, "ground_truth_split", "valid")
        excluded = set()
        if not getattr(self.args, "allow_train_recommendations", False):
            if getattr(self.data, "nodrop", False):
                excluded.update(self.data.train_nodrop_user_list[avatar_id])
            else:
                excluded.update(self.data.train_user_list[avatar_id])
        if split == "test":
            excluded.update(self.data.valid_user_list[avatar_id])
        else:
            excluded.update(self.data.test_user_list[avatar_id])
        return excluded

    def semantic_retrieval_score(self, avatar, item):
        semantic_score = self.profile_semantic_score(avatar, item)
        diaspora_score = self.diaspora_proxy_score(avatar, item)
        price_score = self.price_match_score(avatar, item)
        social_score = self.social_proof_score(item)
        penalty = self.rerank_penalty_score(avatar, item)
        return (
            getattr(self.args, "rerank_semantic_weight", 0.30) * semantic_score
            + getattr(self.args, "rerank_diaspora_weight", 0.25) * diaspora_score
            + getattr(self.args, "rerank_history_weight", 0.0) * self.train_history_score(avatar, item)
            + getattr(self.args, "rerank_city_weight", 0.0) * self.city_match_score(avatar, item)
            + getattr(self.args, "rerank_price_weight", 0.12) * price_score
            + getattr(self.args, "rerank_social_weight", 0.08) * social_score
            - getattr(self.args, "rerank_penalty_weight", 0.35) * penalty
        )

    def hybrid_rerank_score(self, avatar, item_id, local_rank, pool_size):
        item = self.movie_detail.loc[item_id]
        if not hasattr(avatar, "_rerank_profile"):
            avatar._rerank_profile = self.build_rerank_profile(avatar)
        if local_rank is None:
            cf_score = float(getattr(self.args, "semantic_candidate_cf_floor", 0.05))
        else:
            cf_score = 1.0 - (local_rank / max(pool_size - 1, 1))
        semantic_score = self.profile_semantic_score(avatar, item)
        diaspora_score = self.diaspora_proxy_score(avatar, item)
        price_score = self.price_match_score(avatar, item)
        social_score = self.social_proof_score(item)
        penalty = self.rerank_penalty_score(avatar, item)
        return (
            getattr(self.args, "rerank_cf_weight", 0.55) * cf_score
            + getattr(self.args, "rerank_semantic_weight", 0.30) * semantic_score
            + getattr(self.args, "rerank_diaspora_weight", 0.25) * diaspora_score
            + getattr(self.args, "rerank_history_weight", 0.0) * self.train_history_score(avatar, item)
            + getattr(self.args, "rerank_city_weight", 0.0) * self.city_match_score(avatar, item)
            + getattr(self.args, "rerank_price_weight", 0.12) * price_score
            + getattr(self.args, "rerank_social_weight", 0.08) * social_score
            - getattr(self.args, "rerank_penalty_weight", 0.35) * penalty
        )

    def build_rerank_profile(self, avatar):
        weighted_profile = Counter()
        for source, weight in [
            (getattr(avatar, "frequent_categories", []), 3.0),
            (getattr(avatar, "history_categories", []), 2.0),
            (getattr(avatar, "liked_items", []), 1.5),
            (getattr(avatar, "history_titles", []), 1.0),
            (getattr(avatar, "taste", []), 1.0),
        ]:
            for text in source:
                for token in tokenize_for_rerank(text):
                    weighted_profile[token] += weight

        profile_categories = set()
        for text in getattr(avatar, "frequent_categories", []) + getattr(avatar, "history_categories", []):
            profile_categories.update(tokenize_for_rerank(text))

        disliked_tokens = set()
        for text in getattr(avatar, "disliked_items", []):
            disliked_tokens.update(tokenize_for_rerank(text))

        taste_text = " ".join(getattr(avatar, "taste", [])).lower()
        avoid_terms = set()
        if "avoid" in taste_text:
            avoid_terms = set(tokenize_for_rerank(taste_text.split("avoid", 1)[-1]))

        train_weighted_profile = Counter()
        train_categories = Counter()
        train_cities = Counter()
        avatar_id = getattr(avatar, "avatar_id", None)
        if avatar_id is not None:
            if getattr(self.data, "nodrop", False):
                train_item_ids = self.data.train_nodrop_user_list[avatar_id]
            else:
                train_item_ids = self.data.train_user_list[avatar_id]
            for item_id in train_item_ids:
                if item_id not in self.movie_detail.index:
                    continue
                item = self.movie_detail.loc[item_id]
                for token in tokenize_for_rerank(
                    f"{getattr(item, 'title', '')} {getattr(item, 'genres', '')} {getattr(item, 'summary', '')}"
                ):
                    train_weighted_profile[token] += 1.0
                for token in tokenize_for_rerank(getattr(item, "genres", "")):
                    train_categories[token] += 1.0
                city = str(getattr(item, "city", "")).strip().lower()
                if city:
                    train_cities[city] += 1.0

        return {
            "weighted_profile": weighted_profile,
            "profile_categories": profile_categories,
            "disliked_tokens": disliked_tokens,
            "avoid_terms": avoid_terms,
            "profile_weight_sum": sum(weighted_profile.values()),
            "train_weighted_profile": train_weighted_profile,
            "train_profile_weight_sum": sum(train_weighted_profile.values()),
            "train_categories": train_categories,
            "train_category_weight_sum": sum(train_categories.values()),
            "train_cities": train_cities,
            "train_city_weight_sum": sum(train_cities.values()),
        }

    def profile_semantic_score(self, avatar, item):
        item_text = f"{getattr(item, 'title', '')} {getattr(item, 'genres', '')} {getattr(item, 'summary', '')}"
        item_tokens = set(tokenize_for_rerank(item_text))
        if not item_tokens:
            return 0.0
        profile = avatar._rerank_profile
        weighted_profile = profile["weighted_profile"]
        if not weighted_profile:
            return 0.0
        matched = sum(weight for token, weight in weighted_profile.items() if token in item_tokens)
        normalizer = profile["profile_weight_sum"]
        direct_overlap = matched / max(normalizer, 1.0)

        item_categories = set(tokenize_for_rerank(getattr(item, "genres", "")))
        profile_categories = profile["profile_categories"]
        category_overlap = len(item_categories & profile_categories) / max(min(len(profile_categories), 10), 1)

        return min(1.0, 0.65 * direct_overlap * 4.0 + 0.35 * category_overlap)

    def train_history_score(self, avatar, item):
        item_text = f"{getattr(item, 'title', '')} {getattr(item, 'genres', '')} {getattr(item, 'summary', '')}"
        item_tokens = set(tokenize_for_rerank(item_text))
        item_categories = set(tokenize_for_rerank(getattr(item, "genres", "")))
        profile = avatar._rerank_profile

        train_profile = profile.get("train_weighted_profile", Counter())
        direct_overlap = 0.0
        if train_profile and item_tokens:
            matched = sum(weight for token, weight in train_profile.items() if token in item_tokens)
            direct_overlap = min(1.0, 5.0 * matched / max(profile.get("train_profile_weight_sum", 1.0), 1.0))

        train_categories = profile.get("train_categories", Counter())
        category_overlap = 0.0
        if train_categories and item_categories:
            matched_categories = sum(weight for token, weight in train_categories.items() if token in item_categories)
            category_overlap = min(
                1.0,
                matched_categories / max(profile.get("train_category_weight_sum", 1.0) * 0.20, 1.0),
            )

        return min(1.0, 0.55 * direct_overlap + 0.45 * category_overlap)

    def city_match_score(self, avatar, item):
        profile = avatar._rerank_profile
        train_cities = profile.get("train_cities", Counter())
        total = profile.get("train_city_weight_sum", 0.0)
        if not train_cities or total <= 0:
            return 0.0
        city = str(getattr(item, "city", "")).strip().lower()
        return min(1.0, train_cities.get(city, 0.0) / total)

    def diaspora_proxy_score(self, avatar, item):
        item_text = f"{getattr(item, 'title', '')} {getattr(item, 'genres', '')} {getattr(item, 'summary', '')}".lower()
        tokens = set(tokenize_for_rerank(item_text))
        score = 0.0
        score += min(0.65, 0.09 * len(tokens & DIASPORA_PROXY_TERMS))

        ethnic_group = str(getattr(avatar, "ethnic_group", "")).lower()
        if ethnic_group == "hausa":
            if {"halal", "grill", "grilled", "bbq", "barbeque", "rice", "indian", "middle", "eastern"} & tokens:
                score += 0.25
        elif ethnic_group == "igbo":
            if {"bbq", "barbeque", "seafood", "fish", "southern", "soul", "grill", "grilled", "indian", "mexican"} & tokens:
                score += 0.25
        elif ethnic_group == "yoruba":
            if {"spicy", "spice", "szechuan", "mexican", "indian", "seafood", "fish", "bbq", "barbeque", "thai"} & tokens:
                score += 0.25
        else:
            if {"spicy", "spice", "bbq", "barbeque", "indian", "mexican", "thai", "african"} & tokens:
                score += 0.2

        if "price range 1" in item_text or "price range 2" in item_text:
            score += 0.08
        return min(1.0, score)

    def price_match_score(self, avatar, item):
        raw_price = str(getattr(item, "price", "?")).strip()
        try:
            price = int(float(raw_price))
        except (TypeError, ValueError):
            price = 2
        sensitivity = str(getattr(avatar, "price_sensitivity", "medium")).lower()
        table = {
            "high": {1: 1.0, 2: 0.85, 3: 0.25, 4: 0.0},
            "medium": {1: 0.9, 2: 1.0, 3: 0.55, 4: 0.25},
            "low": {1: 0.55, 2: 0.85, 3: 0.9, 4: 0.75},
        }
        return table.get(sensitivity, table["medium"]).get(price, 0.5)

    def social_proof_score(self, item):
        try:
            rating = float(getattr(item, "rating", 0))
        except (TypeError, ValueError):
            rating = 0.0
        try:
            reviews = float(getattr(item, "review_count", 0))
        except (TypeError, ValueError):
            reviews = 0.0
        rating_score = max(0.0, min(1.0, (rating - 3.0) / 2.0))
        review_score = max(0.0, min(1.0, np.log1p(reviews) / np.log1p(3000)))
        return 0.65 * rating_score + 0.35 * review_score

    def rerank_penalty_score(self, avatar, item):
        item_text = f"{getattr(item, 'title', '')} {getattr(item, 'genres', '')} {getattr(item, 'summary', '')}".lower()
        item_tokens = set(tokenize_for_rerank(item_text))
        profile = avatar._rerank_profile
        penalty = 0.0

        disliked_tokens = profile["disliked_tokens"]
        if disliked_tokens and item_tokens & disliked_tokens:
            penalty += 0.35

        avoid_terms = profile["avoid_terms"]
        if avoid_terms & item_tokens:
            penalty += 0.25

        if item_tokens & PRETENTIOUS_OR_LIGHT_TERMS:
            penalty += 0.15
        if str(getattr(avatar, "ethnic_group", "")).lower() == "hausa" and (item_tokens & ALCOHOL_HEAVY_TERMS):
            penalty += 0.3

        raw_price = str(getattr(item, "price", "?")).strip()
        try:
            price = int(float(raw_price))
        except (TypeError, ValueError):
            price = 2
        if str(getattr(avatar, "price_sensitivity", "medium")).lower() == "high" and price >= 3:
            penalty += 0.35
        return min(1.0, penalty)
    
    def page_generator(self, avatar_id):
        """
        generate one page items for one avatar
        """
        i = 0
        while (i+1)*self.items_per_page < self.data.n_items:
            yield self.full_rankings[avatar_id][i*self.items_per_page:(i+1)*self.items_per_page]
            i += 1

    def build_page_behavior(self, avatar_id, page_num, id_on_page, movies_on_page, response):
        parsed = parse_page_response(response, self.movie_detail, id_on_page)
        if parsed["parse_failed"]:
            self.parse_failed_pages[avatar_id].append(page_num)
        else:
            self.parse_valid_pages[avatar_id].append(page_num)
        if parsed["parse_partial"]:
            self.parse_partial_pages[avatar_id].append(page_num)
        if parsed["errors"]:
            self.parse_errors[avatar_id][page_num] = parsed["errors"]

        align_yes_rows = [row for row in parsed["align_rows"] if row["align"] == "yes"]
        like_rows = [row for row in parsed["rating_rows"] if row["rating"] == 5.0]
        watched_movies = [row["title"] for row in parsed["rating_rows"]]
        rating_values = [row["rating"] for row in parsed["rating_rows"]]
        new_train = list(
            dict.fromkeys(
                [row["item_id"] for row in align_yes_rows]
                + [row["item_id"] for row in like_rows]
            )
        )
        ground_truth = [
            id_on_page[idx]
            for idx, _movie in enumerate(movies_on_page)
            if id_on_page[idx] in self.ground_truth_items(avatar_id)
        ]
        natural_relevance = self.natural_relevance_items(avatar_id)
        expanded_ground_truth = [
            id_on_page[idx]
            for idx, _movie in enumerate(movies_on_page)
            if id_on_page[idx] in natural_relevance
        ]

        info_on_page = {
            "page": page_num,
            "ground_truth": ground_truth,
            "natural_relevance": expanded_ground_truth,
            "recommended_id": id_on_page,
            "recommended": [str(movie.title) for movie in movies_on_page],
            "align_id": [row["item_id"] for row in align_yes_rows],
            "like_id": [row["item_id"] for row in like_rows],
            "watch_id": [row["item_id"] for row in parsed["rating_rows"]],
            "watched": watched_movies,
            "rating_id": watched_movies,
            "rating": rating_values,
            "feeling": [row["feeling"] for row in parsed["rating_rows"]],
            "parse_failed": parsed["parse_failed"],
            "parse_partial": parsed["parse_partial"],
            "parse_errors": parsed["errors"],
        }
        average_rating = nanmean_or_nan(rating_values)
        perf = (len(set(new_train) & set(ground_truth)), len(new_train), len(ground_truth))
        expanded_perf = (len(set(new_train) & set(expanded_ground_truth)), len(new_train), len(expanded_ground_truth))
        return info_on_page, new_train, watched_movies, average_rating, perf, expanded_perf, parsed

    def validate_all_avatars(self):
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.precision_list = []
        self.recall_list = []
        self.accuracy_list = []
        self.f1_list = []
        self.start_time = time.time()

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=100)
        tasks = []

        t1 = time.time()
        for avatar_id in self.simulated_avatars_id:
            tasks.append(self.async_validate_one_avatar(avatar_id, loop, executor))
        loop.run_until_complete(asyncio.wait(tasks))
        t2 = time.time()
        print(f"Time cost: {t2-t1}s")

        print("precision_list", self.precision_list)
        print("recall_list", self.recall_list)
        print("accuracy_list", self.accuracy_list)
        print("f1_list", self.f1_list)

        with open(self.storage_base_path + "/validation_metrics.txt", 'w') as f:
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Average precision: {np.mean(self.precision_list)}\n")
            f.write(f"Average recall: {np.mean(self.recall_list)}\n")
            f.write(f"Average accuracy: {np.mean(self.accuracy_list)}\n")
            f.write(f"Average f1: {np.mean(self.f1_list)}\n")

    async def async_validate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        validate the effectiveness of the model for one avatar
        avatar_id: the id of the simulated avatar
        """
        avatar_ = self.avatars[avatar_id]
        train_list, val_list, test_list = self.data.train_user_list[avatar_id], self.data.valid_user_list[avatar_id], self.data.test_user_list[avatar_id]

        # Take the union for calculating precision.
        all_items = list(range(self.data.n_items))
        observed_items = list(set(train_list) | set(val_list) | set(test_list))
        selection_candidates = list(set(val_list) | set(test_list))
        unobserved_items = list(set(all_items) - set(observed_items))
        # Pick 5 randomly from the test_list.
        min_val = min(len(selection_candidates), 20//(self.val_ratio+1))
        print(len(selection_candidates), 10)

        test_observed_items = np.random.choice(selection_candidates, min_val, replace=False)
        test_unobserved_items = np.random.choice(unobserved_items, int(min_val*self.val_ratio), replace=False)

        print("test_all", test_observed_items, test_unobserved_items)

        forced_items_ids = np.concatenate((test_observed_items, test_unobserved_items))
        # Randomly shuffle.
        np.random.shuffle(forced_items_ids)

        print("forced_items_ids", forced_items_ids)

        forced_items = [self.movie_detail.loc[idx] for idx in forced_items_ids]

        truth_tmp = [self.movie_detail.loc[idx] for idx in test_observed_items]
        truth_list = ["<- " + item.title + " ->" 
                            + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                            + " <- Summary:" + item.summary + " ->" + "\n"
                            for item in truth_tmp]
        truth_str = ''.join(truth_list)
        cprint(truth_str, color='white', attrs=['bold'])

        recommended_items = ["<- " + item.title + " ->" 
                            + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                            + " <- Summary:" + item.summary + " ->" + "\n"
                            for item in forced_items]
        recommended_items_str = ''.join(recommended_items)

        validation_rounds = max(1, int(getattr(self.args, "validation_rounds", 1)))
        yes_votes = Counter()
        parsed_rounds = 0
        responses = []
        for round_idx in range(validation_rounds):
            response = await loop.run_in_executor(executor, avatar_.reaction_to_forced_items, recommended_items_str)
            responses.append(response)
            cprint(response, color='yellow', attrs=None)

            watch_choices, parse_errors = parse_forced_watch_response(response, self.movie_detail, forced_items_ids)
            if parse_errors:
                print("validation_parse_errors", parse_errors)
            if watch_choices:
                parsed_rounds += 1
            for choice in watch_choices:
                if choice["watch"] == "yes":
                    yes_votes[choice["item_id"]] += 1

        vote_threshold = validation_rounds // 2 + 1
        like_item_ids = [
            item_id
            for item_id in forced_items_ids
            if yes_votes[item_id] >= vote_threshold
        ]
        like_items = [
            (item_id, str(self.movie_detail.loc[item_id].title), yes_votes[item_id])
            for item_id in like_item_ids
        ]
        print("like_restaurants", like_items)
        if validation_rounds > 1:
            print(f"validation_rounds {validation_rounds}; parsed_rounds {parsed_rounds}; vote_threshold {vote_threshold}")

        pred = np.array([1 if idx in like_item_ids else 0 for idx in forced_items_ids])
        true = np.array([1 if idx in test_observed_items else 0 for idx in forced_items_ids])

        # Calculate precision.
        precision = get_precision(true, pred)
        print("precision", precision)
        # Calculate recall.
        recall = get_recall(true, pred)
        print("recall", recall)
        accuracy = get_accuracy(true, pred)
        print("accuracy", accuracy)
        f1 = get_f1(true, pred)
        print("f1", f1)

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.accuracy_list.append(accuracy)
        self.f1_list.append(f1)

        vars.global_finished_users += 1

    def simulate_all_avatars(self):
        """
        excute the simulation for all avatars
        """
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.start_time = time.time()
        if(self.execution_mode == 'serial'):
            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                self.simulate_one_avatar(avatar_id)
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

        elif(self.execution_mode == 'parallel'):
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=500)
            tasks = []

            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                tasks.append(self.async_simulate_one_avatar(avatar_id, loop, executor))
            loop.run_until_complete(asyncio.wait(tasks))
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

    async def async_simulate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        start_time = time.time()
        time_local = time.localtime(start_time)
        l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start}. User {avatar_id} starts simulation.\n")

        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        user_behavior_dict = {}
        user_interview_dict = {}
        while not avatar_.exit_flag:
            i += 1
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break
            id_on_page = self.maybe_inject_ground_truth(id_on_page, avatar_id, i)
            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page] # movie_detail.csv
            recommended_items = ["<- " + item.title + " ->" 
                            # + " <- Genres: " + (',').join(list(item.genres.split('|'))) + " ->"
                            + " <- History ratings: " + str(round(item.rating,2)) + " ->" 
                            + " <- Summary: " + item.summary + " ->" + "\n"
                            for item in movies_on_page]
            
            if(self.add_advert):
                #store_path = op.join(f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id", f"avatar{avatar_id}_{i}.txt")
                store_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                if not self.display_advert:
                    recommended_items[0], id_on_page, movies_on_page = self.display_only_adver_item(store_path, avatar_id, i, id_on_page, movies_on_page)
                else:
                    recommended_items[0], id_on_page, movies_on_page = self.display_item_with_adver(store_path, avatar_id, i, id_on_page, movies_on_page)


            recommended_items_str = ''.join(recommended_items)
            print(recommended_items_str)

            # Please write down the recommended information.
            avatar_.write_log(f"\n=============    Recommendation Page {i}    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.ground_truth_items(avatar_id)):
                    avatar_.write_log(f"== (√) {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}")
            avatar_.write_log(f"=============          End Page {i}        =============\n")

            # As a translator, I will translate the Chinese sentence you sent me into English. I do not need to understand the meaning of the content to provide a response.
            avatar_.write_log(f"\n==============    Avatar {avatar_.avatar_id} Response {i}   =============")


            # @ most important Waiting for user response.
            response = await loop.run_in_executor(executor, avatar_.reaction_to_recommended_items, recommended_items_str, i)

            info_on_page, new_train, watched_movies, average_rating, perf, expanded_perf, parsed = self.build_page_behavior(
                avatar_id, i, id_on_page, movies_on_page, response
            )
            if self.add_advert and parsed["align_rows"] and parsed["align_rows"][0]["align"] == "yes":
                self.clicked_adverts += 1
            user_behavior_dict[i] = info_on_page

            self.new_train_dict[avatar_id].extend(new_train)
            self.n_likes[avatar_id].append(len(new_train))
            self.ratings[avatar_id].append(average_rating)
            self.watch[avatar_id].extend([movie for movie in watched_movies])

            self.perf_per_page[avatar_id].append(perf)
            self.expanded_perf_per_page[avatar_id].append(expanded_perf)

            vars.global_finished_pages += 1

            # @ Force exit if the number of pages exceeds the maximum limit.
            if(i >= self.max_pages):
                avatar_.exit_flag = True
        
        interview_response = avatar_.response_to_question("Do you feel satisfied with the recommender system you have just interacted? Rate this recommender system from 1-10 and give explanation.\n Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should included in the REASON part.", remember=False)
        # Extract RAING and REASON using re.
        pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        # pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        matches_interview = re.findall(r'(?<=RATING:|REASON:).*', interview_response)
        user_interview_dict['interview'] = matches_interview
        print(matches_interview)
        self.exit_page[avatar_id] = i
        self.finished_num += 1
        self.remaining_users.remove(avatar_id)
        remaining = ", ".join([str(u) for u in self.remaining_users])

        end_time = time.time()
        time_local = time.localtime(end_time)
        l_end = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        vars.global_finished_users += 1
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start} End: {l_end}. User {avatar_id} finished after {i} pages. [{self.finished_num} / {self.n_avatars}]. Total token cost: {round(self.avatars[avatar_id].memory.user_k_tokens, 2)}k. Taking {round(time.time() - start_time, 2)}s\n")
            f.write(f"Remaining users: {remaining}\n")

        # @ Save the behavior of each individual.
        behavior_path = self.storage_base_path+ "/behavior"
        if not os.path.exists(behavior_path):
            os.makedirs(behavior_path)
        with open(behavior_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_behavior_dict, f)

        interview_path = self.storage_base_path+ "/interview"
        if not os.path.exists(interview_path):
            os.makedirs(interview_path)
        with open(interview_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_interview_dict, f)

    def simulate_one_avatar(self, avatar_id):
        """
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        # print("\nIs simulating avatar {}".format(avatar_id))
        start_time = time.time()
        time_local = time.localtime(start_time)
        l_start = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        user_behavior_dict = {}
        user_interview_dict = {}
        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        while not avatar_.exit_flag:
        # for i in range(2):
            i += 1
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break
            id_on_page = self.maybe_inject_ground_truth(id_on_page, avatar_id, i)

            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page]
            recommended_items = ["<- " + item.title + " ->"
                            + " <- History ratings: " + str(round(item.rating,2)) + " ->"
                            + " <- Summary: " + item.summary + " ->" + "\n"
                            for item in movies_on_page]
            recommended_items_str = ''.join(recommended_items)
            print(recommended_items_str)

            avatar_.write_log(f"=============    Recommendation Page {i}    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.ground_truth_items(avatar_id)):
                    avatar_.write_log(f"== (√) {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}")
            avatar_.write_log(f"=============          End Page {i}        =============")
            avatar_.write_log("")
            
            #@ most important
            response = avatar_.reaction_to_recommended_items(recommended_items_str, i)

            avatar_.write_log("")
            avatar_.write_log("=============    Avatar Response    =============")
            avatar_.write_log(response, color='yellow', attrs=None)

            info_on_page, new_train, watched_movies, average_rating, perf, expanded_perf, _parsed = self.build_page_behavior(
                avatar_id, i, id_on_page, movies_on_page, response
            )
            user_behavior_dict[i] = info_on_page

            self.new_train_dict[avatar_id].extend(new_train)
            self.n_likes[avatar_id].append(len(new_train))
            self.ratings[avatar_id].append(average_rating)
            self.watch[avatar_id].extend([movie for movie in watched_movies])

            self.perf_per_page[avatar_id].append(perf)
            self.expanded_perf_per_page[avatar_id].append(expanded_perf)

            vars.global_finished_pages += 1

            if i >= self.max_pages:
                avatar_.exit_flag = True

        interview_response = avatar_.response_to_question("Do you feel satisfied with the recommender system you have just interacted? Rate this recommender system from 1-10 and give explanation.\n Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should included in the REASON part.", remember=False)
        matches_interview = re.findall(r'(?<=RATING:|REASON:).*', interview_response)
        user_interview_dict['interview'] = matches_interview
        print(matches_interview)

        self.exit_page[avatar_id] = i
        self.finished_num += 1
        if avatar_id in self.remaining_users:
            self.remaining_users.remove(avatar_id)
        remaining = ", ".join([str(u) for u in self.remaining_users])

        end_time = time.time()
        time_local = time.localtime(end_time)
        l_end = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        vars.global_finished_users += 1
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start} End: {l_end}. User {avatar_id} finished after {i} pages. [{self.finished_num} / {self.n_avatars}]. Total token cost: {round(self.avatars[avatar_id].memory.user_k_tokens, 2)}k. Taking {round(time.time() - start_time, 2)}s\n")
            f.write(f"Remaining users: {remaining}\n")

        behavior_path = self.storage_base_path + "/behavior"
        if not os.path.exists(behavior_path):
            os.makedirs(behavior_path)
        with open(behavior_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_behavior_dict, f)

        interview_path = self.storage_base_path + "/interview"
        if not os.path.exists(interview_path):
            os.makedirs(interview_path)
        with open(interview_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_interview_dict, f)
    
    def parse_response(self, response):
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*FEELING:\s*(.*?)\s*RATING:\s*(\d)')
        pattern = re.compile(rf'{ITEM_LABEL_PATTERN}:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)', re.IGNORECASE)
        matches = re.findall(pattern, response)

        watched_movies, watched_movies_contain_id = [], []

        for idx, (movie_title, watch, reason, rating, feeling) in enumerate(matches):
            if(self.add_advert and idx == 0 and watch.strip(';') == 'yes'): # If the first one has an advertisement and the user clicked on it.
                self.clicked_adverts += 1
            if(watch.strip(';') == 'yes'):
                watched_movies.append(movie_title.strip(';'))
            print(movie_title, watch, reason, rating, feeling)
        return response

    def display_only_adver_item(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        try:
            with open(store_path, 'r') as f1:
                random_key = int(f1.read())
        except:
            try:
                store_path_minus_1 = op.join(store_path, f"avatar{avatar_id}_{i-1}.txt")
                with open(store_path_minus_1, 'r') as f2:
                    random_key = int(f2.read())
            except:
                store_path_minus_2 = op.join(store_path, f"avatar{avatar_id}_{i-2}.txt")
                with open(store_path_minus_2, 'r') as f3:
                    random_key = int(f3.read())
                    try:
                        store_path_minus_3 = op.join(store_path, f"avatar{avatar_id}_{i-3}.txt")
                        with open(store_path_minus_3, 'r') as f4:
                            random_key = int(f4.read())
                    except:
                            store_path_minus_4 = op.join(store_path, f"avatar{avatar_id}_{i-4}.txt")
                            with open(store_path_minus_4, 'r') as f5:
                                random_key = int(f5.read())


        self.total_adverts += 1
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        adver_information = self.advert[random_key]

        return ( "<- " + adver_information['title'] + " ->" 
                                + " <- History ratings:" + str(round(adver_information['rating'], 2)) + " ->"
                                + " <- Summary:" + adver_information['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    def display_item_with_adver(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        random_key = np.random.choice(list(self.advert.keys()))
        self.total_adverts += 1
        random_advert = self.advert[random_key]
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        advert_item_id = random_key

        with open(store_path, 'w') as f:
            f.write(f"{advert_item_id}")
        
        return ( self.advert_word 
                + "<- " + random_advert['title'] + " ->" 
                + "<- " + random_advert['review'] + " ->"
                + " <- History ratings:" + str(round(random_advert['rating'], 2)) + " ->" 
                + " <- Summary:" + random_advert['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    def save_results(self):
        """
        save the results of the simulation
        """
        # if(self.n_avatars == self.data.n_users):
        def save_user_dict_to_txt(user_dict, base_path, filename):
            with open(base_path + filename, 'w') as f:
                for u, v in user_dict.items():
                    f.write(str(int(u)))
                    for i in v:
                        f.write(' ' + str(int(i)))
                    f.write('\n')

        # save_path = f"datasets/{self.dataset}_{self.modeltype}/cf_data/"
        save_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/"
        save_user_dict_to_txt(self.new_train_dict, save_path, 'train.txt')

        # @ Save overall evaluation indicators.
        # Average number of clicks per user
        cprint("Number of likes", color='green', attrs=['bold'])
        cprint(self.n_likes, color='green', attrs=['bold'])
        average_n_likes = {avatar_id:np.mean(n_likes) for avatar_id, n_likes in self.n_likes.items()}
        cprint(average_n_likes, color='green', attrs=['bold'])

        overall_n_likes = np.mean(list(average_n_likes.values()))
        cprint(f"\nOverall number of likes: {overall_n_likes}", color='green', attrs=['bold'])

        # Average satisfaction
        cprint("\nRatings", color='green', attrs=['bold'])
        cprint(self.ratings, color='green', attrs=['bold'])
        average_ratings = {avatar_id:nanmean_or_nan(ratings) for avatar_id, ratings in self.ratings.items()}
        cprint(average_ratings, color='green', attrs=['bold'])
        overall_rating = nanmean_or_nan(list(average_ratings.values()))
        cprint(f"\nAverage valid page rating: {overall_rating}", color='green', attrs=['bold'])

        # @ Save average click-through rate
        average_click_rate = {avatar_id:len(movies)/(self.max_pages*self.items_per_page) for avatar_id, movies in self.watch.items()}
        cprint(f"\nAverage click rate: {average_click_rate}", color='green', attrs=['bold'])
        overall_click_rate = np.mean(list(average_click_rate.values()))
        cprint(f"\nOverall click rate: {overall_click_rate}", color='green', attrs=['bold'])
        exposure_adjusted_click_rate = {
            avatar_id: len(self.watch[avatar_id]) / max(self.exit_page.get(avatar_id, 0) * self.items_per_page, 1)
            for avatar_id in self.watch
        }
        overall_exposure_adjusted_click_rate = np.mean(list(exposure_adjusted_click_rate.values()))
        cprint(f"\nExposure-adjusted click rate: {exposure_adjusted_click_rate}", color='cyan', attrs=['bold'])
        cprint(f"\nOverall exposure-adjusted click rate: {overall_exposure_adjusted_click_rate}", color='cyan', attrs=['bold'])

        interview_satisfaction = {}
        for avatar_id in self.simulated_avatars_id:
            interview_path = self.storage_base_path + f"/interview/{avatar_id}.pkl"
            if not os.path.exists(interview_path):
                continue
            with open(interview_path, "rb") as f:
                interview = pickle.load(f)
            rating = parse_interview_rating(interview)
            if rating is not None:
                interview_satisfaction[avatar_id] = rating / 10.0
        true_satisfaction = np.mean(list(interview_satisfaction.values())) if interview_satisfaction else 0.0
        cprint(f"\nInterview satisfaction: {interview_satisfaction}", color='cyan', attrs=['bold'])
        cprint(f"\nTrue satisfaction score: {true_satisfaction}", color='cyan', attrs=['bold'])

        total_parse_valid = sum(len(pages) for pages in self.parse_valid_pages.values())
        total_parse_partial = sum(len(pages) for pages in self.parse_partial_pages.values())
        total_parse_failed = sum(len(pages) for pages in self.parse_failed_pages.values())
        total_parse_pages = total_parse_valid + total_parse_failed
        parse_valid_rate = total_parse_valid / total_parse_pages if total_parse_pages > 0 else 0.0
        parse_failure_rate = total_parse_failed / total_parse_pages if total_parse_pages > 0 else 0.0
        parse_partial_rate = total_parse_partial / total_parse_pages if total_parse_pages > 0 else 0.0
        cprint(
            f"\nParse-valid pages: {total_parse_valid}/{total_parse_pages} "
            f"(failed={total_parse_failed}, partial={total_parse_partial})",
            color='cyan',
            attrs=['bold'],
        )
        cprint(f"Parse-valid rate: {parse_valid_rate}", color='cyan', attrs=['bold'])

        # overall_click_rate = np.mean(list(average_ratings.values()))
        # cprint(f"\nOverall satisfaction: {overall_click_rate}", color='green', attrs=['bold'])

        # Average exit page
        mean_exit_page = np.mean(list(self.exit_page.values()))
        cprint("\nExit pages", color='green', attrs=['bold'])
        cprint(self.exit_page, color='green', attrs=['bold'])
        cprint(f"Average exit page: {mean_exit_page}", color='green', attrs=['bold'])

        # Average precision and recall
        cprint("\nPrecision and recall", color='green', attrs=['bold'])
        cprint(self.perf_per_page, color="green", attrs=['bold'])
        total_perf = {avatar_id:[sum([i for i, j, k in perf_per_page]), sum([j for i, j, k in perf_per_page]), sum([k for i, j, k in perf_per_page])] for avatar_id, perf_per_page in self.perf_per_page.items()}
        total_recall_precision = {
            avatar_id: (
                perf[0] / max(perf[1], 1),
                perf[0] / perf[2] if perf[2] > 0 else np.nan,
            )
            for avatar_id, perf in total_perf.items()
        }
        exposed_users = {
            avatar_id: metrics
            for avatar_id, metrics in total_recall_precision.items()
            if not np.isnan(metrics[1])
        }
        exposed_user_count = len(exposed_users)
        exposed_user_hit_count = sum(
            1 for avatar_id, perf in total_perf.items()
            if perf[2] > 0 and perf[0] > 0
        )
        total_hits = sum(perf[0] for perf in total_perf.values())
        total_selected = sum(perf[1] for perf in total_perf.values())
        total_exposed_ground_truth = sum(perf[2] for perf in total_perf.values())
        cprint(total_perf, color="green", attrs=['bold'])
        cprint(total_recall_precision, color="green", attrs=['bold'])
        average_precision = np.mean([metrics[0] for avatar_id, metrics in total_recall_precision.items()])
        average_recall = np.mean([metrics[1] for avatar_id, metrics in exposed_users.items()]) if exposed_users else 0.0
        micro_precision = total_hits / max(total_selected, 1)
        micro_recall = total_hits / total_exposed_ground_truth if total_exposed_ground_truth > 0 else 0.0
        grounded_recognition_rate = total_hits / total_exposed_ground_truth if total_exposed_ground_truth > 0 else 0.0
        exposed_user_hit_rate = exposed_user_hit_count / exposed_user_count if exposed_user_count > 0 else 0.0
        cprint(f"Ground-truth items exposed: {total_exposed_ground_truth}", color="yellow", attrs=['bold'])
        cprint(f"Precision: {average_precision}  Recall over exposed users: {average_recall}", color="green", attrs=['bold'])
        cprint(f"Micro precision: {micro_precision}  Micro recall over exposed items: {micro_recall}", color="green", attrs=['bold'])
        cprint(f"Grounded recognition rate: {grounded_recognition_rate}", color="cyan", attrs=['bold'])
        cprint(f"Exposed-user hit rate: {exposed_user_hit_rate} ({exposed_user_hit_count}/{exposed_user_count})", color="cyan", attrs=['bold'])

        expanded_total_perf = {
            avatar_id: [
                sum([i for i, j, k in perf_per_page]),
                sum([j for i, j, k in perf_per_page]),
                sum([k for i, j, k in perf_per_page]),
            ]
            for avatar_id, perf_per_page in self.expanded_perf_per_page.items()
        }
        expanded_recall_precision = {
            avatar_id: (
                perf[0] / max(perf[1], 1),
                perf[0] / perf[2] if perf[2] > 0 else np.nan,
            )
            for avatar_id, perf in expanded_total_perf.items()
        }
        expanded_exposed_users = {
            avatar_id: metrics
            for avatar_id, metrics in expanded_recall_precision.items()
            if not np.isnan(metrics[1])
        }
        expanded_total_hits = sum(perf[0] for perf in expanded_total_perf.values())
        expanded_total_selected = sum(perf[1] for perf in expanded_total_perf.values())
        expanded_total_exposed = sum(perf[2] for perf in expanded_total_perf.values())
        expanded_precision = np.mean([metrics[0] for metrics in expanded_recall_precision.values()]) if expanded_recall_precision else 0.0
        expanded_recall = np.mean([metrics[1] for metrics in expanded_exposed_users.values()]) if expanded_exposed_users else 0.0
        expanded_micro_precision = expanded_total_hits / max(expanded_total_selected, 1)
        expanded_micro_recall = expanded_total_hits / expanded_total_exposed if expanded_total_exposed > 0 else 0.0
        if getattr(self.args, "natural_relevance_scope", "heldout") != "heldout":
            cprint("\nExpanded natural relevance precision and recall", color="cyan", attrs=['bold'])
            cprint(expanded_total_perf, color="cyan", attrs=['bold'])
            cprint(expanded_recall_precision, color="cyan", attrs=['bold'])
            cprint(f"Expanded relevant items exposed: {expanded_total_exposed}", color="yellow", attrs=['bold'])
            cprint(f"Expanded precision: {expanded_precision}  Expanded recall over exposed users: {expanded_recall}", color="cyan", attrs=['bold'])
            cprint(f"Expanded micro precision: {expanded_micro_precision}  Expanded micro recall over exposed items: {expanded_micro_recall}", color="cyan", attrs=['bold'])
        # metrics_path = self.storage_base_path + "/metrics.txt"
        total_k_tokens = sum([self.avatars[i].memory.user_k_tokens for i in range(self.n_avatars)])

        # Effective advertising rate
        if(self.add_advert):
            cprint("\nAdvert", color='green', attrs=['bold'])
            cprint(f"Total advert: {self.total_adverts}", color='green', attrs=['bold'])
            cprint(f"Clicked advert: {self.clicked_adverts}", color='green', attrs=['bold'])
            cprint(f"Advert click rate: {self.clicked_adverts/self.total_adverts}", color='green', attrs=['bold'])

        end_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        with open(self.storage_base_path + "/metrics.txt", 'w') as f:
            f.write(f"Finished time: {end_time}\n")
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Hybrid rerank: {getattr(self.args, 'hybrid_rerank', False)}\n")
            if getattr(self.args, "hybrid_rerank", False):
                f.write(f"Rerank pool size: {getattr(self.args, 'rerank_pool_size', 100)}\n")
                f.write(f"Semantic candidate expansion: {getattr(self.args, 'semantic_candidate_expansion', False)}\n")
                f.write(f"Semantic candidate pool size: {getattr(self.args, 'semantic_candidate_pool_size', 200)}\n")
                f.write(f"Semantic candidate CF floor: {getattr(self.args, 'semantic_candidate_cf_floor', 0.05)}\n")
                f.write(
                    "Rerank weights: "
                    f"cf={getattr(self.args, 'rerank_cf_weight', 0.55)}, "
                    f"semantic={getattr(self.args, 'rerank_semantic_weight', 0.30)}, "
                    f"diaspora={getattr(self.args, 'rerank_diaspora_weight', 0.25)}, "
                    f"history={getattr(self.args, 'rerank_history_weight', 0.0)}, "
                    f"city={getattr(self.args, 'rerank_city_weight', 0.0)}, "
                    f"price={getattr(self.args, 'rerank_price_weight', 0.12)}, "
                    f"social={getattr(self.args, 'rerank_social_weight', 0.08)}, "
                    f"penalty={getattr(self.args, 'rerank_penalty_weight', 0.35)}\n"
                )
            f.write(f"Average recall: {average_recall}\n")
            f.write(f"Average presion: {average_precision}\n")
            f.write(f"Micro precision: {micro_precision}\n")
            f.write(f"Micro recall over exposed items: {micro_recall}\n")
            f.write(f"Ground-truth items exposed: {total_exposed_ground_truth}\n")
            f.write(f"Grounded recognition rate: {grounded_recognition_rate}\n")
            f.write(f"Exposed-user hit rate: {exposed_user_hit_rate}\n")
            f.write(f"Exposed users with hits: {exposed_user_hit_count}/{exposed_user_count}\n")
            f.write(f"Natural relevance scope: {getattr(self.args, 'natural_relevance_scope', 'heldout')}\n")
            f.write(f"Allow train recommendations: {getattr(self.args, 'allow_train_recommendations', False)}\n")
            f.write(f"Expanded average recall: {expanded_recall}\n")
            f.write(f"Expanded average precision: {expanded_precision}\n")
            f.write(f"Expanded micro precision: {expanded_micro_precision}\n")
            f.write(f"Expanded micro recall over exposed items: {expanded_micro_recall}\n")
            f.write(f"Expanded relevant items exposed: {expanded_total_exposed}\n")
            f.write(f"Total k tokens: {round(total_k_tokens, 2)}k tokens\n")
            f.write(f"Total cost: {round(total_k_tokens*0.0018, 2)} \n")
            # f.write(f"Average precision: {}")
            f.write(f"Maximum exit page: {self.max_pages}\n")
            f.write(f"Overall click rate: {overall_click_rate}\n")
            f.write(f"Overall exposure-adjusted click rate: {overall_exposure_adjusted_click_rate}\n")
            f.write(f"True satisfaction score: {true_satisfaction}\n")
            f.write(f"Average valid page rating: {overall_rating}\n")
            f.write(f"Average number of likes: {overall_n_likes}\n")
            f.write(f"Average exit page: {mean_exit_page}\n")
            f.write(f"Parse-valid pages: {total_parse_valid}/{total_parse_pages}\n")
            f.write(f"Parse-valid rate: {parse_valid_rate}\n")
            f.write(f"Parse failure rate: {parse_failure_rate}\n")
            f.write(f"Parse partial rate: {parse_partial_rate}\n")
            f.write(f"Parse failed pages: {self.parse_failed_pages}\n")
            f.write(f"Parse partial pages: {self.parse_partial_pages}\n")
            f.write(f"Parse errors: {self.parse_errors}\n")
            if(self.add_advert):
                f.write(f"Total advert: {self.total_adverts}\n")
                f.write(f"Clicked advert: {self.clicked_adverts}\n")
                f.write(f"Advert click rate: {self.clicked_adverts/self.total_adverts}\n")
