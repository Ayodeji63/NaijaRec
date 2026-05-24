import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


Provider = Literal["auto", "lightgcn"]

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ARTIFACT_ROOT = PROJECT_ROOT / "model_artifacts"
if not ARTIFACT_ROOT.exists():
    ARTIFACT_ROOT = PROJECT_ROOT


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


CATALOG_PATH = ARTIFACT_ROOT / "datasets/yelp-kimi/simulation/movie_detail.csv"
PERSONA_PATH = ARTIFACT_ROOT / "datasets/yelp-kimi/simulation/all_personas_like_modify.csv"
KG_PATH = ARTIFACT_ROOT / "datasets/yelp-kimi/simulation/knowledge_graph_triples.csv"
BASE_RANKINGS_PATH = first_existing_path(
    ARTIFACT_ROOT / "storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/full_rankings_821.npy",
    ARTIFACT_ROOT / "storage/yelp-kimi/LightGCN/Test/rankings/full_rankings_70.npy",
)
HYBRID_RANKINGS_PATH = first_existing_path(
    ARTIFACT_ROOT / "storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/offline_exposure_small_821_best_rankings.npy",
    ARTIFACT_ROOT / "storage/yelp-kimi/LightGCN/Test/rankings/offline_balanced_rerank_70_best_rankings.npy",
)
CHECKPOINT_PATH = (
    ARTIFACT_ROOT
    / "recommenders/weights/yelp-kimi/LightGCN/"
    / "0515_yelp_lgn_t0p0_Ks=20_patience=10_n_layers=2_batch_size=1024_neg_sample=1_lr=0.0005/"
    / "epoch=84.checkpoint.pth.tar"
)


GENERIC_TOKENS = {
    "restaurant", "restaurants", "food", "bar", "bars", "nightlife", "event", "events",
    "services", "planning", "venues", "shopping", "active", "life", "specialty",
    "traditional", "american", "new", "delivery", "caterers", "catering", "coffee",
    "tea", "the", "and", "with", "for", "spot", "place",
}

DIASPORA_TERMS = {
    "african": 0.60,
    "caribbean": 0.65,
    "jamaican": 0.65,
    "southern": 0.40,
    "soul": 0.45,
    "bbq": 0.55,
    "barbeque": 0.55,
    "barbecue": 0.55,
    "grill": 0.50,
    "grilled": 0.50,
    "spicy": 0.70,
    "spice": 0.70,
    "pepper": 0.80,
    "szechuan": 0.70,
    "chinese": 0.45,
    "indian": 0.65,
    "thai": 0.60,
    "vietnamese": 0.45,
    "malaysian": 0.50,
    "mexican": 0.55,
    "latin": 0.45,
    "cuban": 0.45,
    "brazilian": 0.45,
    "seafood": 0.45,
    "fish": 0.45,
    "mediterranean": 0.35,
    "middle": 0.35,
    "eastern": 0.35,
    "halal": 0.45,
    "rice": 0.40,
    "ramen": 0.35,
    "noodles": 0.35,
    "tacos": 0.35,
    "taqueria": 0.40,
    "chicken": 0.35,
}

ALCOHOL_TERMS = {"beer", "wine", "wines", "cocktail", "cocktails", "whiskey", "brewery", "pub", "pubs", "bar", "bars", "nightlife"}
LIGHT_TERMS = {"dessert", "desserts", "frozen", "yogurt", "bakery", "bakeries", "coffee", "tea", "gelato"}


class ProductDetails(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    category: str = Field("Restaurant", max_length=220)
    city: str = Field("", max_length=120)
    price_range: str = Field("2", max_length=20)
    rating: Optional[float] = Field(None, ge=0, le=5)
    review_count: Optional[int] = Field(None, ge=0)
    summary: str = Field("", max_length=1600)
    image_url: str = Field("", max_length=600)


class PersonaInput(BaseModel):
    cultural_group: str = Field("Nigerian diaspora", max_length=120)
    taste_profile: str = Field(..., min_length=1, max_length=1800)
    price_sensitivity: Literal["low", "medium", "high"] = "medium"
    dietary_context: str = Field("", max_length=500)
    dining_context: str = Field("Dinner after work", max_length=500)
    review_style: Literal["balanced", "yoruba", "igbo", "hausa", "urban"] = "balanced"


class ReviewRequest(BaseModel):
    persona: PersonaInput
    products: List[ProductDetails] = Field(..., min_length=1, max_length=8)
    provider: Provider = "auto"
    max_tokens: int = Field(900, ge=200, le=2000)

    @field_validator("products")
    @classmethod
    def unique_product_names(cls, products: List[ProductDetails]) -> List[ProductDetails]:
        seen = set()
        for product in products:
            key = product.name.strip().casefold()
            if key in seen:
                raise ValueError(f"Duplicate product name: {product.name}")
            seen.add(key)
        return products


class ProductReview(BaseModel):
    name: str
    rating: float = Field(..., ge=1, le=5)
    sentiment: Literal["positive", "mixed", "negative"]
    review: str
    reasons: List[str] = Field(default_factory=list)
    grounded_cues: List[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    provider: str
    used_fallback: bool
    overall_summary: str
    reviews: List[ProductReview]
    warnings: List[str] = Field(default_factory=list)


def tokenize(value: object) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(value).lower())
        if token and token not in GENERIC_TOKENS
    ]


def normalize_title(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def product_text(product: ProductDetails) -> str:
    return " ".join([product.name, product.category, product.city, product.price_range, product.summary])


def split_profile_list(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in re.split(r";|\|", str(value)) if part.strip()]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class NaijaRecModel:
    def __init__(self) -> None:
        missing = [
            str(path)
            for path in [CATALOG_PATH, PERSONA_PATH, KG_PATH, BASE_RANKINGS_PATH, HYBRID_RANKINGS_PATH, CHECKPOINT_PATH]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError("NaijaRec model artifacts are missing from the Docker image: " + "; ".join(missing))

        self.catalog = pd.read_csv(CATALOG_PATH).set_index("movie_id", drop=False)
        self.kg = pd.read_csv(KG_PATH)
        self.base_rankings = np.load(BASE_RANKINGS_PATH)
        self.hybrid_rankings = np.load(HYBRID_RANKINGS_PATH)
        deployed_avatars = min(len(self.base_rankings), len(self.hybrid_rankings))
        self.base_rankings = self.base_rankings[:deployed_avatars]
        self.hybrid_rankings = self.hybrid_rankings[:deployed_avatars]
        self.personas = pd.read_csv(PERSONA_PATH).iloc[:deployed_avatars].copy()
        self.title_to_id = {normalize_title(row.title): int(item_id) for item_id, row in self.catalog.iterrows()}
        self.catalog_tokens = {
            int(item_id): set(tokenize(f"{row.title} {row.genres} {row.city} {row.summary}"))
            for item_id, row in self.catalog.iterrows()
        }
        self.catalog_category_tokens = {
            int(item_id): set(tokenize(row.genres))
            for item_id, row in self.catalog.iterrows()
        }
        self.kg_lookup = self._build_kg_lookup()
        self.base_rank_lookup = self._rank_lookup(self.base_rankings)
        self.hybrid_rank_lookup = self._rank_lookup(self.hybrid_rankings)
        self.persona_profiles = [self._persona_profile(row) for _, row in self.personas.iterrows()]

    def _build_kg_lookup(self) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        if {"head", "relation", "tail"}.issubset(self.kg.columns):
            head_col, relation_col, tail_col = "head", "relation", "tail"
        elif {"subject", "relation", "object"}.issubset(self.kg.columns):
            head_col, relation_col, tail_col = "subject", "relation", "object"
        else:
            return lookup
        for _, row in self.kg.iterrows():
            head = normalize_title(row[head_col])
            relation = str(row[relation_col]).strip()
            tail = str(row[tail_col]).strip()
            if head and relation and tail:
                lookup.setdefault(head, []).append(f"{relation}: {tail}")
        return lookup

    @staticmethod
    def _rank_lookup(rankings: np.ndarray) -> List[Dict[int, int]]:
        lookups = []
        for row in rankings:
            lookups.append({int(item_id): rank for rank, item_id in enumerate(row)})
        return lookups

    def _persona_profile(self, row: pd.Series) -> Counter:
        tokens = Counter()
        fields = ["taste", "liked_items", "frequent_categories", "evidence_items", "knowledge_graph"]
        for field in fields:
            weight = 2.0 if field in {"liked_items", "frequent_categories"} else 1.0
            for part in split_profile_list(row.get(field, "")):
                for token in tokenize(part):
                    tokens[token] += weight
        return tokens

    def select_avatar(self, persona: PersonaInput) -> Tuple[int, float]:
        request_tokens = Counter(tokenize(f"{persona.cultural_group} {persona.taste_profile} {persona.dietary_context} {persona.review_style}"))
        if not request_tokens:
            return 0, 0.0
        best_avatar = 0
        best_score = -1.0
        request_norm = math.sqrt(sum(value * value for value in request_tokens.values()))
        for avatar_id, profile in enumerate(self.persona_profiles):
            dot = sum(request_tokens[token] * profile.get(token, 0.0) for token in request_tokens)
            profile_norm = math.sqrt(sum(value * value for value in profile.values()))
            score = dot / max(request_norm * profile_norm, 1e-9)
            if score > best_score:
                best_avatar = avatar_id
                best_score = score
        return best_avatar, round(best_score, 4)

    def match_product(self, product: ProductDetails) -> Tuple[int, float, str]:
        exact = self.title_to_id.get(normalize_title(product.name))
        if exact is not None:
            return exact, 1.0, "exact catalog title match"

        query_tokens = set(tokenize(product_text(product)))
        query_city = product.city.strip().lower()
        best_item = int(self.catalog.index[0])
        best_score = -1.0
        for item_id, tokens in self.catalog_tokens.items():
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / max(len(query_tokens | tokens), 1)
            category_overlap = len(set(tokenize(product.category)) & self.catalog_category_tokens[item_id]) / max(len(set(tokenize(product.category))) or 1, 1)
            row = self.catalog.loc[item_id]
            city_bonus = 0.12 if query_city and query_city == str(row.city).strip().lower() else 0.0
            score = overlap + 0.35 * category_overlap + city_bonus
            if score > best_score:
                best_item = item_id
                best_score = score
        return best_item, round(clamp(best_score, 0.0, 1.0), 4), "nearest catalog metadata match"

    def score_product(self, avatar_id: int, persona: PersonaInput, product: ProductDetails) -> Tuple[float, List[str], List[str]]:
        item_id, match_confidence, match_type = self.match_product(product)
        item = self.catalog.loc[item_id]
        base_rank = self.base_rank_lookup[avatar_id].get(item_id, len(self.catalog))
        hybrid_rank = self.hybrid_rank_lookup[avatar_id].get(item_id, len(self.catalog))
        n_items = max(len(self.catalog), 1)

        lightgcn_score = 1.0 - base_rank / max(n_items - 1, 1)
        hybrid_score = 1.0 - hybrid_rank / max(n_items - 1, 1)
        top_window_bonus = 0.22 if hybrid_rank < 20 else 0.10 if hybrid_rank < 100 else 0.0

        product_only_text = product_text(product).lower()
        catalog_text = " ".join([str(item.title), str(item.genres), str(item.summary)]).lower()
        text = " ".join([product_only_text, catalog_text])
        diaspora_hits = [term for term in DIASPORA_TERMS if term in product_only_text]
        diaspora_score = min(1.0, sum(DIASPORA_TERMS[term] for term in diaspora_hits))
        price_score = self._price_score(persona, product, item)
        social_score = self._social_score(product, item)
        risk = self._risk_score(persona, product_only_text)
        direct_fit = self._direct_product_fit(persona, product)
        rank_score = clamp(0.65 * hybrid_score + 0.35 * lightgcn_score + top_window_bonus, 0.0, 1.0)

        if product.rating is None and product.review_count is None:
            utility = (
                0.64 * direct_fit
                + 0.16 * rank_score
                + 0.10 * price_score
                + 0.08 * diaspora_score
                + 0.02 * match_confidence
                - 0.22 * risk
            )
        else:
            utility = (
                0.36 * direct_fit
                + 0.36 * rank_score
                + 0.12 * diaspora_score
                + 0.08 * price_score
                + 0.05 * social_score
                + 0.03 * match_confidence
                - 0.16 * risk
            )
        rating = round(clamp(1.0 + 4.0 * utility, 1.0, 5.0) * 2) / 2

        cues = [
            f"LightGCN avatar {avatar_id}",
            f"base rank #{base_rank + 1}",
            f"hybrid rank #{hybrid_rank + 1}",
            f"{match_type} ({match_confidence:.2f})",
            f"catalog item: {item.title}",
        ]
        if diaspora_hits:
            cues.extend(diaspora_hits[:5])
        kg_facts = self.kg_lookup.get(normalize_title(item.title), [])[:3]
        cues.extend(kg_facts)

        reasons = [
            f"Nearest catalog item: {item.title} ({match_type}, confidence {match_confidence:.2f}).",
            f"Direct new-item persona fit score: {direct_fit:.2f}.",
            f"Base LightGCN rank for avatar {avatar_id}: #{base_rank + 1}.",
            f"Hybrid reranker rank after persona, city, price, social, and penalty signals: #{hybrid_rank + 1}.",
        ]
        if diaspora_score > 0:
            reasons.append("Diaspora-relevant food cues found: " + ", ".join(diaspora_hits[:4]) + ".")
        if product.rating is None and product.review_count is None:
            reasons.append("No Yelp rating or review-count input was supplied; social proof was treated as unavailable for this new restaurant.")
        if price_score >= 0.65:
            reasons.append("Price fit: compatible with the persona's price sensitivity.")
        elif price_score <= 0.25:
            reasons.append("Price fit: weak for the persona's budget preference.")
        if risk > 0:
            reasons.append("Risk penalty applied for context or dietary mismatch signals.")
        return rating, reasons, list(dict.fromkeys(cues))

    def _direct_product_fit(self, persona: PersonaInput, product: ProductDetails) -> float:
        product_words = set(tokenize(product_text(product)))
        persona_text = f"{persona.cultural_group} {persona.taste_profile} {persona.dietary_context} {persona.dining_context}".lower()
        product_lower = product_text(product).lower()

        score = 0.24
        persona_words = set(tokenize(persona_text))
        if persona_words and product_words:
            score += 0.18 * (len(persona_words & product_words) / max(len(persona_words), 1)) ** 0.5

        def wants(*terms: str) -> bool:
            return any(term in persona_text for term in terms)

        def has(*terms: str) -> bool:
            return any(term in product_lower for term in terms)

        if wants("pepper", "spicy", "bold", "strong flavor") and has("pepper", "spicy", "spice", "szechuan", "suya", "mexican", "indian", "thai"):
            score += 0.14
        if wants("rice", "jollof") and has("rice", "jollof", "rice bowl", "rice dishes"):
            score += 0.13
        if wants("grilled", "grill", "meat", "fish", "seafood", "chicken", "protein", "smoky") and has("grilled", "grill", "meat", "fish", "seafood", "chicken", "beef", "goat", "suya", "kebab", "barbecue", "barbeque", "bbq", "brisket", "ribs", "smoky"):
            score += 0.15
        if wants("smoky", "comfort") and has("barbecue", "barbeque", "bbq", "brisket", "ribs", "pulled chicken"):
            score += 0.16
        if wants("portion", "portions", "filling", "hearty", "generous") and has("portion", "portions", "large", "hearty", "platter", "plate", "generous"):
            score += 0.10
        if wants("value", "budget", "affordable", "cheap", "practical", "worth the money", "pricing") and has("value", "budget", "affordable", "cheap", "casual", "counter", "counter-service", "quick"):
            score += 0.10
        if wants("lively", "vibes", "social") and has("lively", "vibrant", "social", "music", "room"):
            score += 0.07
        if wants("family", "calm", "clean") and has("family", "calm", "clean", "seating", "no bar"):
            score += 0.09
        alcohol_signal = self._has_alcohol_signal(product_lower)
        if wants("halal") and has("halal") and not alcohol_signal:
            score += 0.16
        if has("nigerian", "west african", "african", "jollof", "suya", "pepper soup"):
            score += 0.10

        if wants("avoid dessert", "avoids dessert", "dessert-only", "coffee-only", "proper hot meal") and has(*LIGHT_TERMS):
            score -= 0.35
        if wants("halal") and alcohol_signal:
            score -= 0.45
        if wants("rice", "meat", "filling", "hearty") and has(*LIGHT_TERMS):
            score -= 0.20

        return clamp(score, 0.0, 1.0)

    @staticmethod
    def _review_voice(persona: PersonaInput) -> str:
        if persona.review_style == "yoruba":
            return "For this Yoruba-diaspora palate"
        if persona.review_style == "igbo":
            return "For this value-and-portion minded Igbo-diaspora palate"
        if persona.review_style == "hausa":
            return "For this Hausa-diaspora dining context"
        if persona.review_style == "urban":
            return "For this urban Nigerian-diaspora mood"
        return "For this Nigerian-diaspora persona"

    def _write_review(self, product: ProductDetails, persona: PersonaInput, rating: float, sentiment: str, reasons: List[str], cues: List[str]) -> str:
        context = persona.dining_context.strip() or "the requested dining context"
        category = product.category.strip() or "restaurant"
        product_only_text = product_text(product).lower()
        persona_text = f"{persona.taste_profile} {persona.dietary_context}".lower()

        positives = []
        concerns = []
        if any(term in product_only_text for term in ["barbecue", "barbeque", "bbq", "brisket", "ribs", "smoky"]):
            positives.append("the smoky meat and barbecue cues match the comfort-food craving")
        if any(term in product_only_text for term in ["pepper", "spicy", "spice", "szechuan", "thai", "indian", "mexican"]):
            positives.append("the flavor profile sounds bold enough to be interesting")
        if any(term in product_only_text for term in ["rice", "grilled", "grill", "meat", "fish", "seafood", "chicken"]):
            positives.append("there are proper meal cues like rice, grilled protein, or seafood")
        if any(term in product_only_text for term in ["value", "budget", "cheap", "affordable", "generous", "portion"]):
            positives.append("it seems practical for the budget and portion expectations")
        if any(term in product_only_text for term in LIGHT_TERMS):
            concerns.append("it may feel too light or dessert-focused for someone looking for a filling meal")
        if "halal" in persona_text and self._has_alcohol_signal(product_only_text):
            concerns.append("the alcohol-heavy signals make it less comfortable for a halal-aware diner")
        if not positives:
            positives.append("the basic restaurant details are clear enough to make a first-pass judgment")

        if rating >= 4.5:
            opening = f"{product.name} sounds like a very strong choice for {context}."
            closing = "I would expect this to feel satisfying, flavorful, and worth choosing."
        elif rating >= 4.0:
            opening = f"{product.name} looks like a good fit for {context}."
            closing = "It has enough of the right signals to feel like a worthwhile pick."
        elif rating >= 3.0:
            opening = f"{product.name} feels like a mixed choice for {context}."
            closing = "I might try it in the right mood, but it would not be my first recommendation for this persona."
        else:
            opening = f"{product.name} does not look like a strong match for {context}."
            closing = "I would probably skip it and look for something that better matches the craving."

        positive_sentence = "On the plus side, " + "; ".join(positives[:2]) + "."
        concern_sentence = " The main concern is that " + "; ".join(concerns[:2]) + "." if concerns else ""
        return f"{opening} {positive_sentence}{concern_sentence} Overall, I would rate this {category} {rating:.1f}/5. {closing}"

    def _price_score(self, persona: PersonaInput, product: ProductDetails, item: pd.Series) -> float:
        raw = product.price_range or item.get("price", 2)
        try:
            price = int(float(str(raw).replace("$", "").strip() or 2))
        except ValueError:
            price = 2
        table = {
            "high": {1: 1.0, 2: 0.82, 3: 0.25, 4: 0.05},
            "medium": {1: 0.75, 2: 0.9, 3: 0.50, 4: 0.25},
            "low": {1: 0.45, 2: 0.70, 3: 0.78, 4: 0.68},
        }
        return table.get(persona.price_sensitivity, table["medium"]).get(price, 0.45)

    @staticmethod
    def _social_score(product: ProductDetails, item: pd.Series) -> float:
        if product.rating is None and product.review_count is None:
            return 0.0
        rating = product.rating if product.rating is not None else 0.0
        reviews = product.review_count if product.review_count is not None else 0
        try:
            rating_score = clamp((float(rating) - 3.0) / 2.0, 0.0, 1.0)
        except (TypeError, ValueError):
            rating_score = 0.0
        try:
            review_score = clamp(math.log1p(float(reviews)) / math.log1p(3000), 0.0, 1.0)
        except (TypeError, ValueError):
            review_score = 0.0
        return 0.65 * rating_score + 0.35 * review_score

    @staticmethod
    def _risk_score(persona: PersonaInput, text: str) -> float:
        persona_text = f"{persona.taste_profile} {persona.dietary_context} {persona.dining_context}".lower()
        risk = 0.0
        if "halal" in persona_text and NaijaRecModel._has_alcohol_signal(text):
            risk += 0.55
        if any(term in persona_text for term in ["avoid dessert", "avoids dessert", "dessert-only", "coffee-only"]) and any(term in text for term in LIGHT_TERMS):
            risk += 0.55
        if any(term in text for term in LIGHT_TERMS) and any(term in persona_text for term in ["hearty", "filling", "portion", "meat", "rice"]):
            risk += 0.35
        if any(term in persona_text for term in ["pepper", "spicy", "bold"]) and not any(
            term in text for term in ["pepper", "spicy", "spice", "szechuan", "indian", "thai", "mexican", "caribbean"]
        ):
            risk += 0.25
        return clamp(risk, 0.0, 1.0)

    @staticmethod
    def _has_alcohol_signal(text: str) -> bool:
        normalized = re.sub(r"\b(no|without|zero)\s+(bar|alcohol|beer|wine|cocktails?)\b", " ", text.lower())
        tokens = set(tokenize(normalized))
        return bool(tokens & ALCOHOL_TERMS)

    def generate(self, request: ReviewRequest) -> ReviewResponse:
        avatar_id, avatar_match = self.select_avatar(request.persona)
        reviews = []
        for product in request.products:
            rating, reasons, cues = self.score_product(avatar_id, request.persona, product)
            if rating >= 4:
                sentiment: Literal["positive", "mixed", "negative"] = "positive"
            elif rating >= 3:
                sentiment = "mixed"
            else:
                sentiment = "negative"
            review = self._write_review(product, request.persona, rating, sentiment, reasons, cues)
            reviews.append(
                ProductReview(
                    name=product.name,
                    rating=rating,
                    sentiment=sentiment,
                    review=review,
                    reasons=reasons,
                    grounded_cues=cues[:10],
                )
            )

        avg = sum(review.rating for review in reviews) / max(len(reviews), 1)
        summary = (
            f"Generated {len(reviews)} restaurant rating(s) locally with the deployed NaijaRec LightGCN "
            f"checkpoint and hybrid reranker. No external LLM/API calls were used."
        )
        return ReviewResponse(
            provider="lightgcn+balanced-reranker",
            used_fallback=False,
            overall_summary=summary,
            reviews=reviews,
            warnings=[],
        )


@lru_cache(maxsize=1)
def get_model() -> NaijaRecModel:
    return NaijaRecModel()


def model_info() -> Dict[str, object]:
    model = get_model()
    return {
        "name": "NaijaRec LightGCN + balanced hybrid reranker",
        "task": "persona + restaurant details -> model-backed rating and review explanation",
        "external_api_calls": False,
        "loaded_artifacts": {
            "lightgcn_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "catalog": str(CATALOG_PATH.relative_to(PROJECT_ROOT)),
            "personas": str(PERSONA_PATH.relative_to(PROJECT_ROOT)),
            "knowledge_graph": str(KG_PATH.relative_to(PROJECT_ROOT)),
            "base_lightgcn_rankings": str(BASE_RANKINGS_PATH.relative_to(PROJECT_ROOT)),
            "hybrid_rankings": str(HYBRID_RANKINGS_PATH.relative_to(PROJECT_ROOT)),
        },
        "catalog_items": int(len(model.catalog)),
        "deployed_avatars": int(len(model.hybrid_rankings)),
        "ranking_config": {
            "rerank_pool_size": 1634 if len(model.hybrid_rankings) > 70 else 800,
            "cf_weight": 0.15,
            "semantic_weight": 0.60 if len(model.hybrid_rankings) > 70 else 0.50,
            "diaspora_weight": 0.0 if len(model.hybrid_rankings) > 70 else 0.05,
            "history_weight": 0.0,
            "city_weight": 0.2,
            "price_weight": 0.05,
            "social_weight": 0.03,
            "penalty_weight": 0.05 if len(model.hybrid_rankings) > 70 else 0.10,
            "semantic_candidate_pool_size": 0 if len(model.hybrid_rankings) > 70 else 200,
            "selection_objective": "exposure" if len(model.hybrid_rankings) > 70 else "balanced",
        },
    }


def generate_reviews(request: ReviewRequest) -> ReviewResponse:
    return get_model().generate(request)
