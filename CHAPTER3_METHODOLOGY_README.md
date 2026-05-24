# Chapter 3 Methodology: Nigerian-Diaspora Food Recommendation Simulation

## 3.1 Introduction

This chapter describes the methodological approach used to adapt an agent-based recommender simulation framework into a Nigerian-diaspora food recommendation system. The project extends the Agent4Rec simulation paradigm from a movie recommendation environment to a restaurant recommendation environment using a Yelp-style dataset. It further incorporates SimUSER-inspired components, including persona construction, episodic memory, knowledge graph grounding, perception, and a brain-like decision process.

The main objective of the methodology is to evaluate whether simulated users can provide more realistic feedback for food recommendation systems when cultural identity, dining history, price sensitivity, contextual needs, and social perception are considered. The system is designed for Nigerian users in diaspora, but it does not assume that such users only prefer Nigerian restaurants. Instead, it models Nigerian-diaspora food preference through culturally meaningful proxy signals available in a United States Yelp dataset, such as pepper-forward cuisines, grilled meat, rice dishes, seafood, generous portions, affordability, lively ambience, halal sensitivity, and comfort-food orientation.

## 3.2 Research Design

The study follows an experimental system-building methodology. A baseline collaborative filtering recommender is first used to generate ranked restaurant candidates. These candidates are then evaluated by LLM-powered avatars that represent Nigerian-diaspora diners. The avatars interact with the recommender in two main settings:

1. Controlled validation mode.
2. Natural page-by-page simulation mode.

Controlled validation mode tests whether avatars can distinguish known positive items from sampled negative items. This setting is closer to the validation approach used in Agent4Rec-style studies, where the agent is asked to classify whether it would select or reject a set of items.

Natural simulation mode tests whether the complete recommender pipeline can naturally expose relevant restaurants during a browsing session. This setting is more difficult because the held-out ground-truth restaurant must first appear in the displayed recommendation pages before the avatar can select it.

The methodology therefore separates two research questions:

```text
Question 1: Can the Nigerian-diaspora avatar recognize relevant restaurants when they are shown?

Question 2: Can the recommender naturally rank relevant restaurants high enough for the avatar to see them?
```

This distinction is important because low natural precision and recall may be caused by poor ranking exposure rather than poor avatar judgment.

## 3.3 Dataset

The project uses the `yelp-kimi` dataset. The dataset contains restaurant-style item metadata and user interaction splits used by the recommender model. The relevant dataset components include:

- restaurant titles
- restaurant categories
- historical Yelp ratings
- review counts
- price range
- item summaries
- user training interactions
- validation interactions
- test interactions
- generated persona files
- user statistics files
- optional knowledge graph triples

Although the internal file name `movie_detail.csv` remains from the original Agent4Rec codebase, the items in this experiment are restaurants. A major implementation correction in the project was to ensure that prompts, parser labels, and evaluation logs refer to restaurants rather than movies.

## 3.4 Baseline Recommender Model

The baseline recommender is LightGCN. LightGCN is used as the collaborative filtering candidate generator. It learns user and item embeddings from historical interaction data and produces a ranked list of restaurants for each avatar.

The recommender pipeline follows this structure:

```text
User historical interactions
        |
        v
LightGCN candidate generation
        |
        v
Full item ranking per avatar
        |
        v
Optional hybrid reranking
        |
        v
Avatar evaluation
```

The trained LightGCN model is restored from a checkpoint and used to generate full rankings. The observed training logs showed that the model's top-20 recall was limited. This created a ground-truth exposure problem in natural simulation, because each avatar typically sees only 20 restaurants when the configuration is 5 pages with 4 restaurants per page.

## 3.5 Ground-Truth Exposure Problem

The project observed that natural precision and recall were low even when the avatars produced reasonable restaurant judgments. The main reason was that the relevant held-out restaurant was often not present in the first displayed recommendation pages.

In this setup, each avatar is typically shown:

```text
5 pages x 4 restaurants per page = 20 restaurants
```

If the LightGCN model does not rank the validation or test restaurant within those 20 positions, then the avatar cannot possibly select it. This means low natural recall may reflect weak ranking exposure rather than weak user simulation.

For this reason, the project reports:

- total ground-truth items exposed
- exposed-user hit rate
- grounded recognition rate
- micro recall over exposed items
- natural precision and recall

This allows the evaluation to distinguish between ranking failure and recognition failure.

## 3.6 Persona Construction

Each avatar is initialized from historical user information. The persona construction step uses available historical behavior to generate a simulated diner profile. The profile includes:

- liked restaurants
- disliked restaurants
- frequent categories
- rating tendency
- price sensitivity
- pickiness
- activity level
- conformity level
- diversity preference
- cultural identity
- historical restaurant titles and categories

The cultural identity variable may include groups such as Yoruba, Igbo, Hausa, or a broader Nigerian identity. These identities are not used as rigid stereotypes. Instead, they operate as soft priors that interact with the user's actual historical record.

For example:

- A Yoruba-influenced avatar may give more weight to pepper level, lively social atmosphere, and expressive dining experience.
- An Igbo-influenced avatar may give more weight to portion size, value for money, hearty meals, and practical quality.
- A Hausa-influenced avatar may give more weight to cleanliness, family suitability, halal comfort, and alcohol-related risk.
- A general Nigerian-diaspora avatar may combine bold flavor, value, social proof, and comfort-food expectations.

The persona is therefore not only a cultural prompt. It is a combination of cultural prior, historical behavior, and situational context.

## 3.7 Episodic Memory Module

The episodic memory module stores the avatar's historical and simulated experiences. It supports continuity across recommendation pages by allowing the avatar to remember previous selections, dislikes, and ratings.

The memory stores information such as:

- restaurants selected on previous pages
- restaurants rejected on previous pages
- reasons for rejection
- ratings given by the avatar
- feelings after trying a restaurant
- context-specific preferences

This module prevents the avatar from acting as a stateless prompt. The avatar can become fatigued, satisfied, or disappointed based on previous pages.

## 3.8 Knowledge Graph Grounding

The knowledge graph module represents lightweight structured relations between the user, preferences, and restaurants. The graph is not intended to be a complete food ontology. Instead, it provides compact grounding evidence for avatar reasoning.

Example triples include:

```text
user -> likes -> spicy food
user -> dislikes -> overpriced restaurants
user -> prefers -> generous portions
user -> price_sensitivity -> high
restaurant -> has_category -> Mexican
restaurant -> has_price -> 2
restaurant -> has_rating -> 4.5
```

The graph supports reasoning such as:

```text
The user likes bold spice.
Han Dynasty is Szechuan.
Szechuan is a strong proxy for pepper-forward food.
Therefore, Han Dynasty may align with the user's taste.
```

This grounding helps the avatar produce decisions that are tied to observed item attributes rather than purely generated opinions.

## 3.9 Perception Module

The perception module estimates how a restaurant may appear to a diner before selection. The original motivation came from the observation that real users are influenced by factors beyond collaborative filtering scores. These factors include visual appeal, thumbnails, social proof, price, ambience, perceived authenticity, and context.

Because the current Yelp-style dataset does not include real thumbnails in the implemented pipeline, perception is approximated using metadata-based proxy signals:

- cuisine category
- price range
- historical Yelp rating
- review count
- nightlife or alcohol-heavy category signals
- bold flavor proxy terms
- halal or alcohol risk signals
- family suitability signals
- social proof
- premium or pretentiousness risk
- value-for-money indicators

This module is a proxy perception layer. A future version can replace or extend it with actual image thumbnails, menu text, review embeddings, and multimodal restaurant representations.

## 3.10 Brain Module

The brain module is the decision-making process used by each avatar. It combines persona, episodic memory, knowledge graph evidence, perception cues, historical rating, and current context.

The brain module follows this decision flow:

```text
1. Read the current recommendation page.
2. Extract restaurant attributes from title, category, rating, price, and summary.
3. Compare each restaurant against persona preferences.
4. Retrieve relevant memory from previous interactions.
5. Check knowledge graph evidence.
6. Apply perception cues such as price, social proof, and cuisine proxy.
7. Decide whether each restaurant aligns with the avatar.
8. Select aligned restaurants.
9. Rate selected restaurants.
10. Decide whether to continue browsing or exit.
```

The decision prompt was modified to use restaurant-specific labels:

```text
RESTAURANT: [restaurant name]; ALIGN: [yes or no]; REASON: [brief reason]
RESTAURANT: [restaurant name]; RATING: [1-5]; FEELING: [brief aftermath]
```

This correction was necessary because the original Agent4Rec codebase used movie-oriented labels such as `MOVIE` and `WATCH`.

## 3.11 Context Layer

The context layer adds situational realism. A user may choose differently depending on time, location, budget, and mood. The implemented context layer includes:

- location
- time of day
- meal occasion
- mood
- goal
- budget

Example context:

```text
location=Lagos Island;
time=night;
meal_occasion=late dinner;
mood=tired and hungry;
goal=buy food;
budget=low but willing to stretch slightly for a strong match
```

The context layer helps prevent generic recommendations. For example, a high-end wine bar may be rejected for a late-night budget meal even if it has a high Yelp rating.

## 3.12 Nigerian-Diaspora Proxy Modeling

The dataset does not contain many explicitly Nigerian or African restaurants. Therefore, the system uses diaspora proxy modeling. This means it identifies non-Nigerian restaurants that may still satisfy Nigerian-diaspora food expectations.

Proxy signals include:

- Szechuan or Chinese cuisine for heat and pepper-forward flavor
- Indian or Pakistani cuisine for spice intensity
- Mexican or Latin cuisine for bold sauces, grilled meats, and value
- Thai cuisine for heat and aromatic flavor
- Vietnamese cuisine for broths and fresh herbs
- Caribbean cuisine for rice, stew, grilled meat, and spice
- Southern or barbecue restaurants for hearty portions and smoky meat
- Middle Eastern or Mediterranean restaurants for grilled meat, rice, halal potential, and communal plates
- seafood restaurants for fish and pepper-soup-adjacent cravings

This approach is more realistic than filtering only for Nigerian restaurants, because Nigerian users in diaspora often adapt preferences across available cuisines.

## 3.13 Hybrid Reranking

The initial LightGCN ranking was improved using a hybrid reranker. The reranker combines collaborative filtering with culturally grounded and restaurant-specific features.

The reranker uses:

- collaborative filtering rank score
- semantic overlap with user history
- frequent category overlap
- Nigerian-diaspora proxy score
- city-consistency score derived from the user's training-history restaurants
- optional training-history content overlap
- price match
- social proof from rating and review count
- penalties for disliked categories
- penalties for high price when the user is price-sensitive
- penalties for alcohol-heavy mismatch where relevant
- penalties for light or pretentious categories when they conflict with the persona

The scoring function can be summarized as:

```text
final_score =
    cf_weight * collaborative_filtering_score
  + semantic_weight * profile_overlap_score
  + diaspora_weight * diaspora_proxy_score
  + history_weight * training_history_overlap_score
  + city_weight * city_consistency_score
  + price_weight * price_match_score
  + social_weight * social_proof_score
  - penalty_weight * mismatch_penalty
```

An offline rerank sweep was implemented to test reranking configurations without making LLM calls. This was necessary because full avatar simulation is expensive and slow.

The balanced non-leaking configuration from the latest offline sweep was:

```text
rerank_pool_size       = 800
rerank_cf_weight       = 0.15
rerank_semantic_weight = 0.50
rerank_diaspora_weight = 0.05
rerank_history_weight  = 0.00
rerank_city_weight     = 0.20
rerank_price_weight    = 0.05
rerank_social_weight   = 0.03
rerank_penalty_weight  = 0.10
semantic_candidates    = 200
```

This configuration is non-leaking because it does not insert validation or test items into the ranked list. It reranks only the candidate items produced by LightGCN using restaurant metadata and user information available from the training profile.

## 3.14 Offline Rerank Sweep

The offline evaluator was created to quickly test whether reranking could improve ground-truth exposure. It does not call the LLM. Instead, it uses saved full rankings, item metadata, persona fields, and held-out validation or test interactions.

The offline evaluator measures:

- hit users at top-k
- exposed ground-truth items
- recall at top-k
- precision at top-k
- NDCG at top-k
- mean reciprocal rank

This tool allows different reranker weights to be compared cheaply before running expensive LLM-based simulations.

The offline rerank sweep showed that reranking could substantially improve top-20 exposure compared with the base ranking. On the 70-avatar diagnostic set, the best configuration produced:

```text
Base LightGCN @20:
Exposed users:       8 / 70
Recall@20:           0.114
NDCG@20:             0.052
Median GT rank:      156.0

Previous hybrid @20:
Exposed users:       18 / 70
Recall@20:           0.257
NDCG@20:             0.131
Median GT rank:      69.5

Metric-focused semantic-city hybrid @20:
Exposed users:       31 / 70
Recall@20:           0.443
NDCG@20:             0.199
Median GT rank:      30.5

Balanced hybrid @20:
Exposed users:       31 / 70
Recall@20:           0.443
NDCG@20:             0.193
Median GT rank:      30.5
```

The result indicates that many relevant restaurants are present in the broader candidate pool but are not ranked high enough by the base LightGCN model. The balanced hybrid reranker moves more of those candidates into the visible top-20 recommendation window while retaining a small diaspora/persona utility signal.

## 3.15 Controlled Validation Mode

Controlled validation mode evaluates the avatar's ability to classify known positive and negative restaurants.

For each avatar:

1. The system samples known positive restaurants from the validation and test sets.
2. It samples unobserved restaurants as negatives.
3. The avatar receives a mixed list of restaurants.
4. The avatar outputs `WATCH: yes` or `WATCH: no` for each restaurant.
5. The system compares the avatar's selections against the known positives.

Although the word `WATCH` remains in some internal variable names for compatibility with Agent4Rec, the visible prompt has been corrected to restaurant-native language:

```text
RESTAURANT: [restaurant name]; WATCH: [yes or no]; REASON: [brief reason]
```

The project later corrected parser behavior so that zero selected restaurants are treated as precision `0` rather than `nan`.

The latest 50-avatar controlled validation results were:

```text
Precision: 0.667
Recall:    0.630
Accuracy:  0.645
F1:        0.622
```

These results are lower than some reported results in SimUSER and Agent4Rec, but the task is not identical. This project evaluates culturally grounded restaurant choice in a non-Nigerian Yelp dataset, which is more difficult than a controlled movie-item classification task.

## 3.16 Natural Simulation Mode

Natural simulation mode evaluates the full recommender-agent interaction loop.

For each avatar:

1. The recommender generates ranked restaurants.
2. The avatar sees one page of restaurants at a time.
3. The avatar evaluates each restaurant.
4. The avatar selects restaurants that align with its persona and context.
5. The avatar rates selected restaurants.
6. The avatar decides whether to continue or exit.
7. The system records behavior and interview satisfaction.

Natural simulation is more realistic than controlled validation because it includes ranking exposure, browsing fatigue, page-level interaction, memory updates, and exit behavior.

However, natural simulation produced low raw precision and recall because many ground-truth restaurants were not exposed in the first 20 displayed items. This is a ranking limitation rather than only an avatar limitation.

## 3.17 Ground-Truth Injection Policy

The project considered a flag called `--inject_ground_truth`, which forces a held-out positive item into a displayed recommendation page when no ground-truth item appears naturally.

This approach was rejected for final natural evaluation because it gives the system access to the answer. It changes the recommendation exposure process and therefore biases natural precision and recall.

The project treats ground-truth injection only as a diagnostic probe. It can answer:

```text
If the correct restaurant is shown, can the avatar recognize it?
```

It cannot answer:

```text
Can the recommender naturally surface the correct restaurant?
```

Therefore, natural simulation results should be reported without ground-truth injection.

## 3.18 Output Parsing and Measurement Integrity

LLM outputs can drift from the required format. For example, an LLM may produce:

```text
RESTAURANT: Bitar's; RATING: 4; REVIEW: ...
```

instead of:

```text
RESTAURANT: Bitar's; RATING: 4; FEELING: ...
```

The project avoids silently imputing invalid ratings as `3.0` because doing so could bias the research. Instead, the parser validates structured outputs and records parse failures or partial parses.

The parser protocol is:

```text
1. Strip hidden thinking blocks.
2. Accept restaurant-native labels.
3. Match restaurant titles against the displayed page.
4. Reject off-page or hallucinated titles.
5. Reject invalid ratings outside the 1-5 range.
6. Record parse failures.
7. Exclude invalid page ratings from rating averages.
8. Report parse-valid rate.
```

This is methodologically safer than silently converting invalid outputs into neutral ratings.

## 3.19 Evaluation Metrics

The project reports two groups of metrics.

### 3.19.1 Controlled Validation Metrics

Controlled validation uses:

- precision
- recall
- accuracy
- F1 score

These metrics measure whether the avatar selects known positive restaurants and rejects sampled negatives.

### 3.19.2 Natural Simulation Metrics

Natural simulation uses:

- number of likes
- average rating
- average valid page rating
- click rate
- exposure-adjusted click rate
- interview satisfaction
- true satisfaction score
- exit page
- ground-truth items exposed
- precision
- recall over exposed users
- micro precision
- micro recall over exposed items
- grounded recognition rate
- exposed-user hit rate
- parse-valid rate

The most important methodological distinction is between raw natural precision and exposed-item recognition. Raw precision can be low when relevant items are never shown. Exposed-item recognition measures whether the avatar selected a ground-truth restaurant after it appeared.

## 3.20 Scalability and Runtime

LLM-based simulation is slow and expensive. A 50-avatar controlled validation run took approximately 20 minutes in the observed setup. Scaling to 800 avatars may require several hours, depending on rate limits, LLM provider, execution mode, and retry behavior.

The project therefore uses a two-level evaluation strategy:

```text
1. Offline rerank sweep for fast, no-LLM ranking diagnostics.
2. LLM-based validation and simulation for behavioral evaluation.
```

This is necessary because running all experiments only through LLM avatars would be impractical under hackathon time and API constraints.

## 3.21 Experimental Procedure

The methodology can be reproduced through the following steps.

### Step 1: Prepare Dataset

The Yelp-style restaurant dataset is stored under:

```text
datasets/yelp-kimi/
```

The dataset includes collaborative filtering splits, restaurant metadata, persona files, user statistics, and optional knowledge graph triples.

### Step 2: Train or Load LightGCN

The trained LightGCN checkpoint is loaded from:

```text
recommenders/weights/yelp-kimi/LightGCN/
```

The model generates full rankings for each avatar.

### Step 3: Initialize Avatars

Each avatar is initialized from:

```text
datasets/yelp-kimi/simulation/all_personas_like_modify.csv
datasets/yelp-kimi/simulation/user_statistic.csv
datasets/yelp-kimi/raw_data/agg_top_25.csv
```

The initialization builds persona, memory, cultural context, and historical preference evidence.

### Step 4: Apply Optional Hybrid Reranking

The reranker can be enabled to improve ranking exposure using persona, price, social proof, and Nigerian-diaspora proxy features.

The recommended metric-focused configuration after offline tuning is:

```bash
python main.py \
  --dataset yelp-kimi \
  --modeltype LightGCN \
  --model_path '0515_yelp_lgn_t0p0_Ks=20_patience=10_n_layers=2_batch_size=1024_neg_sample=1_lr=0.0005' \
  --n_avatars 70 \
  --max_pages 5 \
  --min_browse_pages 5 \
  --items_per_page 4 \
  --execution_mode serial \
  --ground_truth_split valid \
  --sim_llm_provider gemini \
  --sim_embedding_provider local \
  --sim_llm_max_tokens 450 \
  --sim_llm_min_interval 25 \
  --hybrid_rerank \
  --semantic_candidate_expansion \
  --semantic_candidate_pool_size 200 \
  --semantic_candidate_cf_floor 0.05 \
  --rerank_pool_size 800 \
  --rerank_cf_weight 0.15 \
  --rerank_semantic_weight 0.50 \
  --rerank_diaspora_weight 0.05 \
  --rerank_history_weight 0.0 \
  --rerank_city_weight 0.2 \
  --rerank_price_weight 0.05 \
  --rerank_social_weight 0.03 \
  --rerank_penalty_weight 0.10 \
  --natural_relevance_scope heldout_plus_persona
```

For a more realistic browsing run, `--min_browse_pages` may be reduced to `3`, but this will usually lower measured natural exposure because avatars can exit before seeing all top-20 recommendations.

### Step 5: Run Controlled Validation

Controlled validation samples positive and negative restaurants, asks the avatar to select or reject each item, and computes precision, recall, accuracy, and F1.

### Step 6: Run Natural Simulation

Natural simulation exposes each avatar to multiple recommendation pages. The avatar selects restaurants, rates them, updates memory, and decides when to exit.

### Step 7: Save Behavioral Logs

The system saves:

- page-level behavior
- interview responses
- running logs
- train updates
- metrics
- parse diagnostics
- rerank diagnostics

### Step 8: Analyze Results

Results are interpreted by separating:

- recommender exposure quality
- avatar recognition quality
- satisfaction behavior
- parse validity
- runtime cost

This separation prevents incorrect conclusions, such as blaming the avatar for low recall when the ground-truth restaurant was never shown.

## 3.22 Methodological Limitations

The study has several limitations.

First, the dataset is not Nigerian or African. Nigerian-diaspora relevance is inferred from proxy features rather than direct Nigerian restaurant labels.

Second, the system does not currently use real restaurant thumbnails. Perception is approximated using metadata such as category, price, review count, and rating.

Third, the baseline LightGCN ranking is weak for top-20 ground-truth exposure. This limits natural recall.

Fourth, LLM outputs can drift from required formats. The parser has been improved, but format drift remains a general risk.

Fifth, large-scale validation with 800 or more avatars may fail or become expensive due to rate limits, timeouts, and API throughput constraints.

Sixth, the cultural personas are generated and simulated rather than directly collected from Nigerian users in diaspora. Human validation would strengthen the claims.

## 3.23 Ethical Considerations

The system models cultural identity as a soft preference prior rather than a deterministic stereotype. Cultural group labels are used to represent possible food preferences, not fixed behavior.

The methodology avoids claiming that all Yoruba, Igbo, Hausa, or Nigerian users behave the same way. Historical user behavior and item evidence are allowed to override cultural priors.

The study also avoids inflating results through ground-truth injection in natural evaluation. This preserves the integrity of ranking and recall claims.

## 3.24 Summary

This methodology adapts Agent4Rec into a Nigerian-diaspora restaurant recommendation simulator. It combines LightGCN ranking, SimUSER-inspired avatars, persona grounding, episodic memory, knowledge graph evidence, perception cues, contextual decision making, and structured evaluation.

The project found that controlled avatar validation can achieve usable performance, with precision, recall, accuracy, and F1 all above 0.6 in the latest 50-avatar validation run. However, natural simulation remains constrained by ground-truth exposure, meaning that relevant restaurants often do not appear in the first displayed recommendation pages.

The key methodological contribution is therefore not only a higher metric score. It is the construction of a more realistic evaluation framework that separates:

```text
recommender ranking quality
from
avatar recognition quality
from
behavioral satisfaction quality
```

This separation is essential for evaluating culturally grounded food recommender systems.
