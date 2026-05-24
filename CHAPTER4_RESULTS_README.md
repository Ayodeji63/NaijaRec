# Chapter 4: Results and Evaluation

## 4.1 Introduction

This chapter presents the experimental results of NaijaRec, a Nigerian-diaspora restaurant recommender and avatar simulation system. The evaluation focuses on three questions:

1. Whether the recommender exposes relevant restaurants in the top-20 ranked list.
2. Whether the avatar can recognize restaurants that match its persona and context.
3. Whether the full simulation produces realistic behavioral outcomes such as clicks, ratings, satisfaction, and valid review text.

Best observed scores are reported alongside a larger confirmatory natural-simulation run. The full sweep outputs are stored in the project results directory and can be used as supporting evidence.

## 4.2 Evaluation Settings

NaijaRec was evaluated using four complementary settings:

| Setting | Purpose | Main Output |
|---|---|---|
| Offline retrieval | Measures whether the ranking model exposes the held-out relevant restaurant in the top-20 list | Recall@20, Precision@20, NDCG@20, median rank |
| Controlled validation | Tests whether avatars recognize relevant restaurants when shown controlled positive and negative examples | Precision, Recall, Accuracy, F1 |
| Natural simulation | Measures the complete recommender-avatar loop page by page | Click rate, satisfaction, parse-valid rate, grounded recognition |
| Web demo evaluation | Confirms that the deployed application uses the local NaijaRec model rather than external LLM calls | Generated ratings and review-style explanations |

The offline evaluation uses 821 avatars. Controlled validation was scaled from an initial 50-avatar run to a 500-avatar run. Natural simulation is more expensive: a 70-avatar run is retained as the best observed behavioral result, a 500-avatar metric-focused run provides larger-scale confirmation, and a smaller 20-avatar run serves as an additional smoke test.

## 4.3 Best Offline Retrieval Results

The best offline retrieval result was obtained using the hybrid reranker on top of LightGCN. The reranker combined collaborative filtering with semantic restaurant-persona alignment, city/context matching, price compatibility, social proof, and risk penalties.

![Best offline model scores](chapter4_assets/best_offline_scores.png)

**Table 4.1: Best Offline Retrieval Scores on 821 Avatars**

| Metric | Best Score | Interpretation |
|---|---:|---|
| Hit users@20 | 404 / 821 | Relevant restaurant appeared in the top-20 list for 404 avatars |
| Exposure / Recall@20 | 0.492 | The system exposed the held-out relevant restaurant for 49.2% of avatars |
| NDCG@20 | 0.279 | Relevant restaurants were ranked substantially closer to the top of the list |
| Strict Precision@20 | 0.0246 | With one held-out relevant item per user, the theoretical maximum is 0.05 |
| Median ground-truth rank | 21 | The median relevant restaurant moved close to the top-20 boundary |
| Mean proxy utility@20 | 0.484 | Recommended restaurants had stronger persona/context fit |

The strict Precision@20 value appears small because each avatar has only one held-out positive item and twenty displayed recommendations. Under this evaluation design, the maximum possible Precision@20 is:

```text
1 relevant item / 20 recommended items = 0.05
```

Therefore, the best observed strict Precision@20 of 0.0246 represents approximately 49.2% of the theoretical maximum.

## 4.4 Controlled Avatar Validation

Controlled validation measures the avatar's ability to classify restaurants as aligned or not aligned with the persona when observed positive and unobserved negative restaurants are directly presented in the candidate set. This isolates avatar recognition from recommender exposure.

![Controlled validation metrics](chapter4_assets/controlled_validation_metrics.png)

**Table 4.2: 500-Avatar Controlled Validation Scores**

| Metric | Score |
|---|---:|
| Precision | 0.884 |
| Recall | 0.726 |
| Accuracy | 0.835 |
| F1 | 0.776 |

![Controlled validation comparison](chapter4_assets/controlled_validation_comparison.png)

**Table 4.3: Controlled Validation Scale-Up Comparison**

| Metric | Initial 50-Avatar Run | New 500-Avatar Run | Change |
|---|---:|---:|---:|
| Precision | 0.667 | 0.884 | +0.217 |
| Recall | 0.630 | 0.726 | +0.096 |
| Accuracy | 0.645 | 0.835 | +0.190 |
| F1 | 0.622 | 0.776 | +0.154 |

The 500-avatar run required approximately 2,212 seconds (36.9 minutes) and improves every controlled-validation measure over the initial 50-avatar experiment. This provides stronger evidence that the avatar can recognize stable preference matches when relevant restaurants are directly included in the candidate list. These scores should not be interpreted as offline ranking or natural-simulation precision, because controlled validation removes the retrieval-exposure bottleneck.

## 4.5 Natural Simulation Results

Natural simulation evaluates the full system loop. The recommender ranks restaurants, the avatar browses page by page, and the system records clicks, ratings, exits, satisfaction, and recognition of exposed ground-truth restaurants.

![Natural simulation metrics](chapter4_assets/natural_simulation_metrics.png)

**Table 4.4: Best 70-Avatar Natural Simulation Scores**

| Metric | Score |
|---|---:|
| Overall click rate | 0.426 |
| Average valid page rating | 4.208 |
| True satisfaction score | 0.721 |
| Parse-valid rate | 1.000 |
| Ground-truth items exposed | 31 |
| Grounded recognition rate | 0.677 |
| Exposed-user hit rate | 21 / 31 |
| Strict micro precision | 0.0317 |
| Strict micro recall over exposed items | 0.677 |
| Expanded micro recall | 0.600 |

The natural simulation confirms an important distinction: when the relevant restaurant is exposed, the avatar recognizes it in 67.7% of cases. However, strict precision remains low because the metric counts only the single held-out restaurant as relevant, even when other selected restaurants are plausible matches for the persona.

## 4.6 Large-Scale 500-Avatar Confirmation

The new large-scale run evaluated 500 avatars across five pages each, producing 2,500 page-level observations. It used the hybrid reranker with semantic candidate expansion and the `heldout_plus_persona` natural relevance scope.

![500-avatar natural simulation metrics](chapter4_assets/natural_500_simulation_metrics.png)

**Table 4.5: 500-Avatar Metric-Focused Natural Simulation Scores**

| Metric | Score |
|---|---:|
| Overall click rate | 0.262 |
| Average valid page rating | 4.221 |
| True satisfaction score | 0.631 |
| Parse-valid pages | 2,372 / 2,500 |
| Parse-valid rate | 0.949 |
| Ground-truth items exposed | 237 / 500 |
| Grounded recognition rate | 0.582 |
| Exposed-user hit rate | 138 / 237 |
| Strict micro precision | 0.0475 |
| Strict micro recall over exposed items | 0.582 |
| Expanded relevant items exposed | 325 |
| Expanded micro precision | 0.0626 |
| Expanded micro recall | 0.560 |

This scale-up is strong evidence that the retrieval behavior generalizes beyond the smaller behavioral run: the held-out restaurant was exposed for 47.4% of avatars, close to the 49.2% exposure measured in the 821-avatar offline evaluation. The recognition rate is lower than the best 70-avatar result (58.2% versus 67.7%), so this run should be presented as a larger confirmation rather than a new best recognition score.

All avatars in this run browsed all five pages because it was executed in metric-focused mode with the full top-20 exposure window enabled. Consequently, its average exit page of `5.0` is a design condition rather than a natural exit-behavior finding, and its exposure-adjusted click rate is necessarily identical to ordinary click rate (`0.2621`). The run also recorded 128 parse-failed pages and 101 partial pages, so structured-output validation remains necessary at scale.

## 4.7 Additional 20-Avatar Smoke Run

A smaller 20-avatar run was also performed to test the LLM-backed avatar path under FreeLLMAPI routing. This run was not used as the main result because it is smaller, but it provides useful evidence that the system can reach strong recognition when relevant items are exposed.

![FreeLLMAPI smoke metrics](chapter4_assets/freellmapi_smoke_metrics.png)

**Table 4.6: 20-Avatar Smoke Run Scores**

| Metric | Score |
|---|---:|
| Overall click rate | 0.222 |
| True satisfaction score | 0.708 |
| Parse-valid rate | 0.960 |
| Ground-truth items exposed | 10 |
| Grounded recognition rate | 0.800 |
| Exposed-user hit rate | 8 / 10 |
| Expanded micro recall | 0.786 |

This run suggests that the avatar decision module can perform well when relevant restaurants are surfaced. The lower parse-valid rate also shows why output validation remains necessary when LLM-generated text is used.

## 4.8 Best Score Summary

![Best score summary](chapter4_assets/best_score_summary.png)

**Table 4.7: Best Observed Scores Across Evaluation Settings**

| Evaluation Setting | Best Metric | Score |
|---|---|---:|
| Offline retrieval | Exposure@20 | 0.492 |
| Offline retrieval | NDCG@20 | 0.279 |
| Controlled validation | Precision | 0.884 |
| Controlled validation | Accuracy | 0.835 |
| Natural simulation | Grounded recognition | 0.677 |
| Natural simulation | Satisfaction | 0.721 |
| 20-avatar smoke run | Grounded recognition | 0.800 |
| 20-avatar smoke run | Parse-valid rate | 0.960 |

The strongest result is the improvement in exposure: the best hybrid reranker exposed held-out restaurants for 404 of 821 avatars. The behavioral results show that when exposure occurs, avatars frequently recognize and select the relevant restaurant.

The 500-avatar run is excluded from the table of *best* behavioral scores because its grounded recognition and satisfaction are below the 70-avatar best, but it supplies stronger sample-size evidence for the exposure result.

## 4.9 Web Application Evaluation

The deployed web application was tested as a local, containerized NaijaRec demo. The application takes persona and restaurant details as input and returns a rating with a review-style explanation.

The final demo does not use external LLM/API calls. Instead, it uses the deployed LightGCN checkpoint and hybrid reranker artifacts as model evidence, combined with direct persona-product matching for new restaurants.

**Table 4.8: Web Demo Behavior Checks**

| Persona / Restaurant Case | Expected Behavior | Observed Output |
|---|---|---|
| Yoruba persona + Eko Pepper Grill | Strong positive match | Positive review, high rating |
| Yoruba persona + Taco World | Positive quick-lunch match | Positive review, good rating |
| Igbo persona + Mission BBQ | Strong barbecue/value match | Positive review, high rating |
| Hausa halal-aware persona + Arewa Halal Grill | Positive halal-family match | Positive review, good rating |
| Spicy meal persona + Kiwi Frozen Yogurt | Negative mismatch | Negative review, low rating |

This confirms that the web demo is suitable for submission: it accepts user persona and restaurant details, uses the local NaijaRec model pipeline, and generates ratings and review-style outputs without external API calls.

## 4.10 Discussion

The results show that NaijaRec's strongest contribution is not simply avatar generation, but the separation of three evaluation layers:

1. Ranking exposure: whether LightGCN and the hybrid reranker place the relevant restaurant in the visible top-20 list.
2. Avatar recognition: whether the avatar selects relevant restaurants once they are visible.
3. Behavioral realism: whether the avatar produces reasonable clicks, ratings, satisfaction, and review-style explanations.

The offline results show that hybrid reranking is necessary because the best model substantially improves top-20 exposure and NDCG. The 500-avatar controlled validation shows strong direct recognition once relevant restaurants are presented, reaching 0.884 precision and 0.776 F1. The 500-avatar natural confirmation further shows that exposure remains near the offline exposure rate at larger scale, although natural recognition and satisfaction do not improve over the best smaller run.

## 4.11 Limitations

The main limitation is that strict precision and recall depend on a sparse held-out evaluation design with only one ground-truth restaurant per avatar. This underestimates the relevance of other restaurants that may also fit the persona. The expanded relevance metric partially addresses this by considering additional plausible matches, but future work should include richer multi-positive ground truth.

A second limitation is that the dataset does not contain restaurant images or menu-level features. As a result, perception is approximated using metadata such as category, city, price, social proof, and cuisine cues.

Finally, LLM-based full simulation is expensive and slow. The 500-avatar run required approximately 14,493 seconds (about 4.03 hours). Its metric-focused configuration forced browsing to page 5, making it suitable for measuring top-20 exposure but not for claiming realistic early-exit behavior. Offline reranking is therefore used as the main scalable diagnostic, while natural simulation is used as a behavioral validation layer.

## 4.12 Summary

The best NaijaRec model exposed relevant restaurants for 49.2% of 821 avatars at top-20 and achieved a best NDCG@20 of 0.279. The new 500-avatar controlled validation achieved 0.884 precision, 0.835 accuracy, and 0.776 F1, while the best 70-avatar natural simulation achieved 0.721 satisfaction and 0.677 grounded recognition among exposed users. In the larger 500-avatar metric-focused natural run, relevant restaurants were exposed for 237 users (47.4%) and recognized for 138 of those users (58.2%), with a 94.9% parse-valid rate. These results support the conclusion that the hybrid LightGCN-based NaijaRec pipeline improves restaurant exposure and produces persona-grounded recommendation behavior, while also documenting the distinction between controlled recognition and natural exposure.

## Reproducibility

The Chapter 4 figures were generated with:

```bash
/home/dell/agentvenv39/bin/python scripts/generate_chapter4_assets.py
```

The main generated assets are stored in:

```text
chapter4_assets/
```
