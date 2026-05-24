from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("chapter4_assets")
OUT_DIR.mkdir(exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=180)
    plt.close()


def add_bar_labels(ax, fmt="{:.3f}", padding=0.01):
    for patch in ax.patches:
        height = patch.get_height()
        if height == 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + padding,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


offline = pd.DataFrame(
    [
        {
            "Model": "Base LightGCN",
            "Hit Users@20": 129,
            "Recall@20": 0.15712545676004872,
            "Precision@20": 0.007856272838002436,
            "NDCG@20": 0.060777361149330224,
            "Median GT Rank": 139,
            "Proxy Utility": 0.3463491743178204,
        },
        {
            "Model": "Hybrid Exposure",
            "Hit Users@20": 404,
            "Recall@20": 0.4920828258221681,
            "Precision@20": 0.024604141291108404,
            "NDCG@20": 0.2775527034822097,
            "Median GT Rank": 21,
            "Proxy Utility": 0.48026330894422486,
        },
        {
            "Model": "Balanced Hybrid",
            "Hit Users@20": 398,
            "Recall@20": 0.48477466504263095,
            "Precision@20": 0.024238733252131547,
            "NDCG@20": 0.27589825208856833,
            "Median GT Rank": 22,
            "Proxy Utility": 0.481283797904643,
        },
    ]
)
offline.to_csv(OUT_DIR / "offline_821_metrics.csv", index=False)


validation = pd.DataFrame(
    [
        {"Metric": "Precision", "Score": 0.8796183516037352},
        {"Metric": "Recall", "Score": 0.7119366626065774},
        {"Metric": "Accuracy", "Score": 0.8252131546894031},
        {"Metric": "F1", "Score": 0.7637840032480714},
    ]
)
validation.to_csv(OUT_DIR / "controlled_validation_metrics.csv", index=False)

validation_comparison = pd.DataFrame(
    [
        {"Run": "50-avatar validation", "Metric": "Precision", "Score": 0.6666666666666665},
        {"Run": "50-avatar validation", "Metric": "Recall", "Score": 0.630},
        {"Run": "50-avatar validation", "Metric": "Accuracy", "Score": 0.645},
        {"Run": "50-avatar validation", "Metric": "F1", "Score": 0.622},
        {"Run": "500-avatar validation", "Metric": "Precision", "Score": 0.8836666666666666},
        {"Run": "500-avatar validation", "Metric": "Recall", "Score": 0.726},
        {"Run": "500-avatar validation", "Metric": "Accuracy", "Score": 0.8345},
        {"Run": "500-avatar validation", "Metric": "F1", "Score": 0.7755333333333333},
        {"Run": "821-avatar validation", "Metric": "Precision", "Score": 0.8796183516037352},
        {"Run": "821-avatar validation", "Metric": "Recall", "Score": 0.7119366626065774},
        {"Run": "821-avatar validation", "Metric": "Accuracy", "Score": 0.8252131546894031},
        {"Run": "821-avatar validation", "Metric": "F1", "Score": 0.7637840032480714},
    ]
)
validation_comparison.to_csv(OUT_DIR / "controlled_validation_comparison.csv", index=False)


natural = pd.DataFrame(
    [
        {"Metric": "Click Rate", "Score": 0.4264285714285715},
        {"Metric": "Satisfaction", "Score": 0.7214285714285713},
        {"Metric": "Parse Valid", "Score": 1.0},
        {"Metric": "Grounded Recognition", "Score": 0.6774193548387096},
        {"Metric": "Expanded Micro Recall", "Score": 0.6},
    ]
)
natural.to_csv(OUT_DIR / "natural_simulation_metrics.csv", index=False)

natural_500 = pd.DataFrame(
    [
        {"Metric": "Click Rate", "Score": 0.2621},
        {"Metric": "Satisfaction", "Score": 0.6306613226452905},
        {"Metric": "Parse Valid", "Score": 0.9488},
        {"Metric": "Grounded Recognition", "Score": 0.5822784810126582},
        {"Metric": "Expanded Micro Recall", "Score": 0.56},
    ]
)
natural_500.to_csv(OUT_DIR / "natural_500_simulation_metrics.csv", index=False)

natural_821 = pd.DataFrame(
    [
        {"Metric": "Click Rate", "Score": 0.34293544457978076},
        {"Metric": "Satisfaction", "Score": 0.6467635402906208},
        {"Metric": "Parse Valid", "Score": 0.9995127892813642},
        {"Metric": "Grounded Recognition", "Score": 0.6457286432160804},
        {"Metric": "Expanded Micro Recall", "Score": 0.6270270270270271},
    ]
)
natural_821.to_csv(OUT_DIR / "natural_821_simulation_metrics.csv", index=False)

natural_scale = pd.DataFrame(
    [
        {"Run": "70-avatar natural", "Metric": "Click Rate", "Score": 0.4264285714285715},
        {"Run": "70-avatar natural", "Metric": "Satisfaction", "Score": 0.7214285714285713},
        {"Run": "70-avatar natural", "Metric": "Parse Valid", "Score": 1.0},
        {"Run": "70-avatar natural", "Metric": "Grounded Recognition", "Score": 0.6774193548387096},
        {"Run": "70-avatar natural", "Metric": "Expanded Micro Recall", "Score": 0.6},
        {"Run": "500-avatar natural", "Metric": "Click Rate", "Score": 0.2621},
        {"Run": "500-avatar natural", "Metric": "Satisfaction", "Score": 0.6306613226452905},
        {"Run": "500-avatar natural", "Metric": "Parse Valid", "Score": 0.9488},
        {"Run": "500-avatar natural", "Metric": "Grounded Recognition", "Score": 0.5822784810126582},
        {"Run": "500-avatar natural", "Metric": "Expanded Micro Recall", "Score": 0.56},
        {"Run": "821-avatar natural", "Metric": "Click Rate", "Score": 0.34293544457978076},
        {"Run": "821-avatar natural", "Metric": "Satisfaction", "Score": 0.6467635402906208},
        {"Run": "821-avatar natural", "Metric": "Parse Valid", "Score": 0.9995127892813642},
        {"Run": "821-avatar natural", "Metric": "Grounded Recognition", "Score": 0.6457286432160804},
        {"Run": "821-avatar natural", "Metric": "Expanded Micro Recall", "Score": 0.6270270270270271},
    ]
)
natural_scale.to_csv(OUT_DIR / "natural_simulation_scale_comparison.csv", index=False)


smoke = pd.DataFrame(
    [
        {"Metric": "Click Rate", "Score": 0.2225},
        {"Metric": "Satisfaction", "Score": 0.7083333333333334},
        {"Metric": "Parse Valid", "Score": 0.96},
        {"Metric": "Grounded Recognition", "Score": 0.8},
        {"Metric": "Expanded Micro Recall", "Score": 0.7857142857142857},
    ]
)
smoke.to_csv(OUT_DIR / "freellmapi_smoke_metrics.csv", index=False)

best_offline = pd.DataFrame(
    [
        {"Metric": "Exposure / Recall@20", "Score": 0.4920828258221681},
        {"Metric": "NDCG@20", "Score": 0.279449169241899},
        {"Metric": "Strict Precision@20", "Score": 0.024604141291108404},
        {"Metric": "Proxy Utility", "Score": 0.48411319015982585},
    ]
)
best_offline.to_csv(OUT_DIR / "best_offline_scores.csv", index=False)

best_summary = pd.DataFrame(
    [
        {"Evaluation": "Offline retrieval", "Best Metric": "Exposure@20", "Score": 0.4920828258221681},
        {"Evaluation": "Offline retrieval", "Best Metric": "NDCG@20", "Score": 0.279449169241899},
        {"Evaluation": "Controlled validation", "Best Metric": "Precision", "Score": 0.8836666666666666},
        {"Evaluation": "Controlled validation", "Best Metric": "Accuracy", "Score": 0.8345},
        {"Evaluation": "Natural simulation", "Best Metric": "Grounded recognition", "Score": 0.6774193548387096},
        {"Evaluation": "Natural simulation", "Best Metric": "Satisfaction", "Score": 0.7214285714285713},
        {"Evaluation": "20-avatar smoke run", "Best Metric": "Grounded recognition", "Score": 0.8},
        {"Evaluation": "20-avatar smoke run", "Best Metric": "Parse-valid rate", "Score": 0.96},
    ]
)
best_summary.to_csv(OUT_DIR / "best_score_summary.csv", index=False)


plt.figure(figsize=(8.2, 4.6))
ax = offline.set_index("Model")[["Recall@20", "NDCG@20", "Precision@20"]].plot(
    kind="bar",
    ax=plt.gca(),
    color=["#1f7a5b", "#d9902f", "#4467a6"],
    width=0.72,
)
ax.set_title("Offline Retrieval Performance on 821 Avatars")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 0.55)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper left", frameon=False)
plt.xticks(rotation=0)
savefig("offline_retrieval_comparison.png")


fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
axes[0].bar(offline["Model"], offline["Hit Users@20"], color="#1f7a5b")
axes[0].set_title("Relevant Restaurants Exposed@20")
axes[0].set_ylabel("Users")
axes[0].grid(axis="y", alpha=0.25)
axes[0].tick_params(axis="x", rotation=18)
for patch in axes[0].patches:
    axes[0].text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + 5,
        f"{int(patch.get_height())}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axes[1].bar(offline["Model"], offline["Median GT Rank"], color="#b84a3a")
axes[1].set_title("Median Ground-Truth Rank")
axes[1].set_ylabel("Rank, lower is better")
axes[1].grid(axis="y", alpha=0.25)
axes[1].tick_params(axis="x", rotation=18)
for patch in axes[1].patches:
    axes[1].text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + 3,
        f"{int(patch.get_height())}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
savefig("offline_exposure_and_rank.png")


plt.figure(figsize=(7.4, 4.2))
ax = validation.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#4467a6",
)
ax.set_title("821-Avatar Controlled Validation")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
add_bar_labels(ax)
savefig("controlled_validation_metrics.png")

plt.figure(figsize=(8.0, 4.4))
comparison_for_plot = validation_comparison.pivot(index="Metric", columns="Run", values="Score")
ax = comparison_for_plot.loc[["Precision", "Recall", "Accuracy", "F1"]].plot(
    kind="bar",
    ax=plt.gca(),
    color=["#9ca8c0", "#4467a6", "#6f5686"],
    width=0.72,
)
ax.set_title("Controlled Validation Scale-Up")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="lower right")
plt.xticks(rotation=0)
add_bar_labels(ax)
savefig("controlled_validation_comparison.png")


plt.figure(figsize=(8.0, 4.4))
ax = natural.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#1f7a5b",
)
ax.set_title("Natural Simulation Behavioral Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("natural_simulation_metrics.png")


plt.figure(figsize=(8.0, 4.4))
ax = natural_500.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#6f5686",
)
ax.set_title("500-Avatar Metric-Focused Natural Simulation")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("natural_500_simulation_metrics.png")


plt.figure(figsize=(8.0, 4.4))
ax = natural_821.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#3a7db8",
)
ax.set_title("821-Avatar Natural Simulation Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("natural_821_simulation_metrics.png")


plt.figure(figsize=(9.2, 4.8))
natural_scale_for_plot = natural_scale.pivot(index="Metric", columns="Run", values="Score")
ax = natural_scale_for_plot.loc[
    [
        "Click Rate",
        "Satisfaction",
        "Parse Valid",
        "Grounded Recognition",
        "Expanded Micro Recall",
    ]
].plot(
    kind="bar",
    ax=plt.gca(),
    color=["#9fc4a7", "#1f7a5b", "#3a7db8"],
    width=0.76,
)
ax.set_title("Natural Simulation Metrics Across Run Scales")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="lower right")
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("natural_simulation_scale_comparison.png")


plt.figure(figsize=(8.0, 4.4))
ax = smoke.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#d9902f",
)
ax.set_title("20-Avatar FreeLLMAPI Smoke Run")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("freellmapi_smoke_metrics.png")


plt.figure(figsize=(8.0, 4.4))
ax = best_offline.plot(
    kind="bar",
    x="Metric",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color="#1f7a5b",
)
ax.set_title("Best Offline Model Scores")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 0.56)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=18, ha="right")
add_bar_labels(ax)
savefig("best_offline_scores.png")


plt.figure(figsize=(8.8, 4.8))
summary_for_plot = best_summary.copy()
summary_for_plot["Label"] = summary_for_plot["Evaluation"] + "\n" + summary_for_plot["Best Metric"]
ax = summary_for_plot.plot(
    kind="bar",
    x="Label",
    y="Score",
    legend=False,
    ax=plt.gca(),
    color=["#1f7a5b", "#1f7a5b", "#4467a6", "#4467a6", "#d9902f", "#d9902f", "#8a5a99", "#8a5a99"],
)
ax.set_title("Best Observed Scores by Evaluation Setting")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=25, ha="right")
add_bar_labels(ax)
savefig("best_score_summary.png")


print(f"Wrote Chapter 4 assets to {OUT_DIR.resolve()}")
