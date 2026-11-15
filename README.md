# Human Preferences for LLM Responses in Chatbot Arena

Statistical Learning course project (R). Using about 57,000 pairwise "battles" from Chatbot Arena, it asks whether people vote for the better language model or for the longer, more heavily formatted answer.

The project consists of two separate reports:

| Report | Question | Methods |
|---|---|---|
| **1. Supervised** | How well do answer length and formatting predict the vote, and do they still matter once we know which models were compared? | Logistic regression, Lasso, ridge, random forest, Bradley–Terry model with style control, prompt-level cluster bootstrap |
| **2. Unsupervised** | Do prompts fall into topic groups, and does the preference for longer answers differ between them? | TF-IDF, latent semantic analysis (truncated SVD), K-means vs Ward hierarchical clustering, silhouette, Adjusted Rand Index, likelihood-ratio test |

## Main findings

**Supervised**
- In battles where the two answers differ in length, the longer answer wins 62% of the time. Doubling the relative length multiplies the odds of winning by about 1.3, even after controlling for model identity.
- On the test set, style features alone reach about 63% accuracy. A simple "pick the longer answer" rule does almost as well as a random forest.
- Adding model identity raises accuracy to about 68%. Once identity is included, the apparent effects of headers and bold text largely disappear.
- Adjusting for style leaves the top of the leaderboard unchanged but reshuffles the middle (Kendall's τ = 0.86).

**Unsupervised**
- Prompts form weak, overlapping clusters (silhouette ≤ 0.07). K-means (K = 15) beats Ward but agrees with it only weakly (ARI 0.18).
- The clusters mainly reflect task types: creative writing, coding, explanations, logic puzzles, games and so on.
- The length preference differs significantly between clusters (LRT χ² = 153.6, 14 df). Doubling length multiplies the odds of winning by 1.62 for explanatory prompts but has no detectable effect for logic puzzles (1.01).

## Repository structure

```
├── 01_dataprep.R            # download, parse and filter the data; build style features
├── 02_Supervised.R          # supervised models, ranking, bootstrap, figures
├── 03_Unsupervised.R        # TF-IDF, LSA, clustering, link to preferences, figures
├── report_supervised.Rmd    # Report 1 (R Markdown source)
├── report_unsupervised.Rmd  # Report 2 (R Markdown source)
├── reports/
│   ├── report_supervised.pdf
│   └── report_unsupervised.pdf
└── figures/                 # PNG figures produced by the scripts
```

## How to reproduce

1. Install R (≥ 4.2) and the packages:

   ```r
   install.packages(c("data.table", "jsonlite", "stringr", "glmnet", "ranger",
                      "text2vec", "stopwords", "irlba", "cluster", "mclust",
                      "Matrix", "knitr", "rmarkdown"))
   ```

2. Run the scripts in order from the project folder:

   ```r
   source("01_dataprep.R")     # downloads train.csv (~184 MB) and writes arena_clean.rds
   source("02_Supervised.R")   # writes results_supervised.rds and figures
   source("03_Unsupervised.R") # writes results_unsupervised.rds and figures
   ```

3. Knit the reports (a LaTeX installation is needed, e.g. `tinytex`):

   ```r
   rmarkdown::render("report_supervised.Rmd")
   rmarkdown::render("report_unsupervised.Rmd")
   ```

Random seeds are fixed (`set.seed(2026)`), so the results are reproducible. The bootstrap and clustering steps take several minutes.

## Data

The raw data are **not included** in this repository. `01_dataprep.R` downloads them automatically from Hugging Face:
[lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k). This dataset was originally released as the training data of the Kaggle competition [LMSYS – Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena).

After filtering (single-turn battles, mainly ASCII text, models with at least 200 battles), the analysis uses 49,438 battles, of which 33,995 are non-tied.

## References

- Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.
- Chiang, W.-L. et al. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. *ICML 2024*, PMLR 235, 8359–8388.
- Chiang, W.-L. et al. (2024). LMSYS – Chatbot Arena Human Preference Predictions. Kaggle competition.
- Deerwester, S. et al. (1990). Indexing by latent semantic analysis. *JASIS*, 41(6), 391–407.
- Hubert, L. and Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218.
- James, G., Witten, D., Hastie, T. and Tibshirani, R. (2021). *An Introduction to Statistical Learning*, 2nd ed. Springer.
- Li, T., Angelopoulos, A. and Chiang, W.-L. (2024). Does style matter? Disentangling style and substance in Chatbot Arena. LMSYS blog.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53–65.

## Author

[Doan Duy My Phuong] – Statistical Learning course project, 2026
