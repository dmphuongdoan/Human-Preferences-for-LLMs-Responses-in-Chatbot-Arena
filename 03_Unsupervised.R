# =============================================================
# 03_unsupervised.R  -  Topic clustering of Chatbot Arena prompts
# Input : arena_clean.rds (from 01_data_prep.R)
# Output: results_unsupervised.rds + 3 figures (.png)
# =============================================================
# install.packages(c("data.table", "text2vec", "stopwords", "irlba", "cluster", "mclust"))
library(data.table)
library(text2vec)
library(irlba)
library(cluster)
library(mclust)
set.seed(2026)

d  <- readRDS("arena_clean.rds")
up <- unique(d[, .(prompt)]) # cluster each distinct prompt once
cat("Battles:", nrow(d), "| distinct prompts:", nrow(up), "\n")

# --- 1. Text -> TF-IDF matrix ----------------------------------
# lower case, keep letters only, drop words of 1-2 letters
prep <- function(x) gsub("\\b[a-z]{1,2}\\b", " ", gsub("[^a-z ]", " ", tolower(x)))
it    <- itoken(up$prompt, preprocessor = prep, tokenizer = word_tokenizer, progressbar = FALSE)
vocab <- create_vocabulary(it, stopwords = stopwords::stopwords("en"))
vocab <- prune_vocabulary(vocab, term_count_min = 20, doc_proportion_max = 0.3)
dtm   <- create_dtm(it, vocab_vectorizer(vocab))
keep  <- Matrix::rowSums(dtm) > 0
cat("Vocabulary:", nrow(vocab), "terms | prompts without any vocabulary term (dropped):",
    sum(!keep), "\n")
up  <- up[keep]; dtm <- dtm[keep, ]
X   <- fit_transform(dtm, TfIdf$new(norm = "l2"))

# --- 2. Dimension reduction: truncated SVD (LSA) --------------
n_comp <- 100
sv <- irlba(X, nv = n_comp)
Z  <- sv$u %*% diag(sv$d)
cat("Share of TF-IDF squared norm retained by", n_comp, "components:",
    round(sum(sv$d^2) / sum(X@x^2), 3), "\n")
Z  <- Z / pmax(sqrt(rowSums(Z^2)), 1e-12)   # unit length: Euclidean ~ cosine distance

# --- 3. Choosing the number of clusters K ---------------------
sub  <- sample(nrow(Z), 5000) # silhouette and Ward on a subsample
Dsub <- dist(Z[sub, ])
ks   <- 4:15
sel  <- rbindlist(lapply(ks, function(k) {
  km <- kmeans(Z, centers = k, nstart = 10, iter.max = 100, algorithm = "MacQueen")
  data.table(k = k, within_ss = round(km$tot.withinss),
             silhouette = round(mean(silhouette(km$cluster[sub], Dsub)[, "sil_width"]), 4))
}))
cat("\nChoice of K (higher silhouette is better):\n"); print(sel)
K <- sel[which.max(silhouette), k] # silhouette rises almost monotonically: this is just the edge of the range
K_manual <- 15  # K fixed at 15 for interpretability, after inspecting silhouette and elbow
if (!is.na(K_manual)) K <- K_manual
cat("Chosen K:", K, "\n")

# --- 4. Final clusterings: K-means vs Ward hierarchical -------
km <- kmeans(Z, centers = K, nstart = 25, iter.max = 100, algorithm = "MacQueen")
up[, cluster := km$cluster]
hc    <- hclust(Dsub, method = "ward.D2")
cl_hc <- cutree(hc, k = K)
cmp <- data.table(method = c("K-means", "Hierarchical (Ward)"),
                  silhouette = round(c(mean(silhouette(km$cluster[sub], Dsub)[, "sil_width"]),
                                       mean(silhouette(cl_hc, Dsub)[, "sil_width"])), 4))
cat("\nMethod comparison on the same 5,000 prompts:\n"); print(cmp)
cat("Adjusted Rand Index (agreement K-means vs Ward):",
    round(adjustedRandIndex(km$cluster[sub], cl_hc), 3), "\n")

# --- 5. Describe the clusters ---------------------------------
terms <- colnames(X)
desc <- rbindlist(lapply(1:K, function(c) {
  m <- Matrix::colMeans(X[km$cluster == c, , drop = FALSE])
  data.table(cluster = c, prompts = sum(km$cluster == c),
             top_terms = paste(terms[order(-m)][1:10], collapse = ", "))
}))[order(-prompts)]
cat("\nClusters and their most characteristic terms:\n"); print(desc)

cat("\nThree random prompts per cluster:\n")
for (c in desc$cluster) {
  s <- up[cluster == c, prompt]
  s <- substr(gsub("\\s+", " ", sample(s, min(3, length(s)))), 1, 120)
  cat("\n--- Cluster", c, "---\n"); cat(paste0("* ", s), sep = "\n")
}

# --- 6. Link to preferences: does the length effect differ by topic? --
d2  <- merge(d, up[, .(prompt, cluster)], by = "prompt")
ties <- d2[, .(battles = .N, tie_rate = round(mean(outcome == "tie"), 3)), by = cluster]
link <- d2[outcome != "tie", {
  m <- glm(I(outcome == "A") ~ d_len + d_bullets + d_headers + d_bold + d_code,
           family = binomial, data = .SD)
  .(longer_wins = round(mean(((d_len > 0) == (outcome == "A"))[d_len != 0]), 3),
    b_len       = round(coef(m)["d_len"], 3),
    se_len      = round(sqrt(vcov(m)["d_len", "d_len"]), 3),
    OR_double   = round(exp(coef(m)["d_len"] * log(2)), 3))
}, by = cluster]
link <- merge(ties, link, by = "cluster")[order(-longer_wins)]
cat("\nLength preference by topic cluster:\n"); print(link)

# --- 6b. Stability of K-means: ARI between runs with different seeds ---
stab <- sapply(1:10, function(s) {
  set.seed(100 + s)
  adjustedRandIndex(km$cluster,
                    kmeans(Z, centers = K, nstart = 10, iter.max = 100, algorithm = "MacQueen")$cluster)
})
cat("\nK-means stability (ARI vs final solution, 10 reruns): mean", round(mean(stab), 3),
    "| min", round(min(stab), 3), "| max", round(max(stab), 3), "\n")

# --- 6c. Formal test: does the length effect differ across clusters? ---
# Controls for model identity; compares one common length slope vs one slope per cluster.
dn2 <- d2[outcome != "tie"]
dn2[, `:=`(y = as.integer(outcome == "A"), cl = factor(cluster))]
mods <- sort(unique(c(dn2$model_a, dn2$model_b)))
M <- matrix(0, nrow(dn2), length(mods))
M[cbind(seq_len(nrow(dn2)), match(dn2$model_a, mods))] <-  1
M[cbind(seq_len(nrow(dn2)), match(dn2$model_b, mods))] <- -1
f0 <- glm(y ~ d_len + d_bullets + d_headers + d_bold + d_code + M, family = binomial, data = dn2)
f1 <- glm(y ~ d_len:cl + d_bullets + d_headers + d_bold + d_code + M, family = binomial, data = dn2)
lrt <- anova(f0, f1, test = "Chisq")
cat("\nLikelihood-ratio test, common vs cluster-specific length slopes (with model identity):\n")
print(lrt)

# Robustness: add cluster main effects. In this paired design the outcome is "A wins" and the
# predictors are A - B differences, so a cluster main effect only means a cluster-specific
# position (left/right) bias. Both models below include it; the test isolates the slopes.
f0c <- update(f0, . ~ . + cl)
f1c <- update(f1, . ~ . + cl)
cat("\nRobustness: same test with cluster main effects (cluster-specific position bias):\n")
print(anova(f0c, f1c, test = "Chisq"))
cat("Test of the cluster main effects themselves (are position biases different?):\n")
print(anova(f1, f1c, test = "Chisq"))

sl <- grep("^d_len:cl", names(coef(f1)), value = TRUE)
V  <- vcov(f1)
slopes <- data.table(cluster   = as.integer(sub("d_len:cl", "", sl)),
                     b_len_id  = round(coef(f1)[sl], 3),
                     se_id     = round(sqrt(diag(V)[sl]), 3),
                     OR_double_id = round(exp(coef(f1)[sl] * log(2)), 3))
link <- merge(link, slopes, by = "cluster")[order(-OR_double_id)]
cat("\nLength effect by cluster, with model identity:\n")
print(link[, .(cluster, battles, longer_wins, OR_double, OR_double_id, b_len_id, se_id)])

# Pairwise contrast between two clusters (cluster numbers as in this run; edit if they change)
contrast <- function(a, b) {
  ca <- paste0("d_len:cl", a); cb <- paste0("d_len:cl", b)
  diff <- coef(f1)[ca] - coef(f1)[cb]
  se   <- sqrt(V[ca, ca] + V[cb, cb] - 2 * V[ca, cb])
  cat("Cluster", a, "vs", b, ": difference in length slope =", round(diff, 3),
      "| SE =", round(se, 3), "| z =", round(diff / se, 2), "\n")
}
cat("\nContrasts (explanations vs logic puzzles / step-by-step reasoning):\n")
contrast(8, 7); contrast(8, 11)

# --- 7. Figures for the report --------------------------------
cols <- hcl.colors(K, "Dark 3")
png("fig_k_selection.png", width = 1000, height = 420)
par(mfrow = c(1, 2))
plot(sel$k, sel$within_ss, type = "b", pch = 19, xlab = "K", ylab = "Total within-cluster SS",
     main = "Elbow plot")
plot(sel$k, sel$silhouette, type = "b", pch = 19, xlab = "K", ylab = "Mean silhouette width",
     main = "Silhouette (5,000 prompts)")
dev.off()

png("fig_lsa_clusters.png", width = 750, height = 650)
plot(Z[sub, 1], Z[sub, 2], col = cols[km$cluster[sub]], pch = 20, cex = 0.6,
     xlab = "LSA component 1", ylab = "LSA component 2", main = "Prompts in the first two LSA dimensions")
legend("bottomleft", legend = paste("Cluster", 1:K), col = cols, pch = 19, cex = 0.7, bty = "n")
dev.off()

png("fig_dendrogram.png", width = 1000, height = 500)
plot(hc, labels = FALSE, hang = -1, main = "Ward dendrogram (5,000 prompts)", xlab = "", sub = "")
rect.hclust(hc, k = K, border = cols)
dev.off()

saveRDS(list(k_selection = sel, comparison = cmp, clusters = desc, link = link,
             stability_ari = stab, lr_test = lrt,
             assignment = up), "results_unsupervised.rds")