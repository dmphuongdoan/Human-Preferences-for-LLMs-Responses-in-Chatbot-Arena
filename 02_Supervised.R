# =============================================================
# 02_supervised.R  -  Model comparison
# Input : arena_clean.rds (from 01_data_prep.R)
# Output: results_supervised.rds
# =============================================================
# install.packages(c("data.table", "glmnet", "ranger"))
library(data.table)
library(glmnet)
library(ranger)
set.seed(2026)

d  <- readRDS("arena_clean.rds")
dn <- d[outcome != "tie"] # binary task: does A win?
dn[, y := as.integer(outcome == "A")]
feats <- c("d_len", "d_bullets", "d_headers", "d_bold", "d_code")

# --- 1. Train/test split by PROMPT (no prompt in both sets) ---
prompts <- unique(dn$prompt)
test_p  <- sample(prompts, round(0.2 * length(prompts)))
tr <- dn[!prompt %in% test_p]
te <- dn[ prompt %in% test_p]
cat("Train:", nrow(tr), "battles | Test:", nrow(te), "battles\n")

# --- 2. Cap extreme differences at train 1st/99th percentiles -
caps <- sapply(feats, function(v) quantile(tr[[v]], c(0.01, 0.99)))
cat("\nCaps (1st / 99th percentile of train):\n"); print(round(caps, 3))
for (v in feats) {
  if (caps[1, v] == caps[2, v]) {# rare features: capping would make them constant
    cat("Skipping caps for", v, "(1st and 99th percentile are equal)\n"); next
  }
  set(tr, j = v, value = pmin(pmax(tr[[v]], caps[1, v]), caps[2, v]))
  set(te, j = v, value = pmin(pmax(te[[v]], caps[1, v]), caps[2, v]))
}

# --- 3. Evaluation helpers -----------------------------------
logloss <- function(y, p) { p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
-mean(y * log(p) + (1 - y) * log(1 - p)) }
evalm <- function(name, p) data.table(model = name,
                                      log_loss = round(logloss(te$y, p), 4),
                                      accuracy = round(mean((p > 0.5) == te$y), 4))

# Cross-validation folds for glmnet, also grouped by prompt
up     <- unique(tr$prompt)
fold   <- sample(rep(1:10, length.out = length(up)))
foldid <- fold[match(tr$prompt, up)]

Xtr <- as.matrix(tr[, ..feats]); Xte <- as.matrix(te[, ..feats])

# --- 4. Models -----------------------------------------------
# (a) Baseline: always predict the training share of A wins
p_base <- rep(mean(tr$y), nrow(te))

# (b) Logistic regression on style features
fml   <- as.formula(paste("y ~", paste(feats, collapse = " + ")))
m_glm <- glm(fml, data = tr, family = binomial)
p_glm <- predict(m_glm, te, type = "response")

# (c) Lasso logistic regression (lambda chosen by grouped CV)
m_lasso <- cv.glmnet(Xtr, tr$y, family = "binomial", alpha = 1, foldid = foldid)
p_lasso <- as.numeric(predict(m_lasso, Xte, s = "lambda.min", type = "response"))

# (d) Style + model identity (Bradley-Terry), ridge on model terms only
models <- sort(unique(c(dn$model_a, dn$model_b)))
model_matrix <- function(D) {
  M <- matrix(0, nrow(D), length(models), dimnames = list(NULL, models))
  M[cbind(seq_len(nrow(D)), match(D$model_a, models))] <-  1
  M[cbind(seq_len(nrow(D)), match(D$model_b, models))] <- -1
  M
}
Ztr <- cbind(Xtr, model_matrix(tr)); Zte <- cbind(Xte, model_matrix(te))
pf  <- c(rep(0, length(feats)), rep(1, length(models)))   # do not penalise style terms
lam_grid <- 10^seq(0, -6, length.out = 100)   # default ridge grid stops too early
m_bt <- cv.glmnet(Ztr, tr$y, family = "binomial", alpha = 0, foldid = foldid,
                  penalty.factor = pf, standardize = FALSE, lambda = lam_grid)
p_bt <- as.numeric(predict(m_bt, Zte, s = "lambda.min", type = "response"))
cat("\nRidge lambda.min:", signif(m_bt$lambda.min, 3),
    if (m_bt$lambda.min %in% range(lam_grid)) "(WARNING: at the edge of the grid)" else "(inside the grid)", "\n")

# (d') Same model WITHOUT penalty (one model column is dropped: identifiability)
fit_id <- glm.fit(cbind(1, Ztr), tr$y, family = binomial())
b_id   <- fit_id$coefficients; b_id[is.na(b_id)] <- 0
p_id   <- as.numeric(plogis(cbind(1, Zte) %*% b_id))
cat("Unpenalised model: dropped column (reference model):",
    colnames(Ztr)[is.na(fit_id$coefficients[-1])], "\n")

# (e) Random forest on style features
rf_dat <- data.frame(yf = factor(tr$y), tr[, ..feats])
m_rf <- ranger(yf ~ ., data = rf_dat, probability = TRUE, num.trees = 500,
               min.node.size = 50, importance = "permutation", seed = 2026)
p_rf <- predict(m_rf, data.frame(te[, ..feats]))$predictions[, "1"]

# (f) Random forest on style + model identity (model names as unordered factors)
rf_dat2 <- data.frame(yf = factor(tr$y), tr[, ..feats],
                      model_a = factor(tr$model_a, levels = models),
                      model_b = factor(tr$model_b, levels = models))
m_rf2 <- ranger(yf ~ ., data = rf_dat2, probability = TRUE, num.trees = 500,
                min.node.size = 50, respect.unordered.factors = "order", seed = 2026)
te_rf2 <- data.frame(te[, ..feats],
                     model_a = factor(te$model_a, levels = models),
                     model_b = factor(te$model_b, levels = models))
p_rf2 <- predict(m_rf2, te_rf2)$predictions[, "1"]

# --- 5. Results ----------------------------------------------
res <- rbind(evalm("Baseline (class share)",        p_base),
             evalm("Logistic regression (style)",   p_glm),
             evalm("Lasso (style)",                 p_lasso),
             evalm("Logistic (style + model identity, no penalty)", p_id),
             evalm("Ridge (style + model identity)", p_bt),
             evalm("Random forest (style)",         p_rf),
             evalm("Random forest (style + model identity)", p_rf2))
cat("\nTest-set performance (lower log loss is better):\n"); print(res)

# Simple rule "pick the longer answer" vs fitted models, on the SAME test battles
neq <- te$d_len != 0
rule_acc <- mean((te$d_len[neq] > 0) == (te$y[neq] == 1))
acc_on   <- function(p) mean((p[neq] > 0.5) == te$y[neq])
rule_cmp <- c(n = sum(neq), rule = rule_acc, logistic = acc_on(p_glm), rf = acc_on(p_rf))
cat("\nTest battles with unequal length:", sum(neq), "of", nrow(te), "\n")
cat("Accuracy on these battles - longer-answer rule:", round(rule_acc, 4),
    "| logistic (style):", round(acc_on(p_glm), 4),
    "| random forest (style):", round(acc_on(p_rf), 4), "\n")

coefs <- data.table(
  feature           = feats,
  logistic          = round(coef(m_glm)[feats], 4),
  lasso             = round(as.numeric(coef(m_lasso, s = "lambda.min")[feats, 1]), 4),
  with_model_ident  = round(as.numeric(coef(m_bt,    s = "lambda.min")[feats, 1]), 4))
coefs[, `:=`(OR_logistic = round(exp(logistic), 3),
             OR_with_ident = round(exp(with_model_ident), 3))]
cat("\nStyle coefficients (does the length effect survive model identity?):\n"); print(coefs)
cat("Note: OR columns are per unit of each feature; for d_len one unit = e-fold (x2.72) length ratio.\n")
cat("Odds ratio when A is TWICE as long as B - style only:",
    round(exp(coefs[feature == "d_len", logistic] * log(2)), 3),
    "| with model identity:",
    round(exp(coefs[feature == "d_len", with_model_ident] * log(2)), 3), "\n")

# Standardised coefficients: effect of a one-SD change (comparable across features)
sds <- sapply(feats, function(v) sd(tr[[v]])) # SDs of the capped training features
coefs[, `:=`(sd = round(sds[feature], 3),
             std_logistic   = round(logistic * sds[feature], 3),
             std_with_ident = round(with_model_ident * sds[feature], 3))]
cat("\nStandardised coefficients (per 1 SD of each feature):\n")
print(coefs[, .(feature, sd, std_logistic, std_with_ident)])

cat("\nRandom forest permutation importance:\n")
print(round(sort(importance(m_rf), decreasing = TRUE), 5))

n_battles <- table(c(dn$model_a, dn$model_b))
beta <- as.numeric(coef(m_bt, s = "lambda.min")[models, 1])
ranking <- data.table(model  = models,
                      score  = round(1000 + 400 / log(10) * beta),
                      n_nontie = as.integer(n_battles[models]))[order(-score)]
# (train-only ridge ranking kept in the saved results; the report uses the full-data ranking in 5b)

# --- 5b. Rankings with and without style control + bootstrap CI -----
# Fitted on ALL non-tie battles (train + test, capped), unpenalised logistic.
full <- rbind(tr, te)
Mf <- model_matrix(full); Xf <- as.matrix(full[, ..feats])
center_score <- function(b) 1000 + 400 / log(10) * (b - mean(b))   # 1000 = average model
fit_models <- function(X, y) {# returns model coefficients (reference = 0)
  b <- glm.fit(cbind(1, X), y, family = binomial())$coefficients
  names(b) <- c("(Intercept)", colnames(X))
  b <- b[colnames(Mf)]
  n_na <- sum(is.na(b)) # 1 = only the reference model; > 1 = not identified
  b[is.na(b)] <- 0
  attr(b, "n_na") <- n_na
  b
}
rk <- data.table(model      = models,
                 raw        = round(center_score(fit_models(Mf, full$y))),
                 style_ctrl = round(center_score(fit_models(cbind(Xf, Mf), full$y))),
                 n_nontie   = as.integer(n_battles[models]))
rk[, `:=`(rank_raw   = frank(-raw,        ties.method = "min"),
          rank_style = frank(-style_ctrl, ties.method = "min"))]
rk[, rank_change := rank_raw - rank_style]  # > 0: moves UP after style control
cat("\nKendall's tau between raw and style-controlled scores:",
    round(cor(rk$raw, rk$style_ctrl, method = "kendall"), 3), "\n")
cat("Largest moves after style control (positive = moves up):\n")
print(rk[order(-abs(rank_change))][1:8, .(model, raw, style_ctrl, rank_raw, rank_style, rank_change)])

# Bootstrap (200 replicates), refitting the style-controlled model each time
# Cluster bootstrap: resample PROMPTS with replacement and keep all their battles,
# because battles sharing a prompt are not independent.
B <- 200
rows_by_prompt <- split(seq_len(nrow(full)), full$prompt)
n_na_boot <- integer(B)
boot <- sapply(seq_len(B), function(r) {
  i <- unlist(rows_by_prompt[sample(length(rows_by_prompt), replace = TRUE)], use.names = FALSE)
  b <- fit_models(cbind(Xf, Mf)[i, ], full$y[i])
  n_na_boot[r] <<- attr(b, "n_na")
  center_score(b)
}) # rows = models, columns = replicates
cat("\nBootstrap replicates by number of undetermined model coefficients (1 = connected):\n")
print(table(n_na_boot))
if (any(n_na_boot > 1)) {# drop replicates where strengths are not identified
  cat("Dropping", sum(n_na_boot > 1), "replicates with a disconnected comparison graph\n")
  boot <- boot[, n_na_boot == 1, drop = FALSE]
}
boot_rank <- apply(boot, 2, function(s) rank(-s, ties.method = "min"))
rk[, `:=`(lo = round(apply(boot, 1, quantile, 0.025)), hi = round(apply(boot, 1, quantile, 0.975)),
          rank_lo = apply(boot_rank, 1, quantile, 0.025, type = 1), rank_hi = apply(boot_rank, 1, quantile, 0.975, type = 1))]
cat("\nStyle-controlled ranking with 95% bootstrap intervals (top 15):\n")
print(rk[order(-style_ctrl)][1:15, .(model, style_ctrl, lo, hi, rank_style, rank_lo, rank_hi, n_nontie)])

# --- 6. Figures for the report --------------------------------
# (1) Model comparison on the test set
png("fig_model_comparison.png", width = 1300, height = 450)
par(mfrow = c(1, 2), mar = c(4, 19, 3, 1))
o <- rev(seq_len(nrow(res)))
barplot(res$log_loss[o], names.arg = res$model[o], horiz = TRUE, las = 1,
        xlim = c(0.55, 0.70), xpd = FALSE, main = "Test log loss (lower is better)")
abline(v = res$log_loss[1], lty = 2)
barplot(res$accuracy[o], names.arg = res$model[o], horiz = TRUE, las = 1,
        xlim = c(0.45, 0.70), xpd = FALSE, main = "Test accuracy (higher is better)")
abline(v = 0.5, lty = 2)
dev.off()

# (2) Length effect: observed win rate (test deciles) vs fitted curves
br <- unique(quantile(te$d_len, seq(0, 1, by = 0.1)))
te[, len_bin := cut(d_len, br, include.lowest = TRUE)]
emp  <- te[, .(x = mean(d_len), win = mean(y)), by = len_bin]
grid <- data.frame(d_len = seq(-3, 3, length.out = 200),
                   d_bullets = 0, d_headers = 0, d_bold = 0, d_code = 0)
b_bt    <- coef(m_bt, s = "lambda.min")
p_style <- predict(m_glm, grid, type = "response")
p_ident <- plogis(b_bt[1] + b_bt["d_len", 1] * grid$d_len)   # average model matchup
png("fig_length_effect.png", width = 800, height = 600)
plot(exp(grid$d_len), p_style, log = "x", type = "l", lwd = 2, ylim = c(0.25, 0.75),
     xlab = "Length ratio A / B (log scale)", ylab = "P(A wins)",
     main = "Probability of winning vs relative length")
lines(exp(grid$d_len), p_ident, lwd = 2, lty = 2, col = "firebrick")
points(exp(emp$x), emp$win, pch = 19, col = "grey40")
abline(h = 0.5, v = 1, col = "grey80")
legend("topleft", bty = "n", lwd = c(2, 2, NA), lty = c(1, 2, NA), pch = c(NA, NA, 19),
       col = c("black", "firebrick", "grey40"),
       legend = c("Style-only logistic", "Style + model identity (equal-strength matchup)", "Observed (test deciles)"))
dev.off()

# (3) Standardised style coefficients with 95% CI, with and without model identity
m_id_glm <- glm(tr$y ~ Ztr, family = binomial) # unpenalised, for standard errors
est1 <- coef(m_glm)[feats] * sds
se1  <- sqrt(diag(vcov(m_glm)))[feats] * sds
est2 <- coef(m_id_glm)[paste0("Ztr", feats)] * sds
se2  <- sqrt(diag(vcov(m_id_glm)))[paste0("Ztr", feats)] * sds

# Inference table: estimate, SE and p-value per unit, both models (training set, capped features)
b_s  <- coef(m_glm)[feats];                    s_s <- sqrt(diag(vcov(m_glm)))[feats]
b_i  <- coef(m_id_glm)[paste0("Ztr", feats)];  s_i <- sqrt(diag(vcov(m_id_glm)))[paste0("Ztr", feats)]
coef_inf <- data.table(feature = feats,
                       b_style = round(b_s, 4), se_style = round(s_s, 4), p_style = signif(2 * pnorm(-abs(b_s / s_s)), 3),
                       b_ident = round(b_i, 4), se_ident = round(s_i, 4), p_ident = signif(2 * pnorm(-abs(b_i / s_i)), 3))
cat("\nStyle coefficients with standard errors and p-values (identity model unpenalised):\n")
print(coef_inf)
png("fig_style_coefficients.png", width = 850, height = 500)
par(mar = c(5, 8, 3, 1)); yy <- seq_along(feats)
plot(est1, yy + 0.1, pch = 19, yaxt = "n", ylab = "",
     xlim = range(c(0, est1 - 1.96 * se1, est1 + 1.96 * se1, est2 - 1.96 * se2, est2 + 1.96 * se2)),
     ylim = c(0.5, length(feats) + 0.5),
     xlab = "Standardised coefficient (log-odds per 1 SD)", main = "Style effects with 95% CI")
segments(est1 - 1.96 * se1, yy + 0.1, est1 + 1.96 * se1, yy + 0.1)
points(est2, yy - 0.1, pch = 17, col = "firebrick")
segments(est2 - 1.96 * se2, yy - 0.1, est2 + 1.96 * se2, yy - 0.1, col = "firebrick")
axis(2, at = yy, labels = feats, las = 1); abline(v = 0, lty = 2)
legend("bottomright", bty = "n", pch = c(19, 17), col = c("black", "firebrick"),
       legend = c("Style only", "Style + model identity"))
dev.off()

# (4) Style-controlled ranking, top 20 models
top <- rk[order(-style_ctrl)][1:20][.N:1] # reversed so the best model is on top
png("fig_ranking.png", width = 900, height = 650)
par(mar = c(5, 15, 3, 1)); yy <- seq_len(nrow(top))
plot(top$style_ctrl, yy, pch = 19, yaxt = "n", ylab = "",
     xlim = range(c(top$lo, top$hi, top$raw)),
     xlab = "Score (Elo-like scale, 1000 = average model)",
     main = "Top 20 models: style-controlled score with 95% bootstrap CI")
segments(top$lo, yy, top$hi, yy)
points(top$raw, yy, pch = 4, col = "firebrick")
axis(2, at = yy, labels = top$model, las = 1, cex.axis = 0.8)
legend("bottomright", bty = "n", pch = c(19, 4), col = c("black", "firebrick"),
       legend = c("Style-controlled", "Not controlled (identity only)"))
dev.off()
cat("\nFigures saved: fig_model_comparison.png, fig_length_effect.png,",
    "fig_style_coefficients.png, fig_ranking.png\n")

saveRDS(list(results = res, coefs = coefs, coef_inference = coef_inf, ranking = ranking, rankings_full = rk,
             boot_scores = boot, caps = caps),
        "results_supervised.rds")