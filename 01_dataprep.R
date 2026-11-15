# =============================================================
# 01_data_prep.R  -  Load, clean and describe Arena 55k data
# =============================================================
# install.packages(c("data.table", "jsonlite", "stringr"))
library(data.table) # read and processing big data table, fast
library(jsonlite) # read JSON
library(stringr) # count, find character in the text

# --- 1. Load --------------------------------------------------
url <- paste0("https://huggingface.co/datasets/lmarena-ai/",
              "arena-human-preference-55k/resolve/main/train.csv")
options(timeout = 1800)   # default 60 s is too short for 184 MB, max time = 60s*30m = 1800
if (file.exists("train.csv") && file.size("train.csv") < 1.5e8) file.remove("train.csv")
if (!file.exists("train.csv")) download.file(url, "train.csv", mode = "wb")
cat("File size (MB):", round(file.size("train.csv") / 1e6), "\n")   # should be ~184
raw <- fread("train.csv", encoding = "UTF-8")
cat("Rows:", nrow(raw), "\n"); print(names(raw))

# --- 2. Parse JSON text columns -------------------------------
# Each text column is a JSON list (one element per turn).
parse_turns <- function(s) {
  s <- gsub('""', '"', s, fixed = TRUE)   # fread keeps CSV-escaped quotes ("") -> restore valid JSON
  x <- tryCatch(fromJSON(s), error = function(e) NULL)
  if (is.null(x) || length(x) == 0) return(list(n = NA_integer_, first = NA_character_))
  list(n = length(x), first = as.character(x[1]))
}
p  <- lapply(raw$prompt,     parse_turns)
ra <- lapply(raw$response_a, parse_turns)
rb <- lapply(raw$response_b, parse_turns)

dt <- data.table(
  model_a  = raw$model_a, model_b = raw$model_b,
  n_turns  = sapply(p, `[[`, "n"),
  prompt   = sapply(p,  `[[`, "first"),
  resp_a   = sapply(ra, `[[`, "first"),
  resp_b   = sapply(rb, `[[`, "first"),
  outcome  = fcase(raw$winner_model_a == 1, "A",
                   raw$winner_model_b == 1, "B",
                   default = "tie")
)
# Remove invalid UTF-8 bytes (otherwise nchar() fails with "invalid multibyte string")
for (col in c("prompt", "resp_a", "resp_b"))
  set(dt, j = col, value = iconv(dt[[col]], "UTF-8", "UTF-8", sub = ""))

# --- 3. Descriptive statistics (for the Data section) ---------
cat("\nOutcome distribution:\n"); print(prop.table(table(dt$outcome)))
cat("\nMulti-turn share:", mean(dt$n_turns > 1, na.rm = TRUE), "\n")
cat("Missing responses:", sum(is.na(dt$resp_a) | is.na(dt$resp_b)), "\n")
cat("Distinct prompts:", uniqueN(dt$prompt), "of", nrow(dt), "\n")
games <- sort(table(c(dt$model_a, dt$model_b)), decreasing = TRUE)
cat("\nBattles per model (top 10):\n"); print(head(games, 10))

# --- 4. Filter --------------------------------------------------
# Single-turn, non-missing, (mostly) English, models with >= 200 battles.
ascii_share <- function(s) nchar(gsub("[^ -~]", "", s)) / pmax(nchar(s), 1)
keep_models <- names(games)[games >= 200]
d <- dt[n_turns == 1 & !is.na(resp_a) & !is.na(resp_b) &
          ascii_share(prompt) > 0.9 &
          model_a %in% keep_models & model_b %in% keep_models]
cat("\nRows after filtering:", nrow(d), "\n")

# --- 5. ----
# a. Style features (difference A - B) ---------------------
count <- function(s, pattern) str_count(s, regex(pattern, multiline = TRUE))
feat <- function(s) data.table(
  len     = log(nchar(s) + 1),
  bullets = count(s, "^\\s*([-*+]|\\d+\\.)\\s"),
  headers = count(s, "^#{1,6}\\s"),
  bold    = count(s, "\\*\\*[^*]+\\*\\*"),
  code = count(s, "^\\s*```") %/% 2
)
fa <- feat(d$resp_a); fb <- feat(d$resp_b)
diffs <- fa - fb; setnames(diffs, paste0("d_", names(diffs)))
d <- cbind(d, diffs)

# b. Describe style features (for the Data section) -------
raw_feat <- function(s) data.table(
  chars   = nchar(s),
  bullets = count(s, "^\\s*([-*+]|\\d+\\.)\\s"),
  headers = count(s, "^#{1,6}\\s"),
  bold    = count(s, "\\*\\*[^*]+\\*\\*"),
  code = count(s, "^\\s*```") %/% 2
)
f <- rbind(raw_feat(d$resp_a), raw_feat(d$resp_b))   # every answer, A and B pooled
cat("\nStyle features per answer (median / mean / share using it):\n")
print(f[, lapply(.SD, function(x) c(median = median(x), mean = round(mean(x), 2),
                                    share_used = round(mean(x > 0), 3)))])
cat("\nDifferences A - B (should be centred near 0):\n")
print(summary(d[, .(d_len, d_bullets, d_headers, d_bold, d_code)]))

# --- 6. First look --------------------------------------------
dn <- d[outcome != "tie"]
eq <- dn$d_len == 0
cat("\nNon-tied battles with answers of exactly equal length:", sum(eq), "\n")
cat("Share of games won by the LONGER answer (equal-length battles excluded):",
    round(mean((dn$d_len[!eq] > 0) == (dn$outcome[!eq] == "A")), 4), "\n")
m0 <- glm(I(outcome == "A") ~ d_len + d_bullets + d_headers + d_bold + d_code,
          data = dn, family = binomial)
print(summary(m0))

# Odds ratios with 95% confidence intervals (exp of log-odds coefficients)
cat("\nOdds ratios (per unit increase) with 95% CI:\n")
print(round(exp(cbind(OR = coef(m0), confint.default(m0))), 3))
cat("\nOdds ratio when A is TWICE as long as B:",
    round(exp(coef(m0)["d_len"] * log(2)), 3), "\n")

# Predicted P(A wins) when lengths are equal, A 2x longer, A 4x longer
new <- data.frame(d_len = log(c(1, 2, 4)), d_bullets = 0, d_headers = 0, d_bold = 0, d_code = 0)
cat("P(A wins) for length ratio 1 / 2 / 4:",
    round(predict(m0, newdata = new, type = "response"), 3), "\n")
cat("McFadden pseudo-R2:", round(1 - m0$deviance / m0$null.deviance, 4), "\n")

saveRDS(d, "arena_clean.rds")



