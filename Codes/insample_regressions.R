## =============================================================================
## In-sample predictive regressions 
## Per model: OLS coefs with Newey-West t-stats, fixed-regressor wild bootstrap p-values.
## =============================================================================
rm(list = ls())
t0 <- Sys.time()
pkgs <- c("readxl", "sandwich", "lmtest", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
data_file <- "./Datasets/qrtly_data_2020.xlsx"
y_col   <- "QERET"
nw_lags <- 2
nboot   <- 1000
star_cut <- c(0.01, 0.05, 0.10)

## sample windows
panels <- list(
  list(name = "PANEL A: Full Sample; 1947Q1-2020Q4",        full = c("1947-03", "2020-12"), unem = c("1949-03", "2020-12")),
  list(name = "PANEL B: First Half Sample; 1947Q1-1983Q4",  full = c("1947-03", "1983-12"), unem = c("1949-03", "1984-12")),
  list(name = "PANEL C: Second Half Sample; 1984Q1-2020Q4", full = c("1984-03", "2020-12"), unem = c("1985-03", "2020-12"))
)

m3 <- c("SVAR", "LPE", "INFL")
models <- list("SVAR", "LPE", "INFL",
               c("SVAR", "LPE"), c("SVAR", "INFL"), c("LPE", "INFL"),
               m3, c("SVAR", "LPE", "TBL"),
               c(m3, "TBL"), c(m3, "DFRWIN"), c(m3, "LTR"), c(m3, "IK"), c(m3, "UNEM"),
               c(m3, "TBL", "DFR", "LTR", "IK", "UNEM"))
table_cols <- c("Intercept", "SVAR", "LPE", "INFL", "TBL", "DFR", "LTR", "IK", "UNEM")

## ---- Data -------------------------------------------------------------------
ym <- function(d) format(as.Date(d), "%Y-%m")
d_all <- read_excel(data_file)
d_all$Date_lag <- as.Date(d_all$Date_lag)
d_all <- d_all[order(d_all$Date_lag), ]
lag_cols <- paste0(unique(unlist(models)), "_LAG")
for (v in c(y_col, lag_cols)) d_all[[v]] <- as.numeric(as.character(d_all[[v]]))
window <- function(w) d_all[ym(d_all$Date_lag) >= w[1] & ym(d_all$Date_lag) <= w[2], ]

## ---- Functions --------------------------------------------------------------
fit_model <- function(d, preds) {
  vars <- paste0(preds, "_LAG")
  dd <- d[complete.cases(d[, c(y_col, vars)]), ]
  fit <- lm(as.formula(paste(y_col, "~", paste(vars, collapse = " + "))), data = dd)
  nw  <- coeftest(fit, vcov = NeweyWest(fit, lag = nw_lags))
  list(coef = nw[, 1], t = nw[, 3], adj_r2 = summary(fit)$adj.r.squared, n = nrow(dd), data = dd, vars = vars)
}

## wild bootstrap p-values
wild_boot_p <- function(m) {
  X <- cbind(1, as.matrix(m$data[, m$vars])); b <- m$coef
  flip <- c(1, ifelse(b[-1] < 0, -1, 1))
  X <- sweep(X, 2, flip, `*`); b <- b * flip
  y <- m$data[[y_col]]; e <- y - mean(y); n <- length(y)
  E  <- vapply(seq_len(nboot), function(i) { set.seed(i); rnorm(n) }, numeric(n))
  Bb <- solve(crossprod(X), crossprod(X, mean(y) + e * E))
  setNames(rowMeans(Bb > b), names(m$coef))
}

stars <- function(p) ifelse(p <= star_cut[1], "***", ifelse(p <= star_cut[2], "**", ifelse(p <= star_cut[3], "*", "")))
fmt3  <- function(x) { r <- sign(x) * floor(abs(x) * 1000 + 0.5 + 1e-9) / 1000; formatC(r, format = "f", digits = 3) }

## ---- Run --------------------------------------------------------------------
rows <- list(); k <- 0
for (pa in panels) {
  rows[[length(rows) + 1]] <- c(Model = pa$name, setNames(rep("", length(table_cols)), table_cols), R2 = "")
  for (preds in models) {
    k <- k + 1
    has_unem <- "UNEM" %in% preds
    m <- fit_model(window(if (has_unem) pa$unem else pa$full), preds)
    p <- wild_boot_p(m)
    lab <- c("(Intercept)", paste0(preds, "_LAG")); col <- c("Intercept", sub("DFRWIN", "DFR", preds))
    est <- tst <- setNames(rep("", length(table_cols)), table_cols)
    est[col] <- fmt3(m$coef[lab])
    tst[col] <- paste0("(", fmt3(m$t[lab]), ")", stars(p[lab]))
    rows[[length(rows) + 1]] <- c(Model = paste0(k, if (has_unem) "\u2021" else ""), est, R2 = fmt3(m$adj_r2))
    rows[[length(rows) + 1]] <- c(Model = "", tst, R2 = "")
    cat(sprintf("Model %2d  %-35s n = %3d  adjR2 = %6.3f\n", k, paste(preds, collapse = "+"), m$n, m$adj_r2))
  }
}
tab <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
print(tab, row.names = FALSE)
cat("Done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

## ---- Export -----------------------------------------------------------------
dir.create("output", showWarnings = FALSE)
write_xlsx(list(table = tab), "output/insample_table.xlsx")

tex <- c(paste0("\\begin{tabular}{l", strrep("r", length(table_cols) + 1), "}"), "\\toprule",
         paste(paste(c("", table_cols, "$\\bar R^2$"), collapse = " & "), "\\\\"), "\\midrule")
for (i in seq_len(nrow(tab))) {
  r <- unlist(tab[i, ])
  tex <- c(tex, if (grepl("^PANEL", r[1]))
    sprintf("\\multicolumn{%d}{c}{%s} \\\\ \\midrule", length(r), r[1])
    else paste(paste(gsub("\u2021", "$^\\\\ddagger$", r), collapse = " & "), "\\\\"))
}
writeLines(c(tex, "\\bottomrule", "\\end{tabular}"), "output/insample_table.tex")