## =============================================================================
## MIDAS conditional equity premium: Great Depression dummy and absolute inflation
## Sample 1928Q2 - 2021Q1, D = 504 daily returns.
##   Row 1: MIDAS + LPE + INFL
##   Row 2: MIDAS + LPE + INFL     + GD interactions
##   Row 3: MIDAS + LPE + |INFL|
##   Row 4: MIDAS + LPE + |INFL|   + GD interactions
## =============================================================================
rm(list = ls())
t0 <- Sys.time()

pkgs <- c("readxl", "numDeriv", "lubridate", "dplyr", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
daily_file <- "./Datasets/crsp_dailyret_1926.xlsx"
q_start <- "1928-06-30"; q_end <- "2021-03-31"
gd_start <- "1930-09-01"; gd_end <- "1933-12-01"      
D       <- 504
k_grid  <- seq(-0.00001, -0.005, -0.0001)              
optim_ctrl <- list(reltol = 1e-12, maxit = 5000)       
star_cut <- c(0.01, 0.05, 0.10)                        

## ---- Data -------------------------------------------------------------------
day_data <- read_excel(daily_file)
day_data$Date <- as.Date(day_data$Date)
day_data$DailyReturnDate_504 <- as.Date(ymd(day_data$DailyReturnDate_504))
day_data$index <- seq_len(nrow(day_data))
r <- as.numeric(day_data$Dailyreturn_504)

q <- as.data.frame(subset(day_data, Date >= as.Date(q_start) & Date <= as.Date(q_end)))
for (v in c("QERET", "LPE_LAG", "INFL_LAG")) q[[v]] <- as.numeric(q[[v]])
gd <- as.numeric(q$Date >= as.Date(gd_start) & q$Date < as.Date(gd_end))
q$DUMMY_LAG   <- c(0, head(gd, -1))
q$ABSINFL_LAG <- abs(q$INFL_LAG)
n <- nrow(q)

## last trading day of each quarter in the daily file
qe <- day_data %>%
  filter(!is.na(DailyReturnDate_504), month(DailyReturnDate_504) %in% c(3, 6, 9, 12)) %>%
  mutate(yrmon = format(DailyReturnDate_504, "%Y-%m")) %>%
  group_by(yrmon) %>% slice_max(DailyReturnDate_504, n = 1, with_ties = FALSE) %>%
  ungroup() %>% arrange(index)
T_idx <- qe$index[-(1:6)]
stopifnot(length(T_idx) == n, T_idx[1] >= D)

## ---- Model blocks -----------------------------------------------------------
## MIDAS variance for all quarters, weights on days 0..D-1 back from each quarter-end.
R2 <- t(vapply(T_idx, function(ti) r[ti:(ti - D + 1)]^2, numeric(D)))   # n x D
jj <- 0:(D - 1)
midas_var <- function(k1, k2) {
  w <- exp(k1 * jj + k2 * jj^2); w <- w / sum(w)
  66 * drop(R2 %*% w)
}

make_X <- function(vt, spec) {
  X <- cbind(1, vt, as.matrix(q[, spec$vars]))
  if (spec$gd) X <- cbind(X, vt * q$DUMMY_LAG, as.matrix(q[, spec$vars]) * q$DUMMY_LAG)
  X
}
wls_beta <- function(vt, spec) unname(lm.wfit(make_X(vt, spec), q$QERET, w = 1 / vt)$coefficients)

negll <- function(theta, spec) {
  p  <- length(theta) - 2
  vt <- midas_var(theta[p + 1], theta[p + 2])
  mu <- drop(make_X(vt, spec) %*% theta[1:p])
  0.5 * sum(log(vt)) + 0.5 * sum((q$QERET - mu)^2 / vt)
}
ll_i <- function(theta, spec) {
  p  <- length(theta) - 2
  vt <- midas_var(theta[p + 1], theta[p + 2])
  mu <- drop(make_X(vt, spec) %*% theta[1:p])
  -0.5 * log(vt) - 0.5 * (q$QERET - mu)^2 / vt
}
## profile likelihood in k
negll_k <- function(k, spec) {
  vt <- midas_var(k[1], k[2])
  negll(c(wls_beta(vt, spec), k), spec)
}

fit_spec <- function(spec, k_start = NULL) {
  ## 1. grid search: profile likelihood
  if (is.null(k_start)) {
    best <- list(val = Inf, k = NULL)
    for (k1 in k_grid) for (k2 in k_grid) {
      op <- optim(c(k1, k2), negll_k, spec = spec)
      if (op$value < best$val) best <- list(val = op$value, k = op$par)
    }
    k_start <- best$k
  }
  ## 2. joint QML of betas and k
  theta0 <- c(wls_beta(midas_var(k_start[1], k_start[2]), spec), k_start)
  er <- optim(theta0, negll, spec = spec, hessian = TRUE, control = optim_ctrl)
  p  <- length(theta0) - 2
  ## 3. sandwich SEs 
  G  <- jacobian(ll_i, er$par, spec = spec)
  H  <- er$hessian[1:p, 1:p] / n
  Om <- crossprod(G)[1:p, 1:p] / n
  V  <- solve(H) %*% Om %*% t(solve(H)) / n
  se <- sqrt(abs(diag(V)))
  ## 4. adjusted R^2 
  vt_opt <- midas_var(er$par[p + 1], er$par[p + 2])
  adjr2  <- summary(lm(q$QERET ~ make_X(vt_opt, spec)[, -1], weights = 1 / vt_opt))$adj.r.squared
  b  <- er$par[1:p]; tv <- b / se
  list(coef = b, se = se, t = tv, p = 2 * pt(-abs(tv), df = n - p),
       k = er$par[p + 1:2], adjr2 = adjr2, loglik = -er$value,
       convergence = er$convergence, names = spec$labels)
}

## ---- The four specifications ------------------------------------------------
specs <- list(
  list(vars = c("LPE_LAG", "INFL_LAG"),    gd = FALSE,
       labels = c("Constant", "MIDAS", "LPE", "INFL")),
  list(vars = c("LPE_LAG", "INFL_LAG"),    gd = TRUE,
       labels = c("Constant", "MIDAS", "LPE", "INFL", "MIDAS_GD", "LPE_GD", "INFL_GD")),
  list(vars = c("LPE_LAG", "ABSINFL_LAG"), gd = FALSE,
       labels = c("Constant", "MIDAS", "LPE", "ABSINFL")),
  list(vars = c("LPE_LAG", "ABSINFL_LAG"), gd = TRUE,
       labels = c("Constant", "MIDAS", "LPE", "ABSINFL", "MIDAS_GD", "LPE_GD", "ABSINFL_GD"))
)

fits <- vector("list", length(specs))
for (i in seq_along(specs)) {
  cat("Row", i, "...", format(Sys.time(), "%H:%M:%S"), "\n")
  fits[[i]] <- fit_spec(specs[[i]])
  cat("   k =", signif(fits[[i]]$k, 4), " adj R2 =", round(fits[[i]]$adjr2, 3),
      " conv =", fits[[i]]$convergence, "\n")
}

## Output
stars <- function(p) ifelse(p < star_cut[1], "***", ifelse(p < star_cut[2], "**",
                                                           ifelse(p < star_cut[3], "*", "")))
all_cols <- c("Constant", "MIDAS", "LPE", "INFL", "ABSINFL",
              "MIDAS_GD", "LPE_GD", "INFL_GD", "ABSINFL_GD")

table_rows <- do.call(rbind, lapply(seq_along(fits), function(i) {
  f <- fits[[i]]
  est <- setNames(rep("", length(all_cols)), all_cols)
  tst <- est
  est[f$names] <- sprintf("%.3f", f$coef)
  tst[f$names] <- sprintf("(%.3f)%s", f$t, stars(f$p))
  rbind(data.frame(row = i, stat = "coef", t(est), adjR2 = sprintf("%.3f", f$adjr2)),
        data.frame(row = i, stat = "t",    t(tst), adjR2 = ""))
}))
print(table_rows, row.names = FALSE)

k_table <- data.frame(row = seq_along(fits),
                      k1 = sapply(fits, function(f) f$k[1]),
                      k2 = sapply(fits, function(f) f$k[2]),
                      loglik = sapply(fits, function(f) f$loglik),
                      adjR2  = sapply(fits, function(f) f$adjr2))

write_xlsx(list(table = table_rows, k_and_fit = k_table), "midas_gd_table.xlsx")
