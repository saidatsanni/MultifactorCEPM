## =============================================================================
## Out-of-sample forecasts, 1953Q3 - 2020Q4, recursive expanding window
##   Model 1: MIDAS + LPE + INFL   + GD interactions   
##   Model 2: MIDAS + LPE + |INFL| + GD interactions   
##   Model 3: SVAR  + LPE + INFL   + GD interactions  
## =============================================================================

rm(list = ls())
t0 <- Sys.time()

pkgs <- c("readxl", "lubridate", "dplyr", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
daily_file <- "./Datasets/crsp_dailyret_1926.xlsx"
q_start  <- "1928-06-30"; q_end <- "2021-03-31"
is_end   <- "1953-06-30"                               
oos_end  <- "2020-12-31"                               
gd_start <- "1930-09-01"; gd_end <- "1933-12-01" #dummy = 1 for 1930Q3-1933Q3
D        <- 504
k_grid   <- seq(-0.00001,-0.005, -0.0001)             
optim_ctrl <- list(reltol = 1e-10, maxit = 5000)
svar_weighted <- FALSE 

## ---- Data -------------------------------------------------------------------
day_data <- read_excel(daily_file)
day_data$Date <- as.Date(day_data$Date)
day_data$DailyReturnDate_504 <- as.Date(ymd(day_data$DailyReturnDate_504))
day_data$index <- seq_len(nrow(day_data))
r <- as.numeric(day_data$Dailyreturn_504)

q <- as.data.frame(subset(day_data, Date >= as.Date(q_start) & Date <= as.Date(q_end)))
for (v in c("QERET", "LPE_LAG", "INFL_LAG", "SVAR_LAG")) q[[v]] <- as.numeric(q[[v]])
gd <- as.numeric(q$Date >= as.Date(gd_start) & q$Date < as.Date(gd_end))
q$DUMMY_LAG   <- c(0, head(gd, -1))
q$ABSINFL_LAG <- abs(q$INFL_LAG)
n <- nrow(q)

qe <- day_data %>%
  filter(!is.na(DailyReturnDate_504), month(DailyReturnDate_504) %in% c(3, 6, 9, 12)) %>%
  mutate(yrmon = format(DailyReturnDate_504, "%Y-%m")) %>%
  group_by(yrmon) %>% slice_max(DailyReturnDate_504, n = 1, with_ties = FALSE) %>%
  ungroup() %>% arrange(index)
T_idx <- qe$index[-(1:6)]
stopifnot(length(T_idx) == n, T_idx[1] >= D)

R2 <- t(vapply(T_idx, function(ti) r[ti:(ti - D + 1)]^2, numeric(D)))   
jj <- 0:(D - 1)

## ---- Model blocks ----------------------------
midas_var <- function(k1, k2, rows) {
  w <- exp(k1 * jj + k2 * jj^2); w <- w / sum(w)
  66 * drop(R2[rows, , drop = FALSE] %*% w)
}
make_X <- function(vt, spec, rows) {
  Z <- as.matrix(q[rows, spec$vars, drop = FALSE]); d <- q$DUMMY_LAG[rows]
  cbind(1, vt, Z, vt * d, Z * d)
}
wls_beta <- function(vt, spec, rows)
  unname(lm.wfit(make_X(vt, spec, rows), q$QERET[rows], w = 1 / vt)$coefficients)
negll <- function(theta, spec, rows) {
  p  <- length(theta) - 2
  vt <- midas_var(theta[p + 1], theta[p + 2], rows)
  mu <- drop(make_X(vt, spec, rows) %*% theta[1:p])
  0.5 * sum(log(vt)) + 0.5 * sum((q$QERET[rows] - mu)^2 / vt)
}
negll_k <- function(k, spec, rows) {
  vt <- midas_var(k[1], k[2], rows)
  negll(c(wls_beta(vt, spec, rows), k), spec, rows)
}
grid_start <- function(spec, rows) {
  best <- list(val = Inf, k = NULL)
  for (k1 in k_grid) for (k2 in k_grid) {
    op <- optim(c(k1, k2), negll_k, spec = spec, rows = rows)
    if (op$value < best$val) best <- list(val = op$value, k = op$par)
  }
  best$k
}

## ---- Sample layout ----------------------------------------------------------
n_is     <- sum(q$Date <= as.Date(is_end))           
oos_rows <- (n_is + 1):sum(q$Date <= as.Date(oos_end)) 
T1       <- length(oos_rows)
oos_dates <- q$Date[oos_rows]
actual    <- q$QERET[oos_rows]
cat("Forecasting", T1, "quarters:", format(oos_dates[1]), "to", format(oos_dates[T1]), "\n")

## ---- Recursive OOS: MIDAS models (joint QML re-estimated every quarter) -----
oos_midas <- function(spec) {
  cat("Model", spec$name, "- grid search on initial window ...", format(Sys.time(), "%H:%M:%S"), "\n")
  k_prev <- grid_start(spec, 1:n_is)
  p <- length(spec$labels)                              
  theta <- matrix(NA_real_, T1, p + 2, dimnames = list(NULL, c(spec$labels, "K1", "K2")))
  fc <- vt_fc <- rep(NA_real_, T1)
  for (s in seq_len(T1)) {
    tr <- oos_rows[s]; est <- 1:(tr - 1)
    theta0 <- c(wls_beta(midas_var(k_prev[1], k_prev[2], est), spec, est), k_prev)
    er <- optim(theta0, negll, spec = spec, rows = est, control = optim_ctrl)
    theta[s, ] <- er$par; k_prev <- er$par[p + 1:2]
    vt_fc[s] <- midas_var(k_prev[1], k_prev[2], tr)
    fc[s]    <- drop(make_X(vt_fc[s], spec, tr) %*% er$par[1:p])
    if (s %% 30 == 0) cat("  ", s, "/", T1, format(Sys.time(), "%H:%M:%S"), "\n")
  }
  list(fc = fc, theta = theta, vt = vt_fc)
}

## ---- Recursive OOS: SVAR model -------------------
oos_svar <- function(spec) {
  cat("Model", spec$name, "...\n")
  theta <- matrix(NA_real_, T1, length(spec$labels), dimnames = list(NULL, spec$labels))
  fc <- rep(NA_real_, T1)
  for (s in seq_len(T1)) {
    tr <- oos_rows[s]; est <- 1:(tr - 1)
    X  <- make_X(q$SVAR_LAG[est], spec, est)
    w  <- if (svar_weighted) 1 / q$SVAR_LAG[est] else rep(1, length(est))
    b  <- lm.wfit(X, q$QERET[est], w = w)$coefficients
    theta[s, ] <- b
    fc[s] <- drop(make_X(q$SVAR_LAG[tr], spec, tr) %*% b)
  }
  list(fc = fc, theta = theta)
}

## ---- Specs ----------------------------------------------------------
spec1 <- list(name = "MIDAS",        vars = c("LPE_LAG", "INFL_LAG"),
              labels = c("Constant", "MIDAS", "LPE", "INFL", "MIDAS_GD", "LPE_GD", "INFL_GD"))
spec2 <- list(name = "MIDAS_absINFL", vars = c("LPE_LAG", "ABSINFL_LAG"),
              labels = c("Constant", "MIDAS", "LPE", "ABSINFL", "MIDAS_GD", "LPE_GD", "ABSINFL_GD"))
spec3 <- list(name = "SVAR",         vars = c("LPE_LAG", "INFL_LAG"),
              labels = c("Constant", "SVAR", "LPE", "INFL", "SVAR_GD", "LPE_GD", "INFL_GD"))

m1 <- oos_midas(spec1)
m2 <- oos_midas(spec2)
m3 <- oos_svar(spec3)
fc_hm <- sapply(oos_rows, function(tr) mean(q$QERET[1:(tr - 1)]))  

## ---- OOS R^2 ------------------------------------------------------------------
fcsq <- data.frame(HM = (actual - fc_hm)^2, MIDAS = (actual - m1$fc)^2,
                   MIDAS_absINFL = (actual - m2$fc)^2, SVAR = (actual - m3$fc)^2)
cum_r2 <- function(num, den) sapply(seq_len(T1), function(i) 1 - sum(num[i:T1]) / sum(den[i:T1]))
oos_r2 <- data.frame(Date = oos_dates,
                     OOS_MIDAS         = cum_r2(fcsq$MIDAS,         fcsq$HM),
                     OOS_MIDAS_ABSINFL = cum_r2(fcsq$MIDAS_absINFL, fcsq$HM),
                     OOS_SVAR          = cum_r2(fcsq$SVAR,          fcsq$HM))
cat("Full-period OOS R^2, 1953Q3-2020Q4:\n"); print(round(oos_r2[1, -1], 4))

## ---- Figure 4-------------------------------------------------------------------
cut_years <- 10
cutoff    <- seq(max(oos_r2$Date), by = paste0("-", cut_years, " years"), length.out = 2)[2]
pr        <- oos_r2[oos_r2$Date <= cutoff, ]              

pdf("output/fig_oos_gd_models.pdf", width = 7, height = 4.5)
par(mar = c(3, 4.5, 1, 1))
plot(pr$Date, pr$OOS_MIDAS, type = "n", xlab = "", las = 1, cex.axis = 0.8, xaxt = "n",
     ylab = expression("Out-of-sample " * R^2), ylim = c(-0.2, 0.2))
axis.Date(1, at = seq(as.Date("1960-12-01"), max(pr$Date), by = "10 years"),
          format = "%Y", cex.axis = 0.8)
lines(pr$Date, pr$OOS_SVAR,          lwd = 2.5, lty = 1)   
lines(pr$Date, pr$OOS_MIDAS,         lwd = 1,   lty = 1)   
lines(pr$Date, pr$OOS_MIDAS_ABSINFL, lwd = 1,   lty = 2)   
legend("topleft", bty = "n", cex = 0.85, lwd = c(1, 1, 2.5), lty = c(2, 1, 1),
       legend = c("MIDAS variance, ABSINFL", "MIDAS variance", "SVAR"))
dev.off()

## ---- Save Output-------------------------------------------------------------------
write_xlsx(list(oos_r2    = oos_r2,
                forecasts = data.frame(Date = oos_dates, actual, fc_hm, fc_midas = m1$fc,
                                       fc_midas_absinfl = m2$fc, fc_svar = m3$fc, fcsq),
                params_midas   = data.frame(Date = oos_dates, m1$theta),
                params_absinfl = data.frame(Date = oos_dates, m2$theta),
                params_svar    = data.frame(Date = oos_dates, m3$theta)),
           "output/oos_gd_models.xlsx")
cat("Done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
