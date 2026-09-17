## =============================================================================
## MIDAS In-sample Regressions
## Figure 3 & Table 7
## =============================================================================

rm(list = ls())
t0 <- Sys.time()
pkgs <- c("readxl", "numDeriv", "lubridate", "dplyr", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
qtr_file    <- "./Datasets/qrtly_data_2020.xlsx"
daily_252   <- "./Datasets/daily_stock_return.xlsx"
daily_504   <- "./Datasets/crsp_dailyret_1926.xlsx"
k_grid      <- seq(-0.00001, -0.005, -0.0001)
optim_ctrl  <- list(reltol = 1e-12, maxit = 5000)
star_cut    <- c(0.01, 0.05, 0.10)                     
gd_start <- "1930-09-01"; gd_end <- "1933-12-01"       
ym <- function(d) format(as.Date(d), "%Y-%m")

## ---- MIDAS estimator ----------------------------------------------------
## y: returns; R2: n x D squared daily returns (most recent day first); Z: n x m regressors; 
## gd: dummy vector (adds vt*gd and Z*gd interactions)
fit_midas <- function(y, R2, Z = NULL, gd = NULL) {
  n <- length(y); D <- ncol(R2); jj <- 0:(D - 1)
  midas_w   <- function(k1, k2) { w <- exp(k1 * jj + k2 * jj^2); w / sum(w) }
  midas_var <- function(k1, k2) 66 * drop(R2 %*% midas_w(k1, k2))
  make_X <- function(vt) {
    X <- cbind(1, vt); if (!is.null(Z)) X <- cbind(X, Z)
    if (!is.null(gd)) { X <- cbind(X, vt * gd); if (!is.null(Z)) X <- cbind(X, Z * gd) }
    X
  }
  p <- ncol(make_X(rep(1, n)))
  wls_beta <- function(vt) unname(lm.wfit(make_X(vt), y, w = 1 / vt)$coefficients)
  negll <- function(th) { vt <- midas_var(th[p + 1], th[p + 2]); mu <- drop(make_X(vt) %*% th[1:p])
  0.5 * sum(log(vt)) + 0.5 * sum((y - mu)^2 / vt) }
  ll_i  <- function(th) { vt <- midas_var(th[p + 1], th[p + 2]); mu <- drop(make_X(vt) %*% th[1:p])
  -0.5 * log(vt) - 0.5 * (y - mu)^2 / vt }
  negll_k <- function(k) negll(c(wls_beta(midas_var(k[1], k[2])), k))
  
  # 1. grid search (profile likelihood)
  best <- list(val = Inf, k = NULL)                     
  for (k1 in k_grid) for (k2 in k_grid) {
    op <- optim(c(k1, k2), negll_k)
    if (op$value < best$val) best <- list(val = op$value, k = op$par)
  }
  th0 <- c(wls_beta(midas_var(best$k[1], best$k[2])), best$k)
  
  # 2. joint QML
  er <- optim(th0, negll, control = optim_ctrl)         
  repeat { er2 <- optim(er$par, negll, control = optim_ctrl); if (er$value - er2$value < 1e-10) break; er <- er2 }
  er2 <- optim(er$par, negll, method = "BFGS", control = list(reltol = 1e-14, maxit = 1000))
  if (er2$value < er$value) er <- er2
  stopifnot(er$convergence == 0)
  th <- er$par; k <- th[p + 1:2]
  
  # 3. sandwich SEs
  H  <- numDeriv::hessian(negll, th)[1:p, 1:p] / n      
  G  <- jacobian(ll_i, th); Om <- crossprod(G)[1:p, 1:p] / n
  se <- sqrt(abs(diag(solve(H) %*% Om %*% t(solve(H)) / n)))
  tv <- th[1:p] / se
  vt <- midas_var(k[1], k[2])                           
  adjr2 <- summary(lm(y ~ make_X(vt)[, -1], weights = 1 / vt))$adj.r.squared
  list(coef = th[1:p], se = se, t = tv, p = 2 * pt(-abs(tv), df = n - p), k = k, n = n,
       adjr2 = adjr2, loglik = -er$value, vt = vt, w = midas_w(k[1], k[2]))
}

## quarter-end index 
quarter_ends <- function(day, date_col) {
  day %>% filter(!is.na(.data[[date_col]]), month(.data[[date_col]]) %in% c(3, 6, 9, 12)) %>%
    mutate(yrmon = format(.data[[date_col]], "%Y-%m")) %>%
    group_by(yrmon) %>% slice_max(.data[[date_col]], n = 1, with_ties = FALSE) %>% ungroup() %>% arrange(index)
}
sq_matrix <- function(r, T_idx, D) t(vapply(T_idx, function(ti) r[ti:(ti - D + 1)]^2, numeric(D)))

## =============================================================================
## Panels A-C: D = 252
## =============================================================================
D1 <- 252
q_all <- read_excel(qtr_file)
q_all$Date <- as.Date(q_all$Date); q_all$Date_lag <- as.Date(q_all$Date_lag)
for (v in c("QERET", "LPE_LAG", "INFL_LAG", "SVAR_LAG", "SVAR_GW_LAG")) q_all[[v]] <- as.numeric(as.character(q_all[[v]]))
q_all <- as.data.frame(q_all[order(q_all$Date), ])

day1 <- read_excel(daily_252)
day1$DailyReturnDate <- as.Date(ymd(day1$DailyReturnDate)); day1$index <- seq_len(nrow(day1))
r1 <- as.numeric(day1$Dailyreturn)
qe1 <- quarter_ends(day1, "DailyReturnDate")

samplesABC <- list(list(panel = "A", label = "1947Q1-2020Q4", s = "1947-03", e = "2020-12"),
                   list(panel = "B", label = "1947Q1-1983Q4", s = "1947-03", e = "1983-12"),
                   list(panel = "C", label = "1984Q1-2020Q4", s = "1984-03", e = "2020-12"))
fits <- list(); row_info <- list()
for (sm in samplesABC) {
  q <- q_all[ym(q_all$Date_lag) >= sm$s & ym(q_all$Date_lag) <= sm$e, ]
  T_idx <- qe1$index[match(ym(q$Date_lag), qe1$yrmon)]
  stopifnot(!anyNA(T_idx), T_idx[1] >= D1, !anyNA(q[, c("QERET", "LPE_LAG", "INFL_LAG")]))
  R2 <- sq_matrix(r1, T_idx, D1)
  cat("Panel", sm$panel, sm$label, format(Sys.time(), "%H:%M:%S"), "\n")
  fits[[length(fits) + 1]] <- c(fit_midas(q$QERET, R2), list(names = c("Intercept", "MIDAS"), dates = q$Date_lag))
  fits[[length(fits) + 1]] <- c(fit_midas(q$QERET, R2, Z = as.matrix(q[, c("LPE_LAG", "INFL_LAG")])),
                                list(names = c("Intercept", "MIDAS", "LPE", "INFL")))
  row_info <- c(row_info, list(list(panel = sm$panel, title = paste0("Panel ", sm$panel, ": ", sm$label))),
                list(list(panel = sm$panel)))
}

## =============================================================================
## Panels D-E: D = 504, 1928Q2 - 2021Q1, Great Depression interactions
## =============================================================================
D2 <- 504
day2 <- read_excel(daily_504)
day2$Date <- as.Date(day2$Date); day2$DailyReturnDate_504 <- as.Date(ymd(day2$DailyReturnDate_504))
day2$index <- seq_len(nrow(day2)); r2 <- as.numeric(day2$Dailyreturn_504)
qg <- as.data.frame(subset(day2, Date >= as.Date("1928-06-30") & Date <= as.Date("2021-03-31")))
for (v in c("QERET", "LPE_LAG", "INFL_LAG")) qg[[v]] <- as.numeric(qg[[v]])
gd <- c(0, head(as.numeric(qg$Date >= as.Date(gd_start) & qg$Date < as.Date(gd_end)), -1))
T_idx2 <- quarter_ends(day2, "DailyReturnDate_504")$index[-(1:6)]
stopifnot(length(T_idx2) == nrow(qg), T_idx2[1] >= D2, !anyNA(qg[, c("QERET", "LPE_LAG", "INFL_LAG")]))
R2g <- sq_matrix(r2, T_idx2, D2)
Zi <- as.matrix(qg[, c("LPE_LAG", "INFL_LAG")]); Za <- cbind(qg$LPE_LAG, abs(qg$INFL_LAG))

cat("Panel D", format(Sys.time(), "%H:%M:%S"), "\n")
fits[[7]]  <- c(fit_midas(qg$QERET, R2g, Z = Zi),          list(names = c("Intercept", "MIDAS", "LPE", "INFL")))
fits[[8]]  <- c(fit_midas(qg$QERET, R2g, Z = Zi, gd = gd), list(names = c("Intercept", "MIDAS", "LPE", "INFL", "MIDASGD", "LPEGD", "INFLGD")))
cat("Panel E", format(Sys.time(), "%H:%M:%S"), "\n")
fits[[9]]  <- c(fit_midas(qg$QERET, R2g, Z = Za),          list(names = c("Intercept", "MIDAS", "LPE", "INFL")))
fits[[10]] <- c(fit_midas(qg$QERET, R2g, Z = Za, gd = gd), list(names = c("Intercept", "MIDAS", "LPE", "INFL", "MIDASGD", "LPEGD", "INFLGD")))
row_info <- c(row_info, list(list(panel = "D", title = "Panel D: 1928Q1-2020Q4"), list(panel = "D"),
                             list(panel = "E", title = "Panel E: 1928Q1-2020Q4, Absolute Inflation"), list(panel = "E")))

## =============================================================================
## Table 7
## =============================================================================
stars <- function(p) ifelse(p <= star_cut[1], "***", ifelse(p <= star_cut[2], "**", ifelse(p <= star_cut[3], "*", "")))
cols  <- c("Intercept", "MIDAS", "LPE", "INFL", "MIDASGD", "LPEGD", "INFLGD")
rows <- list()
for (i in seq_along(fits)) {
  f <- fits[[i]]
  if (!is.null(row_info[[i]]$title))
    rows[[length(rows) + 1]] <- c(Model = row_info[[i]]$title, setNames(rep("", length(cols)), cols), R2 = "")
  est <- tst <- setNames(rep("", length(cols)), cols)
  est[f$names] <- sprintf("%.3f", f$coef)
  tst[f$names] <- sprintf("(%.3f)%s", f$t, stars(f$p))
  rows[[length(rows) + 1]] <- c(Model = as.character(i), est, R2 = sprintf("%.3f", f$adjr2))
  rows[[length(rows) + 1]] <- c(Model = "", tst, R2 = "")
}
tab <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
print(tab, row.names = FALSE)

est_table <- do.call(rbind, lapply(seq_along(fits), function(i) data.frame(
  row = i, term = fits[[i]]$names, estimate = fits[[i]]$coef, sandwich_se = fits[[i]]$se,
  t = fits[[i]]$t, p_value = fits[[i]]$p, row.names = NULL)))
k_table <- data.frame(row = seq_along(fits), n = sapply(fits, `[[`, "n"),
                      k1 = sapply(fits, function(f) f$k[1]), k2 = sapply(fits, function(f) f$k[2]),
                      loglik = sapply(fits, `[[`, "loglik"), adjR2 = sapply(fits, `[[`, "adjr2"))
print(k_table, row.names = FALSE)

## =============================================================================
## Figure 3: SVAR, Goyal-Welch SVAR, MIDAS variance 
## =============================================================================
vq <- q_all[ym(q_all$Date_lag) >= "1947-03" & ym(q_all$Date_lag) <= "2020-12", ]  
stopifnot(nrow(vq) == fits[[1]]$n)
pdf("output/fig_variances.pdf", width = 7, height = 4.5)
par(mar = c(3, 4.5, 1, 1))
plot(vq$Date_lag, vq$SVAR_LAG, type = "n", xlab = "", ylab = "Quarterly variance", las = 1, cex.axis = 0.8,
     ylim = c(0, max(c(vq$SVAR_LAG, vq$SVAR_GW_LAG, fits[[1]]$vt), na.rm = TRUE)))
lines(vq$Date_lag, vq$SVAR_GW_LAG, col = "#0072B2", lwd = 2, lty = 2)
lines(vq$Date_lag, fits[[1]]$vt,   col = "#D55E00", lwd = 1)
lines(vq$Date_lag, vq$SVAR_LAG,    col = "black",   lwd = 2)
legend("topleft", bty = "n", cex = 0.85, col = c("black", "#0072B2", "#D55E00"), lwd = c(2, 1.5, 1), lty = c(1, 2, 1),
       legend = c("SVAR", "Goyal and Welch SVAR", "MIDAS variance"))
dev.off()

## =============================================================================
## Export
## =============================================================================
write_xlsx(list(table = tab, estimates = est_table, k_and_fit = k_table,
                midas_var_row1 = data.frame(Date_lag = vq$Date_lag, SVAR = vq$SVAR_LAG, SVAR_GW = vq$SVAR_GW_LAG, MIDAS = fits[[1]]$vt),
                weights_row1 = data.frame(lag_days = seq_along(fits[[1]]$w) - 1, weight = fits[[1]]$w)),
           "output/midas_table7.xlsx")

tex <- c("\\begin{tabular}{lccccccc r}", "\\toprule")
hdr1 <- " & Intercept & MIDAS & LPE & INFL & & & & $\\bar R^2$ \\\\ \\midrule"
hdrD <- " & Intercept & MIDAS & LPE & INFL & MIDASGD & LPEGD & INFLGD & \\\\ \\midrule"
hdrE <- " & Intercept & MIDAS & LPE & ABSINFL & MIDASGD & LPEGD & ABSINFLGD & \\\\ \\midrule"
for (i in seq_len(nrow(tab))) {
  r <- unlist(tab[i, ])
  if (grepl("^Panel", r[1])) {
    tex <- c(tex, sprintf("\\multicolumn{9}{c}{%s} \\\\ \\midrule", r[1]),
             if (grepl("^Panel A", r[1])) hdr1 else if (grepl("^Panel D", r[1])) hdrD else if (grepl("^Panel E", r[1])) hdrE else NULL)
  } else tex <- c(tex, paste(paste(r, collapse = " & "), "\\\\"))
}
writeLines(c(tex, "\\bottomrule", "\\end{tabular}"), "output/midas_table7.tex")
writeLines(capture.output(sessionInfo()), "output/sessionInfo_midas_table7.txt")
cat("Done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
