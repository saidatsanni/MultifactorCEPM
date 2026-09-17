## =============================================================================
## Out-of-sample forecast evaluation: ENC-NEW, MSE-F, and HLN
## =============================================================================
rm(list = ls())
t0 <- Sys.time()
pkgs <- c("readxl", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
data_file <- "./Datasets/qrtly_data_2020.xlsx"
oos_start <- "1964-12"                    
end_ym    <- "2020-12"                     
panels    <- c(A = "1965-03", B = "1976-03", C = "2000-03", D = "2010-03")   
B         <- 1000; boot_seed <- 1
star_cut  <- c(0.01, 0.05, 0.10)
mul_vars  <- c("SVAR_LAG", "LPE_LAG", "INFL_LAG")
comb_vars <- c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "NTIS_LAG", "LDP_LAG", "LDY_LAG", "LDE_LAG",
               "DFY_LAG", "DFR_LAG", "TMS_LAG", "LTY_LAG", "BM_LAG", "LTR_LAG", "TBL_LAG", "IK_LAG")

## ---- Data -------------------------------------------------------------------
ym <- function(x) format(as.Date(x), "%Y-%m")
d <- read_excel(data_file)
d$Date <- as.Date(d$Date); d$Date_lag <- as.Date(d$Date_lag)
d <- as.data.frame(d[ym(d$Date_lag) >= "1947-03" & ym(d$Date_lag) <= end_ym, ])
d <- d[order(d$Date), ]
num_cols <- unique(c("QERET", "QERET_LAG", "LPE", "SVAR", "INFL", comb_vars))
for (v in num_cols) d[[v]] <- as.numeric(as.character(d[[v]]))
y <- d$QERET

fc_rows <- which(ym(d$Date) > oos_start & ym(d$Date) <= end_ym)      
T1 <- length(fc_rows); fc_dates <- d$Date[fc_rows]
cat("Forecasting", T1, "quarters:", format(fc_dates[1]), "to", format(fc_dates[T1]), "\n")

## ---- Forecasts ----------------------------------------------------
ols_fc <- function(vars, est, tr) {
  X <- cbind(1, as.matrix(d[est, vars, drop = FALSE]))
  sum(c(1, as.numeric(d[tr, vars])) * lm.fit(X, y[est])$coefficients)
}
actual  <- y[fc_rows]
fc_hm   <- sapply(fc_rows, function(tr) mean(y[1:(tr - 1)]))
fc_mul  <- sapply(fc_rows, function(tr) ols_fc(mul_vars, 1:(tr - 1), tr))
fc_comb <- sapply(fc_rows, function(tr) mean(vapply(comb_vars, function(v) ols_fc(v, 1:(tr - 1), tr), numeric(1))))

## ---- Test statistics ----------------------------------------------------------------
r2_oos <- function(a, f, bench) 1 - sum((a - f)^2) / sum((a - bench)^2)
enc_new <- function(a, f1, f2) { u1 <- a - f1; u2 <- a - f2; length(u1) * mean(u1^2 - u1 * u2) / mean(u2^2) }
mse_f   <- function(a, f1, f2) { u1 <- a - f1; u2 <- a - f2; sum(u1^2 - u2^2) / mean(u2^2) }
hln <- function(a, f1, f2) {                       # H0: f1 encompasses f2
  u1 <- a - f1; u2 <- a - f2; dd <- (u1 - u2) * u1
  stat <- mean(dd) / sqrt(mean((dd - mean(dd))^2) / length(dd))
  c(stat = stat, p = 1 - pnorm(stat))
}

## ---- Bootstrap-----------------------------------------------------------------------
boot_stats <- function(is_end_lag) {
  bd <- d[ym(d$Date_lag) <= "2020-09", ] 
  nobs <- nrow(bd)
  eqs <- list(lm(LPE  ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bd),
              lm(SVAR ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bd),
              lm(INFL ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bd))
  rmod  <- lm(QERET ~ 1, data = bd)
  coefs <- rbind(t(sapply(eqs, coef)), c(coef(rmod), 0, 0, 0, 0))
  errs  <- cbind(sapply(eqs, resid), resid(rmod))
  init  <- c(mean(bd$LPE), mean(bd$SVAR), mean(bd$INFL), mean(bd$QERET))
  n_is  <- which(ym(bd$Date_lag) == is_end_lag)    
  n_oos <- nobs - n_is
  set.seed(boot_seed)
  out <- matrix(NA_real_, B, 2 * n_oos)           
  for (b in seq_len(B)) {
    e <- errs[sample(nobs, B, replace = TRUE), ]
    x <- matrix(NA_real_, B, 4)
    x[1, ] <- coefs[, 1] + coefs[, 2:5] %*% init + e[1, ]
    for (j in 2:B) x[j, ] <- coefs[, 1] + coefs[, 2:5] %*% x[j - 1, ] + e[j, ]
    s   <- x[(B - nobs + 1):B, ]; s_lag <- x[(B - nobs):(B - 1), ]
    ys  <- s[, 4]; Xs <- cbind(1, s_lag[, 1:3])
    oos <- (n_is + 1):nobs
    f_const <- mean(ys[1:n_is])                     
    f_mul   <- sapply(oos, function(i) sum(Xs[i, ] * lm.fit(Xs[1:(i - 1), ], ys[1:(i - 1)])$coefficients))
    out[b, ] <- c(ys[oos] - f_const, ys[oos] - f_mul)
  }
  list(u1 = out[, 1:n_oos, drop = FALSE], u2 = out[, n_oos + 1:n_oos, drop = FALSE])
}

## ---- Output ---------------------------------------------------------------
stars <- function(p) ifelse(p <= star_cut[1], "***", ifelse(p <= star_cut[2], "**", ifelse(p <= star_cut[3], "*", "")))
f3 <- function(x) formatC(x, format = "f", digits = 3)
rows <- list()
for (pn in names(panels)) {
  w <- which(ym(fc_dates) >= panels[[pn]]); a <- actual[w]
  enc  <- enc_new(a, fc_hm[w], fc_mul[w]); msf <- mse_f(a, fc_hm[w], fc_mul[w])
  is_end <- ym(seq(as.Date(paste0(panels[[pn]], "-01")), by = "-6 months", length.out = 2)[2])
  cat("Bootstrap for panel", pn, "(in-sample Date_lag through", is_end, ") ...", format(Sys.time(), "%H:%M:%S"), "\n")
  bs <- boot_stats(is_end)
  enc_b <- sapply(seq_len(B), function(b) enc_new(0, -bs$u1[b, ], -bs$u2[b, ]))
  msf_b <- sapply(seq_len(B), function(b) mse_f(0, -bs$u1[b, ], -bs$u2[b, ]))
  p_enc <- mean(enc_b > enc); p_msf <- mean(msf_b > msf)
  h01 <- hln(a, fc_mul[w], fc_comb[w]); h02 <- hln(a, fc_comb[w], fc_mul[w])
  lab <- sprintf("Panel %s: %sQ%d-2020Q4", pn, substr(panels[[pn]], 1, 4), ceiling(as.integer(substr(panels[[pn]], 6, 7)) / 3))
  rows[[length(rows) + 1]] <- c(Model = lab, R2_MUL = "", R2_COMB = "", ENC_NEW = "", MSE_F = "", HLN_H01 = "", HLN_H02 = "")
  rows[[length(rows) + 1]] <- c(Model = "Historical average vs. Multi-factor",
                                R2_MUL = f3(r2_oos(a, fc_mul[w], fc_hm[w])), R2_COMB = "",
                                ENC_NEW = paste0(f3(enc), stars(p_enc)), MSE_F = paste0(f3(msf), stars(p_msf)),
                                HLN_H01 = "", HLN_H02 = "")
  rows[[length(rows) + 1]] <- c(Model = "Combination vs. Multi-factor",
                                R2_MUL = "", R2_COMB = f3(r2_oos(a, fc_comb[w], fc_hm[w])), ENC_NEW = "", MSE_F = "",
                                HLN_H01 = paste0(f3(h01["stat"]), stars(h01["p"])),
                                HLN_H02 = paste0(f3(h02["stat"]), stars(h02["p"])))
  cat(sprintf("%s  R2 mul %.3f comb %.3f | ENC %.3f (p=%.3f) MSEF %.3f (p=%.3f) | HLN H01 %.3f H02 %.3f\n",
              lab, r2_oos(a, fc_mul[w], fc_hm[w]), r2_oos(a, fc_comb[w], fc_hm[w]), enc, p_enc, msf, p_msf, h01["stat"], h02["stat"]))
}
tab <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
print(tab, row.names = FALSE)
cat("Done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

## ---- Export ---------------------------------------------------------------------------
dir.create("Output", showWarnings = FALSE)
write_xlsx(list(table = tab, forecasts = data.frame(Date = fc_dates, actual, fc_hm, fc_mul, fc_comb)), "Output/oos_tests.xlsx")
tex <- c("\\begin{tabular}{lcccccc}", "\\toprule",
         "Model 1 vs. Model 2 & $\\bar R^2_{OOS}$ (Multi-factor) & $\\bar R^2_{OOS}$ (Combination) & ENC-NEW & MSE-F & HLN ($H_{01}$) & HLN ($H_{02}$) \\\\ \\midrule")
for (i in seq_len(nrow(tab))) {
  r <- unlist(tab[i, ])
  tex <- c(tex, if (grepl("^Panel", r[1])) sprintf("\\multicolumn{7}{c}{%s} \\\\ \\midrule", r[1])
                else paste(paste(r, collapse = " & "), "\\\\"))
}
writeLines(c(tex, "\\bottomrule", "\\end{tabular}"), "output/oos_tests.tex")
