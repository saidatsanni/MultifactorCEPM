#---------------------------------------------
# VARIABLE SELECTION
#---------------------------------------------

rm(list = ls())

# Load required packages
pkgs <- c("glmnet", "readxl", "devtools", "BayesVarSel", "dplyr", "leaps", 
          "plotmo", "stargazer", "EnvStats", "moments", "stringi", "xtable")
install.packages(setdiff(pkgs, rownames(installed.packages())))
if (!require("FarmSelect")) devtools::install_github("kbose28/FarmSelect")
invisible(lapply(c(pkgs, "FarmSelect"), library, character.only = TRUE))

# Load dataset
data_qtr <- read_excel("./Datasets/qrtly_data_2020.xlsx")
data_qtr$Date_lag <- as.Date(data_qtr$Date_lag, format= "%Y-%m-%d")

data_q <- data_qtr[, c("Date_lag", "QERET", "SVAR_LAG","LPE_LAG","INFL_LAG",
                       "NTIS_LAG","LDP_LAG","LDY_LAG","DFY_LAG","DFR_LAG",
                       "TMS_LAG","RREL_LAG","BM_LAG","LTR_LAG","TBL_LAG",
                       "IK_LAG","UNEM_LAG","CAY_LAG","CC_LAG","NOS_LAG")]

# Define sample periods
full_q <- subset(data_q, Date_lag >= "1947-03-01" & Date_lag <= "2020-12-01",
                 select = -c(Date_lag, UNEM_LAG, CAY_LAG, CC_LAG, NOS_LAG)) 
unem_q <- subset(data_q, Date_lag >= "1949-03-01" & Date_lag <= "2020-12-01",
                 select = -c(Date_lag, CAY_LAG, CC_LAG, NOS_LAG))
cay_q  <- subset(data_q, Date_lag >= "1952-03-01" & Date_lag <= "2020-12-01",
                 select = -c(Date_lag, CC_LAG, NOS_LAG)) 
cc_q   <- subset(data_q, Date_lag >= "1954-03-01" & Date_lag <= "2020-12-01",
                 select = -c(Date_lag, NOS_LAG)) 
nos_q  <- subset(data_q, Date_lag >= "1958-03-01" & Date_lag <= "2020-12-01",
                 select = -c(Date_lag))

samples <- list("Full sample" = full_q, "UNEM sample" = unem_q, 
                "CAY sample"  = cay_q, "CC sample" = cc_q, "NOS sample" = nos_q)


#---------------------------------------------
# Best subset selection (BIC / AIC)
select_vars <- function(dat, criterion = "BIC") {
  dat <- data.frame(lapply(dat, as.numeric))
  sm <- summary(regsubsets(QERET ~ ., data = dat, nvmax = ncol(dat) - 1))
  p <- apply(sm$which, 1, sum)
  score <- if (criterion == "BIC") sm$bic else sm$bic - log(nrow(dat)) * p + 2 * p
  best <- which.min(score)
  names(sm$which[best, ])[sm$which[best, ]][-1]
}

sel_BIC <- lapply(samples, select_vars, criterion = "BIC")
sel_AIC <- lapply(samples, select_vars, criterion = "AIC")

vars_order <- c("SVAR_LAG","LPE_LAG","INFL_LAG","DFR_LAG","LTR_LAG","UNEM_LAG","IK_LAG")
vars_report <- vars_order[vars_order %in% sort(unique(c(unlist(sel_BIC), unlist(sel_AIC))))]

build_panel <- function(sel_list) {
  mat <- matrix("", nrow = length(samples), ncol = length(vars_report),
                dimnames = list(names(samples), vars_report))
  for (i in seq_along(sel_list)) {
    mat[i, vars_report %in% sel_list[[i]]] <- "X"
  }
  as.data.frame(mat)
}

print(xtable(build_panel(sel_BIC)), sanitize.text.function = identity)
print(xtable(build_panel(sel_AIC)), sanitize.text.function = identity)


#---------------------------------------------
# Expanding-window variable selection frequencies
expanding_freq <- function(dat, start = 100) {
  dat_rev <- dat[nrow(dat):1, ]
  steps <- seq(start + 1, nrow(dat))
  coe <- lapply(steps, function(i) {
    tmp <- dat_rev[1:i, ]
    fit <- regsubsets(QERET ~ ., data = tmp, nbest = 2, nvmax = ncol(dat) - 1)
    setdiff(names(coef(fit, which.min(summary(fit)$bic))), "(Intercept)")
  })
  mat <- data.frame(stri_list2matrix(coe), byrow = TRUE)
  occ <- data.frame(table(unlist(mat)))
  colnames(occ) <- c("variable", "freq")
  occ$proportion <- occ$freq / length(steps)
  occ[order(occ$proportion, decreasing = TRUE), ]
}

freq_list <- lapply(samples, expanding_freq)
vars_order2 <- c("SVAR_LAG","LPE_LAG","LDY_LAG","BM_LAG","INFL_LAG","DFR_LAG")

table2 <- matrix("", length(samples), length(vars_order2), dimnames = list(names(samples), vars_order2))
for (i in seq_along(freq_list)) {
  tmp <- freq_list[[i]]
  vars <- intersect(tmp$variable, vars_order2)
  table2[i, vars] <- paste0(round(100 * tmp$proportion[tmp$variable %in% vars]), "%")
}
table2 <- as.data.frame(table2)
table2 <- table2[, colSums(table2 != "") > 0]
print(table2)


#---------------------------------------------
# Bootstrap BIC
bootstrap_datalist <- function(fit, data, nboot = 1000, seed = 10) {
  set.seed(seed)
  n <- length(resid(fit))
  res_boot <- replicate(nboot, sample(resid(fit), n, replace = TRUE))
  y_boot <- matrix(rep(fit$fitted.values, nboot), ncol = nboot)
  lapply(1:nboot, function(i) cbind(QERET_SIM = y_boot[, i] + res_boot[, i],
                                    data[, setdiff(names(data), "QERET")]))
}

bootstrap_bic <- function(datalist) {
  selected_vars <- lapply(datalist, function(dat) {
    z <- regsubsets(QERET_SIM ~ ., data = dat, nbest = 2, nvmax = ncol(dat) - 1)
    names(coef(z, which.min(summary(z)$bic)))[-1]
  })
  mat <- data.frame(stri_list2matrix(selected_vars), byrow = TRUE)
  occ <- data.frame(table(unlist(mat[, -ncol(mat)])))
  colnames(occ) <- c("Selected variables", "Frequency")
  occ$Proportion <- occ$Frequency / length(datalist)
  occ[order(occ$Proportion, decreasing = TRUE), ]
}

mod1 <- lm(QERET ~ LPE_LAG + INFL_LAG + SVAR_LAG, data = cay_q)
mod2 <- lm(QERET ~ 1, data = cay_q)
results_mod1 <- bootstrap_bic(bootstrap_datalist(mod1, cay_q))
results_mod2 <- bootstrap_bic(bootstrap_datalist(mod2, cay_q))
cbind(results_mod1, results_mod2)


#---------------------------------------------
# Alternative variable selection methods
vars_to_track <- c("SVAR_LAG","LPE_LAG","CAY_LAG","LDY_LAG","BM_LAG",
                   "LDP_LAG","INFL_LAG","DFR_LAG","LTR_LAG","IK_LAG")

get_selected_vars <- function(data) {
  X <- as.matrix(data[, setdiff(names(data), "QERET")])
  Y <- data$QERET
  
  # Adaptive Lasso
  cv_ridge <- cv.glmnet(X, Y, family="gaussian", intercept=TRUE, standardize=TRUE,
                        type.measure="mse", nfold=10, alpha=0)
  ridge_coef <- coef(cv_ridge, s=cv_ridge$lambda.min)[-1]
  ridge_coef[ridge_coef == 0] <- 1e-6
  alasso <- glmnet(X, Y, family="gaussian", intercept=TRUE, standardize=TRUE,
                   alpha=1, penalty.factor=1/abs(ridge_coef))
  cv_alasso <- cv.glmnet(X, Y, family="gaussian", intercept=TRUE, standardize=TRUE,
                         type.measure="mse", nfold=10, alpha=1, penalty.factor=1/abs(ridge_coef),
                         keep=TRUE)
  fl2d <- data.frame(alasso$df, alasso$dev.ratio, alasso$lambda)
  colnames(fl2d) <- c("df","dev.ratio","lambda")
  thresh <- if(any(fl2d$df==6)) min(fl2d$lambda[fl2d$df==6]) else cv_alasso$lambda.min
  best_alasso_coef <- coef(cv_alasso, s = thresh)
  alasso_vars <- rownames(best_alasso_coef)[which(best_alasso_coef!=0)][-1]
  
  # FarmSelect
  farm_out <- farm.select(X, Y, K.factors=2, loss="lasso")
  farm_vars <- colnames(X)[farm_out$beta.chosen]
  
  # Bayesian Variable Selection
  bvs_mod <- Bvs(QERET ~ ., data=data, prior.betas="Robust", prior.models="Constant")
  bvs_vars <- bvs_mod$variables[bvs_mod$HPMbin == 1]
  
  list(alasso = alasso_vars, farm = farm_vars, bvs = bvs_vars)
}

results <- data.frame(matrix(0, nrow=length(samples), ncol=length(vars_to_track)))
rownames(results) <- names(samples)
colnames(results) <- vars_to_track

for (sample_name in names(samples)) {
  selected <- get_selected_vars(samples[[sample_name]])
  for (var in vars_to_track) {
    results[sample_name, var] <- sum(c(var %in% selected$alasso,
                                       var %in% selected$farm,
                                       var %in% selected$bvs))
  }
}

print(results)
