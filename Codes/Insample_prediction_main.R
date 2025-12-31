# In-Sample Predictive Regression Analysis
# ===================================
rm(list=ls())

# Load required packages
pkgs <- c("readxl", "sandwich", "lmtest", "lubridate", "dplyr")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

# DATA PREPARATION
# Load data
load_data <- function(file = "./Datasets/qrtly_data_2020.xlsx") {
  data <- read_excel(file)
  data$Date_lag <- as.Date(data$Date_lag, format = "%Y-%m-%d")
  data <- subset(data, Date_lag >= "1947-03-01" & Date_lag <= "2020-12-01")
  data[sapply(data, is.factor)] <- lapply(data[sapply(data, is.factor)], 
                                          function(x) as.numeric(as.character(x)))
  return(data)
}

# Model specifications
get_models <- function() {
  list(
    "SVAR_LAG",
    "LPE_LAG", 
    "INFL_LAG",
    c("SVAR_LAG", "LPE_LAG"),
    c("SVAR_LAG", "INFL_LAG"),
    c("LPE_LAG", "INFL_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG"),
    c("SVAR_LAG", "LPE_LAG", "TBL_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "TBL_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "DFR_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "LTR_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "IK_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "UNEM_LAG"),
    c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "TBL_LAG", "DFR_LAG", "LTR_LAG", "IK_LAG", "UNEM_LAG")
  )
}

# Run all regressions
run_regressions <- function(data, y = "QERET", nw_lags = 2) {
  models <- get_models()
  results <- list()
  
  for (i in seq_along(models)) {
    # Fit model
    formula <- as.formula(paste(y, "~", paste(models[[i]], collapse = " + ")))
    fit <- lm(formula, data = data)
    
    # Store results
    results[[i]] <- list(
      model = fit,
      nw = coeftest(fit, vcov = NeweyWest(fit, lag = nw_lags)),
      adj_r2 = summary(fit)$adj.r.squared
    )
  }
  
  return(results)
}

# Wild bootstrap p-values
bootstrap_pvalues <- function(data, predictors, y = "QERET", nboot = 1000, nw_lags = 2) {
  # Run original regression to check coefficient signs
  formula <- as.formula(paste(y, "~", paste(predictors, collapse = " + ")))
  orig_fit <- lm(formula, data = data)
  orig_nw <- coeftest(orig_fit, vcov = NeweyWest(orig_fit, lag = nw_lags))
  orig_coef <- orig_nw[, 1]  # Original coefficients
  
  # Create modified data by flipping signs of variables with negative coefficients
  boot_data <- data
  for (var in predictors) {
    if (var %in% names(orig_coef) && orig_coef[var] < 0) {
      boot_data[[var]] <- -boot_data[[var]]
    }
  }
  
  # Original model (after conditional negation, coefficients should be positive)
  orig_fit <- lm(formula, data = boot_data)
  orig_nw <- coeftest(orig_fit, vcov = NeweyWest(orig_fit, lag = nw_lags))
  orig_coef <- orig_nw[, 1]  # Coefficients
  orig_tstat <- orig_nw[, 3] # T-statistics
  
  # Null model residuals (constant-only model under null hypothesis)
  null_fit <- lm(as.formula(paste(y, "~ 1")), data = boot_data)
  resid <- residuals(null_fit)
  fitted <- fitted(null_fit)
  
  # Bootstrap under null hypothesis (predictors have no effect)
  boot_tstats <- matrix(NA, nboot, length(orig_coef))
  boot_coefs <- matrix(NA, nboot, length(orig_coef))
  
  for (i in 1:nboot) {
    set.seed(i)
    # Generate bootstrap data under null (no predictor effects)
    boot_y <- fitted + resid * rnorm(length(resid), 0, 1)
    boot_data[[y]] <- boot_y
    
    # Fit model to bootstrap data
    boot_fit <- lm(formula, data = boot_data)
    boot_nw <- coeftest(boot_fit, vcov = NeweyWest(boot_fit, lag = nw_lags))
    boot_tstats[i, ] <- boot_nw[, 3]  # Store t-statistics
    boot_coefs[i, ] <- boot_nw[, 1]  # Store coefs
    
  }
  
  # P-values: Compare original coefficients to bootstrap coefficients
  p_vals <- sapply(1:length(orig_coef), function(j) {
    sum(boot_coefs[, j] > orig_coef[j]) / nboot
  })
  names(p_vals) <- names(orig_coef)
  
  return(p_vals)
}

# Format results table
format_table <- function(results) {
  n_models <- length(results)
  
  # Get all variables
  all_vars <- unique(c("(Intercept)", unlist(get_models())))
  
  # Initialize matrices
  coefs <- matrix(NA, length(all_vars), n_models)
  tstats <- matrix(NA, length(all_vars), n_models)
  r2 <- sapply(results, function(x) x$adj_r2)
  
  rownames(coefs) <- rownames(tstats) <- all_vars
  colnames(coefs) <- colnames(tstats) <- 1:n_models
  
  # Fill matrices
  for (i in 1:n_models) {
    nw_result <- results[[i]]$nw
    for (var in rownames(nw_result)) {
      idx <- match(var, all_vars)
      coefs[idx, i] <- nw_result[var, 1]
      tstats[idx, i] <- nw_result[var, 3]
    }
  }
  
  return(list(coefs = coefs, tstats = tstats, r2 = r2))
}

# Print results
print_results <- function(formatted) {
  cat("Coefficients:\n")
  print(round(formatted$coefs, 4), na.print = "")
  cat("\nT-statistics:\n") 
  print(round(formatted$tstats, 3), na.print = "")
  cat("\nAdjusted R-squared:\n")
  print(round(formatted$r2, 3))
}

# Bootstrap all models
bootstrap_all_models <- function(data, nboot = 1000) {
  models <- get_models()
  boot_results <- list()
  
  cat("Running bootstrap for all models...\n")
  
  for (i in seq_along(models)) {
    cat("Model", i, "...")
    boot_results[[i]] <- bootstrap_pvalues(data, models[[i]], nboot = nboot)
    cat(" done\n")
  }
  
  return(boot_results)
}

# Print bootstrap results
print_bootstrap <- function(boot_results) {
  models <- get_models()
  
  cat("\nBootstrap p-values for all models:\n")
  cat("==================================\n")
  
  for (i in seq_along(boot_results)) {
    cat("\nModel", i, "(", paste(models[[i]], collapse = ", "), "):\n")
    print(round(boot_results[[i]], 4))
  }
}

# Main analysis
analyze <- function(nboot = 1000) {
  # Load data and run regressions
  data <- load_data()
  results <- run_regressions(data)
  formatted <- format_table(results)
  
  # Print regression results
  print_results(formatted)
  
  # Bootstrap all models
  boot_results <- bootstrap_all_models(data, nboot)
  print_bootstrap(boot_results)
  
  return(list(
    data = data, 
    results = results, 
    formatted = formatted,
    bootstrap = boot_results
  ))
}

# Run analysis with bootstrap for all models
output <- analyze(nboot = 1000)
