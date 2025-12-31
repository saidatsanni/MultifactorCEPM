# In-Sample Predictive Regression Analysis
# ===================================
rm(list=ls())

pkgs <- c("readxl", "sandwich", "lmtest", "lubridate", "dplyr")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

# DATA PREPARATION
load_data <- function(file = "./Datasets/G7_International_data.xlsx", sheet = "DATA_INT") {
  
  data <- read_excel(file, sheet = sheet)
  data <- subset(data, Date >= "1947-03-01" & Date <= "2020-12-31")

  data[sapply(data, is.factor)] <-
    lapply(data[sapply(data, is.factor)], function(x) as.numeric(as.character(x)))
  
  # Construct average G7 returns
  data$QERET_AVG <- rowMeans(
    data[, c("QERET_CN", "QERET_FR", "QERET_BD",
             "QERET_IT", "QERET_JP", "QERET_UK", "QERET_US")],
    na.rm = TRUE
  )
  
  return(data)
}

# Dependent variables
y_vars <- c("QERET_CN", "QERET_FR", "QERET_BD", "QERET_IT",
            "QERET_JP", "QERET_UK", "QERET_US", "QERET_AVG")

# Regressors (fixed)
x_vars <- c("VARQC.wfit_US_LAG", "LPE_US_LAG", "INFL_US_LAG")

# Run regressions
run_regressions <- function(data, y_vars, x_vars, nw_lags = 2) {
  
  results <- vector("list", length(y_vars))
  names(results) <- y_vars
  
  for (y in y_vars) {
    fml <- as.formula(paste(y, "~", paste(x_vars, collapse = " + ")))
    fit <- lm(fml, data = data)
    
    results[[y]] <- list(
      model   = fit,
      nw      = coeftest(fit, vcov = NeweyWest(fit, lag = nw_lags)),
      adj_r2  = summary(fit)$adj.r.squared
    )
  }
  
  results
}


# Wild bootstrap p-values
bootstrap_pvalues <- function(data, y, x_vars, nboot = 1000, nw_lags = 2) {
  
  formula <- as.formula(paste(y, "~", paste(x_vars, collapse = " + ")))
  
  orig_fit <- lm(formula, data = data)
  orig_nw  <- coeftest(orig_fit, vcov = NeweyWest(orig_fit, lag = nw_lags))
  orig_coef <- orig_nw[, 1]
  
  boot_data <- data
  for (var in x_vars) {
    if (var %in% names(orig_coef) && orig_coef[var] < 0) {
      boot_data[[var]] <- -boot_data[[var]]
    }
  }
  
  orig_fit <- lm(formula, data = boot_data)
  orig_nw  <- coeftest(orig_fit, vcov = NeweyWest(orig_fit, lag = nw_lags))
  orig_coef <- orig_nw[, 1]
  
  null_fit <- lm(as.formula(paste(y, "~ 1")), data = boot_data)
  resid  <- residuals(null_fit)
  fitted <- fitted(null_fit)
  
  boot_coefs <- matrix(NA, nboot, length(orig_coef))
  
  for (i in 1:nboot) {
    set.seed(i)
    boot_data[[y]] <- fitted + resid * rnorm(length(resid))
    
    boot_fit <- lm(formula, data = boot_data)
    boot_nw  <- coeftest(boot_fit, vcov = NeweyWest(boot_fit, lag = nw_lags))
    boot_coefs[i, ] <- boot_nw[, 1]
  }
  
  p_vals <- sapply(seq_along(orig_coef), function(j) {
    mean(boot_coefs[, j] > orig_coef[j])
  })
  
  names(p_vals) <- names(orig_coef)
  p_vals
}


# Format results table
format_table <- function(results, x_vars) {
  
  all_vars <- c("(Intercept)", x_vars)
  n_models <- length(results)
  
  coefs  <- matrix(NA, length(all_vars), n_models)
  tstats <- matrix(NA, length(all_vars), n_models)
  r2     <- sapply(results, function(x) x$adj_r2)
  
  rownames(coefs) <- rownames(tstats) <- all_vars
  colnames(coefs) <- colnames(tstats) <- names(results)
  
  for (i in seq_along(results)) {
    nw <- results[[i]]$nw
    coefs[rownames(nw), i]  <- nw[, 1]
    tstats[rownames(nw), i] <- nw[, 3]
  }
  
  list(coefs = coefs, tstats = tstats, r2 = r2)
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
bootstrap_all_models <- function(data, y_vars, x_vars, nboot = 1000) {
  
  boot_results <- list()
  
  cat("Running bootstrap for all dependent variables...\n")
  
  for (y in y_vars) {
    cat("Bootstrap:", y, "...\n")
    boot_results[[y]] <- bootstrap_pvalues(
      data = data,
      y = y,
      x_vars = x_vars,
      nboot = nboot
    )
  }
  
  boot_results
}

# Print bootstrap results
print_bootstrap <- function(boot_results) {
  
  cat("\nBootstrap p-values\n")
  cat("==================\n")
  
  for (y in names(boot_results)) {
    cat("\nDependent variable:", y, "\n")
    print(round(boot_results[[y]], 4))
  }
}

# Main analysis
analyze <- function(nboot = 1000) {
  
  data <- load_data()
  
  results <- run_regressions(data, y_vars, x_vars)
  formatted <- format_table(results, x_vars)
  
  print_results(formatted)
  
  boot_results <- bootstrap_all_models(data, y_vars, x_vars, nboot)
  print_bootstrap(boot_results)
  
  list(
    data = data,
    results = results,
    formatted = formatted,
    bootstrap = boot_results
  )
}


# Run analysis with bootstrap for all models
output <- analyze(nboot = 1000)
