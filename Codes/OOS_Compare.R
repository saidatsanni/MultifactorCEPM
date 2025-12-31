
# OUT-OF-SAMPLE TESTS
rm(list=ls())

pkgs <- c("readxl", "lubridate", "dplyr")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

##Load the dataset
data_orig <- read_excel("./Datasets/qrtly_data_2020.xlsx")
data_orig$Date <- as.Date(data_orig$Date, format = "%Y-%m-%d")
data_orig$Date_lag <- as.Date(data_orig$Date_lag, format = "%Y-%m-%d")
data_orig <- subset(data_orig, data_orig$Date_lag >= "1947-03-01" & data_orig$Date_lag <= "2020-12-01") 
data_orig$YEAR <- year(ymd(data_orig$Date))


###BEGIN THE OOS PREDICTION
first_oos <- 1965
D1 <- first_oos - 1
rd_seed <- c(1:10)
fin_pred <- list()


T1 = ((max(as.numeric(data_orig$YEAR)) - (D1+1)))*4

for (t in 1:T1){
  
  yt <- unique(data_orig$Date[data_orig$Date >= "1964-12-01" & data_orig$Date <= "2020-12-01"])
  
  ########training/testing data
  train_data <- data_orig[data_orig$Date<=yt[t],]
  test_data <- data_orig[data_orig$Date==yt[t+1],]
  
  # Fit models
  mod_int <- lm(QERET ~ 1, data = train_data)
  mod_lin <- lm(QERET ~ SVAR_LAG + LPE_LAG + INFL_LAG, data = train_data)
  
  # Univariate models
  predictors <- c("SVAR_LAG", "LPE_LAG","INFL_LAG", "NTIS_LAG", 
                  "LDP_LAG", "LDY_LAG", "LDE_LAG", "DFY_LAG", "DFR_LAG", "TMS_LAG", "LTY_LAG",
                  "BM_LAG", "LTR_LAG", "TBL_LAG", "IK_LAG")
  
  uni_preds <- sapply(predictors, function(p) {
    mod <- lm(as.formula(paste("QERET ~", p)), data = train_data)
    predict(mod, test_data)
  })
  
  # Predictions
  pred_int <- predict(mod_int, test_data)
  pred_lin <- predict(mod_lin, test_data)
  pred_rMC <- mean(uni_preds)
  
  # Forecast squared errors
  actual <- test_data$QERET
  fcsq_int <- (actual - pred_int)^2
  fcsq_lin <- (actual - pred_lin)^2
  fcsq_rMC <- (actual - pred_rMC)^2
  
  # Store results
  fin_pred[[t]] <- data.frame(test_data$Date, actual, pred_int, pred_lin, pred_rMC, fcsq_int, fcsq_lin, 
                              fcsq_rMC)
}

# Compile results
nn_df <- do.call(rbind, fin_pred)
colnames(nn_df) <- c("Date", "Yvar", "pred_int", "pred_lin", "pred_rMC", "fcsq_int", "fcsq_lin", "fcsq_rMC")

N <- nrow(nn_df)

# Calculate OOS R2
OOS_LIN <-  OOS_rMC <- rep(0, N)
for (i in 1:N) {
  OOS_LIN[i] <- 1 - (sum(nn_df$fcsq_lin[i:N]) / sum(nn_df$fcsq_int[i:N]))
  OOS_rMC[i] <- 1 - (sum(nn_df$fcsq_rMC[i:N]) / sum(nn_df$fcsq_int[i:N]))
}

# Results
outr <- data.frame(nn_df$Date, OOS_LIN, OOS_rMC)
print(t(as.matrix(outr[1, ][-1])))


#######################################OOS TESTS
calculate_hln_test <- function(actual, forecast1, forecast2) {
  lambda=(forecast2-forecast1)/(actual-forecast1)
  u_1=actual-forecast1;
  u_2=actual-forecast2;
  d=(u_1-u_2)*u_1;
  n=length(u_1)
  d_bar=(1/n)*sum(d)
  phi_0=as.numeric((1/n)*t(as.matrix(d-d_bar))%*%(as.matrix(d-d_bar)))
  V_d_bar=(1/n)*phi_0;
  hln_stat =((V_d_bar)^(-0.5))*d_bar
  list(statistic = hln_stat, p_value = 1 - pnorm(hln_stat))
}

calculate_enc_test <- function(actual, forecast1, forecast2) {
  u1 <- actual - forecast1
  u2 <- actual - forecast2
  d <- u1^2 - u2 * u1
  p <- length(u1)
  d_bar <- mean(d)
  dd_bar <- mean(u2^2)
  enc_stat <- p * (d_bar / dd_bar)
  list(statistic = enc_stat, p_value = 1 - pnorm(enc_stat))
}

calculate_msef_test <- function(actual, forecast1, forecast2) {
  u1 <- actual - forecast1
  u2 <- actual - forecast2
  num <- sum(u1^2 - u2^2)
  denom <- mean(u2^2)
  msef_stat <- num / denom
  list(statistic = msef_stat, p_value = 1 - pnorm(msef_stat))
}

# Calculate and store original test statistics
hln_result <- calculate_hln_test(nn_df$Yvar, nn_df$pred_lin, nn_df$pred_rMC)
enc_result <- calculate_enc_test(nn_df$Yvar, nn_df$pred_int, nn_df$pred_lin)
msef_result <- calculate_msef_test(nn_df$Yvar, nn_df$pred_int, nn_df$pred_lin)

# Print original test results
cat("Original HLN Test - Statistic:", hln_result$statistic, "P-value:", hln_result$p_value, "\n")
cat("Original ENC Test - Statistic:", enc_result$statistic, "P-value:", enc_result$p_value, "\n")
cat("Original MSEF Test - Statistic:", msef_result$statistic, "P-value:", msef_result$p_value, "\n")

# Store statistics for bootstrap comparison
enc_stat <- enc_result$statistic
msef_stat <- msef_result$statistic



# Bootstrap p-values
bootstrap_data <- subset(data_orig, data_orig$Date_lag >= "1947-03-01" & data_orig$Date_lag <= "2020-09-01") 

bootstrap_p_values <- function(bootstrap_data, B = 1000, enc_stat, msef_stat) {
  
  set.seed(1)
  nobs <- nrow(bootstrap_data)
  
  # Estimate individual equations of the VAR system
  lmod <- lm(LPE ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bootstrap_data)
  vmod <- lm(SVAR ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bootstrap_data)
  imod <- lm(INFL ~ LPE_LAG + SVAR_LAG + INFL_LAG + QERET_LAG, data = bootstrap_data)
  rmod <- lm(QERET ~ 1, data = bootstrap_data)  
  
  coefs <- rbind(
    coef(lmod),
    coef(vmod),
    coef(imod),
    c(coef(rmod)[1], 0, 0, 0, 0) 
  )
  
  errs <- data.frame(
    lpe_resid = resid(lmod),
    svar_resid = resid(vmod),
    infl_resid = resid(imod),
    qeret_resid = resid(rmod)
  )
  
  bootstrap_stats <- replicate(B, {
    
    # Sample residuals with replacement for B observations
    resid_indices <- sample(nobs, B, replace = TRUE)
    bootstrap_residuals <- as.matrix(errs[resid_indices, ])
    
    xsim <- matrix(NA, B, 4)
    colnames(xsim) <- c("LPE", "SVAR", "INFL", "QERET")
    
    initial_values <- c(
      mean(bootstrap_data$LPE, na.rm = TRUE),
      mean(bootstrap_data$SVAR, na.rm = TRUE),
      mean(bootstrap_data$INFL, na.rm = TRUE),
      mean(bootstrap_data$QERET, na.rm = TRUE)
    )
    
    # First observation
    xsim[1, ] <- coefs[, 1] + coefs[, 2:5] %*% initial_values + bootstrap_residuals[1, ]
    
    # Subsequent observations 
    for (j in 2:B) {
      xsim[j, ] <- coefs[, 1] + coefs[, 2:5] %*% xsim[j-1, ] + bootstrap_residuals[j, ]
    }
    
    # Use last nobs observations from the B-length simulation
    start_idx <- B - nobs + 1
    end_idx <- B
    
    xsim_extract <- xsim[start_idx:end_idx, ]
    xsim_lag_extract <- xsim[(start_idx-1):(end_idx-1), ]
    
    # Create dates sequence matching original data
    date_seq <- seq(from = as.Date("1947-03-01"), by = "quarter", length.out = nobs)
    
    bootstrap_sim_data <- data.frame(
      Date = date_seq,
      LPESIM = xsim_extract[, 1],
      VARSIM = xsim_extract[, 2],
      INFLSIM = xsim_extract[, 3],
      QERETSIM = xsim_extract[, 4],
      LPESIM_LAG = xsim_lag_extract[, 1],
      VARSIM_LAG = xsim_lag_extract[, 2],
      INFLSIM_LAG = xsim_lag_extract[, 3],
      QERETSIM_LAG = xsim_lag_extract[, 4]
    )
    
    in_sample_end_date <- as.Date("1964-09-01")
    in_sample_size <- which(bootstrap_sim_data$Date == in_sample_end_date)
    
    if (length(in_sample_size) == 0 || in_sample_size >= nobs) {
      return(c(NA, NA))  
    }
    
    # Out-of-sample actual values
    oos_indices <- (in_sample_size + 1):nobs
    actual_sim <- bootstrap_sim_data$QERETSIM[oos_indices]
    
    # historical mean forecast
    forecast_const <- rep(mean(bootstrap_sim_data$QERETSIM[1:in_sample_size]), 
                          length(actual_sim))
    
    # recursive regression forecast
    forecast_complex <- sapply(1:length(actual_sim), function(i) {
      train_end <- in_sample_size + i - 1
      train_data <- bootstrap_sim_data[1:train_end, ]
      
      # Fit model on expanding window
      tryCatch({
        mod <- lm(QERETSIM ~ LPESIM_LAG + VARSIM_LAG + INFLSIM_LAG, 
                  data = train_data)
        predict(mod, bootstrap_sim_data[train_end + 1, ])
      }, error = function(e) {
        mean(train_data$QERETSIM)  
      })
    })
    
    # Forecast errors
    u1 <- actual_sim - forecast_const      
    u2 <- actual_sim - forecast_complex   
    d <- (u1^2) - (u2 * u1)               
    
    if (mean(u2^2) == 0) return(c(NA, NA))
    
    # Test statistics
    enc_boot <- length(u1) * (mean(d) / mean(u2^2))
    msef_boot <- sum((u1^2) - (u2^2)) / mean(u2^2)
    
    c(enc_boot, msef_boot)
  })
  
  valid_sims <- !is.na(bootstrap_stats[1, ])
  bootstrap_stats <- bootstrap_stats[, valid_sims]
  
  if (ncol(bootstrap_stats) == 0) {
    return(list(enc_pval = NA, msef_pval = NA, valid_sims = 0))
  }
  
  # Calculate p-values
  enc_pval <- mean(bootstrap_stats[1, ] > enc_stat, na.rm = TRUE)
  msef_pval <- mean(bootstrap_stats[2, ] > msef_stat, na.rm = TRUE)
  
  list(
    enc_pval = enc_pval,
    msef_pval = msef_pval,
    valid_sims = ncol(bootstrap_stats),
    bootstrap_stats = bootstrap_stats
  )
}

boot_results <- bootstrap_p_values(bootstrap_data, B = 1000, enc_stat, msef_stat)

# Display results
cat("=== BOOTSTRAP FORECAST EVALUATION RESULTS ===\n")
cat("Valid simulations:", boot_results$valid_sims, "out of 1000\n")
cat("Bootstrap p-values:\n")
cat("  - ENC test:", sprintf("%.4f", boot_results$enc_pval), "\n")
cat("  - MSEF test:", sprintf("%.4f", boot_results$msef_pval), "\n")
