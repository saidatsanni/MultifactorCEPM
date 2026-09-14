
##OUT-OF-SAMPLE PREDICTION - Nonlinear and ML
rm(list=ls())

pkgs <- c("readxl","leaps", "mgcv", "randomForest", "gbm", "neuralnet", "lubridate", "dplyr", "stringi", "pls")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

##Load the dataset
# data_orig <- read_excel("./Datasets/qrtly_data_2020.xlsx")
data_orig <- read_excel("qrtly_data_2026.xlsx")
data_orig$Date <- as.Date(data_orig$Date, format = "%Y-%m-%d")
data_orig$Date_lag <- as.Date(data_orig$Date_lag, format = "%Y-%m-%d")
data_orig <- subset(data_orig, data_orig$Date_lag >= "1947-03-01" & data_orig$Date_lag <= "2025-12-01") 
data_orig <- data_orig[order(data_orig$Date), ]
data_orig$YEAR <- year(ymd(data_orig$Date))


get_bic_selected <- function(model_subset) {
  sm <- summary(model_subset)
  lbestt <- seq_along(sm$bic)[sm$bic == min(sm$bic)][1]
  tff <- sm$which[lbestt, ]
  sel <- names(tff)[tff][-1]
  return(sel)
}

##OOS PREDICTION
rd_seed <- c(1:10)
fin_pred <- list()
sel_vars <- list()

yt <- unique(data_orig$Date[data_orig$Date >= "1964-12-01" & data_orig$Date <= "2025-12-01"])
T1 <- length(yt) - 1


# Define predictor variables for main models
# Option 1: 3 predictors
# vars <- c("QERET", "SVAR_LAG", "LPE_LAG", "INFL_LAG")

# Option 2: All predictors
vars <- c("QERET", "SVAR_LAG","LPE_LAG","INFL_LAG","NTIS_LAG",
          "LDP_LAG","LDY_LAG","DFY_LAG","DFR_LAG","TMS_LAG",
          "RREL_LAG","BM_LAG","LTR_LAG","TBL_LAG","IK_LAG")

data_orig[,vars] <- data.frame(lapply(data_orig[,vars], function(x) as.numeric(as.character(x))))

for (t in 1:T1){
  
  train_data <- data_orig[data_orig$Date<=yt[t],][,vars]
  test_data <- data_orig[data_orig$Date==yt[t+1],][,vars]
  x_train <- as.matrix(train_data[, setdiff(vars, "QERET")])
  x_test <- as.matrix(test_data[, setdiff(vars, "QERET")])
  
  k <- nrow(test_data)
  
  # ###Variable Selection
  model_subset <- regsubsets(QERET ~ . , data = train_data, nbest = 2, nvmax = 3, really.big = TRUE)
  sel <- get_bic_selected(model_subset)
  #sel_vars[[t]] <- sel
  new_train <- data.frame(QERET = train_data$QERET, train_data[, sel, drop = FALSE])
  x_newtest <- test_data[, sel, drop = FALSE]
  
  ##models
  mod_int <- lm(QERET ~ 1, data=train_data)
  mod_int_pred <- predict(mod_int,data.frame(test_data))
  
  mod_lin <- lm(QERET ~ ., data=train_data)
  mod_lin_pred <-  predict(mod_lin,test_data)
  
  mod_mult <- lm(QERET ~ SVAR_LAG + LPE_LAG + INFL_LAG, data=train_data)
  mod_mult_pred <-  predict(mod_mult,test_data)
  
  mod_linsel <- lm(train_data$QERET ~ ., data = new_train)
  mod_linsel_pred <- predict(mod_linsel, x_newtest)
  
  #gam
  gam_formula <- as.formula(paste("QERET ~", paste("s(", setdiff(vars, "QERET"), ")", sep = "", collapse = " + ")))
  gam_pred <- tryCatch({
    gam_model <- mgcv::gam(gam_formula, data = train_data)
    mgcv::predict.gam(gam_model, test_data)
  }, error = function(e) {
    cat("GAM failed for time", t, ":", e$message, "\n")
    NA
  })

  # Initialize prediction matrices for RF and BRT
  all_rf1  <- matrix(NA, nrow = k, ncol = length(rd_seed))
  all_rf2  <- matrix(NA, nrow = k, ncol = length(rd_seed))
  all_brt1 <- matrix(NA, nrow = k, ncol = length(rd_seed))
  all_brt2 <- matrix(NA, nrow = k, ncol = length(rd_seed))

  # Train RF and BRT over multiple seeds
  for (s in 1:length(rd_seed)) {
    set.seed(rd_seed[s])
    all_rf1[, s]  <- predict(randomForest(QERET ~ ., data = train_data, importance = TRUE, ntree = 500),   test_data)

    set.seed(rd_seed[s])
    all_rf2[, s]  <- predict(randomForest(QERET ~ ., data = train_data, importance = TRUE, ntree = 10000), test_data)

    set.seed(rd_seed[s])
    all_brt1[, s] <- predict(gbm(QERET ~ ., data = train_data, distribution = "gaussian",
                                 n.trees = 1000,  shrinkage = 0.001, interaction.depth = 4), test_data)

    set.seed(rd_seed[s])
    all_brt2[, s] <- predict(gbm(QERET ~ ., data = train_data, distribution = "gaussian",
                                 n.trees = 10000, shrinkage = 0.001, interaction.depth = 2), test_data)
  }

  # Average across seeds
  rf1_pred  <- rowMeans(all_rf1,  na.rm = TRUE)
  rf2_pred  <- rowMeans(all_rf2,  na.rm = TRUE)
  brt1_pred <- rowMeans(all_brt1, na.rm = TRUE)
  brt2_pred <- rowMeans(all_brt2, na.rm = TRUE)

  ##NN
  ##Initialize prediction matrices for each NN architecture
  all_pred1 <- matrix(NA, nrow = k, ncol = length(rd_seed))
  all_pred2 <- matrix(NA, nrow = k, ncol = length(rd_seed))
  all_pred3 <- matrix(NA, nrow = k, ncol = length(rd_seed))

  # Neural network architectures
  nn_hidden <- list(c(5), c(5,3), c(32,16,8))
  all_pred_list <- list(all_pred1, all_pred2, all_pred3)

  # Train neural networks with multiple seeds
  for (s in 1:length(rd_seed)) {
    set.seed(rd_seed[s])

    for (i in 1:3) {
      nn_model <- neuralnet(QERET ~ ., data = train_data, hidden = nn_hidden[[i]],linear.output = TRUE, stepmax = 1e6)
      all_pred_list[[i]][,s] <- neuralnet::compute(nn_model, test_data)$net.result
    }
  }

  # Calculate average predictions
  modnn_predavg1 <- rowMeans(all_pred_list[[1]], na.rm = TRUE)
  modnn_predavg2 <- rowMeans(all_pred_list[[2]], na.rm = TRUE)
  modnn_predavg3 <- rowMeans(all_pred_list[[3]], na.rm = TRUE)

  ##PCR PACKAGE - UNSCALED
  pca1 <- pcr(train_data$QERET ~ x_train, ncomp = 3, scale = FALSE)
  pca_pred <- drop(predict(pca1, ncomp = 3, newdata = x_test))
  
  #pls
  pls1 <- plsr(train_data$QERET ~ x_train, ncomp = 3, scale = FALSE)
  pls_pred <- drop(predict(pls1, ncomp = 3, newdata = x_test))
  
  #SCALED PREDICTORS
  pcr_train <- data.frame(QERET = train_data$QERET, x_train)
  pca_fit <- pcr(QERET ~ ., data = pcr_train, ncomp = 3, scale = TRUE)
  pca_predsc <- drop(predict(pca_fit, newdata = data.frame(x_test), ncomp = 3))
  
  pls_fit <- plsr(QERET ~ ., data = pcr_train, ncomp = 3, scale = TRUE)
  pls_predsc <- drop(predict(pls_fit, newdata = data.frame(x_test), ncomp = 3))
  
  # ##Forecast combination (mean)
  uni_preds <- sapply(setdiff(vars, "QERET"), function (p){
    mod <- lm(as.formula(paste("QERET ~ ", p)), data = train_data)
    predict(mod, test_data)
  })
  comb_pred <- mean(uni_preds)
  
  #################25-year rolling window (100 quarters)
  W <- 100L
  if (nrow(train_data) >= W) {
    roll_data <- train_data[(nrow(train_data) - W + 1):nrow(train_data), ]
    
    rollint       <- lm(QERET ~ 1, data = roll_data)
    rollint_pred  <- predict(rollint, test_data)
    
    rollmult      <- lm(QERET ~ SVAR_LAG + LPE_LAG + INFL_LAG, data = roll_data)
    rollmult_pred <- predict(rollmult, test_data)
    
    rolllin       <- lm(QERET ~ ., data = roll_data)
    rolllin_pred  <- predict(rolllin, test_data)
    
    roll_uni <- sapply(setdiff(vars, "QERET"), function(p) {
      mod <- lm(as.formula(paste("QERET ~", p)), data = roll_data)
      predict(mod, test_data)
    })
    rollcomb_pred <- mean(roll_uni)
  } else {
    rollint_pred <- rollmult_pred <- rolllin_pred <- rollcomb_pred <- rep(NA_real_, k)
  }
  
  
  # Store results
  fin_pred[[t]] <- data.frame(
    Yvar = test_data$QERET,
    pred_int = mod_int_pred,
    pred_mult = mod_mult_pred,
    pred_lin = mod_lin_pred,
    pred_linsel = mod_linsel_pred,
    pred_gam = gam_pred,
    pred_rf1 = rf1_pred,
    pred_rf2 = rf2_pred,
    pred_brt1 = brt1_pred,
    pred_brt2 = brt2_pred,
    pred_nn1 = modnn_predavg1,
    pred_nn2 = modnn_predavg2,
    pred_nn3 = modnn_predavg3,
    pred_pca = pca_pred,
    pred_pcasc = pca_predsc,
    pred_pls = pls_pred,
    pred_plssc = pls_predsc,
    pred_comb = comb_pred,
    pred_rollint = rollint_pred,
    pred_rollmult = rollmult_pred,
    pred_rolllin = rolllin_pred,
    pred_rollcomb = rollcomb_pred
  )
}

# Combine all predictions
nn_df <- do.call(rbind, fin_pred)

# Calculate squared forecast errors
prediction_cols <- grep("^pred_", names(nn_df), value = TRUE)
for (col in prediction_cols) {
  nn_df[[paste0("fcsq_", gsub("pred_", "", col))]] <- (nn_df$Yvar - nn_df[[col]])^2
}

# Calculate out-of-sample R²
N <- nrow(nn_df)
bench_cols  <- c("int", "rollint")
model_names <- setdiff(gsub("pred_", "", prediction_cols), bench_cols)
bench_for   <- ifelse(grepl("^roll", model_names), "rollint", "int")

oos_results <- matrix(NA_real_, nrow = N, ncol = length(model_names))
colnames(oos_results) <- paste0("OOS_", toupper(model_names))

for (j in seq_along(model_names)) {
  num_col <- nn_df[[paste0("fcsq_", model_names[j])]]
  den_col <- nn_df[[paste0("fcsq_", bench_for[j])]]
  ok <- !is.na(num_col) & !is.na(den_col)
  for (i in 1:N) {
    idx <- i:N
    idx <- idx[ok[idx]]
    if (length(idx) > 0) {
      oos_results[i, j] <- 1 - (sum(num_col[idx]) / sum(den_col[idx]))
    }
  }
}

# Combine results
outr <- data.frame(oos_results)
print(t(as.matrix(outr[1,])))
