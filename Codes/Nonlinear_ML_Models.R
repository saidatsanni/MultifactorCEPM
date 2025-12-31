
##OUT-OF-SAMPLE PREDICTION - Nonlinear and ML
rm(list=ls())

pkgs <- c("readxl", "mgcv", "randomForest", "gbm", "neuralnet", "lubridate", "dplyr", "stringi")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

##Load the dataset
data_orig <- read_excel("./Datasets/qrtly_data_2020.xlsx")
data_orig$Date <- as.Date(data_orig$Date, format = "%Y-%m-%d")
data_orig$Date_lag <- as.Date(data_orig$Date_lag, format = "%Y-%m-%d")
data_orig <- subset(data_orig, data_orig$Date_lag >= "1947-03-01" & data_orig$Date_lag <= "2020-12-01") 
data_orig$YEAR <- year(ymd(data_orig$Date))


##OOS PREDICTION
first_oos <- 1965
D1 <- first_oos - 1
rd_seed <- c(1:10)
fin_pred <- list()


T1 = ((max(as.numeric(data_orig$YEAR)) - (D1+1)))*4

# Define predictor variables for main models
# Option 1: 3 predictors
vars <- c("QERET", "SVAR_LAG", "LPE_LAG", "INFL_LAG")

# Option 2: All predictors
# vars <- c("QERET", "SVAR_LAG", "LPE_LAG","INFL_LAG", "NTIS_LAG",
#           "LDP_LAG", "LDY_LAG", "LDE_LAG", "DFY_LAG", "DFR_LAG", "TMS_LAG", "LTY_LAG",
#           "BM_LAG", "LTR_LAG", "TBL_LAG", "IK_LAG")

data_orig[,vars] <- data.frame(lapply(data_orig[,vars], function(x) as.numeric(as.character(x))))

for (t in 1:T1){
  
  yt <- unique(data_orig$Date[data_orig$Date >= "1964-12-01" & data_orig$Date <= "2020-12-01"])

  ##training/testing data
  train_data <- data_orig[data_orig$Date<=yt[t],][,vars]
  test_data <- data_orig[data_orig$Date==yt[t+1],][,vars]
  k <- nrow(test_data)
  
  ##models
  mod_int <- lm(QERET ~ 1, data=train_data)
  mod_int_pred <- predict(mod_int,data.frame(test_data))
  
  mod_lin <- lm(QERET ~ ., data=train_data)
  mod_lin_pred <-  predict(mod_lin,test_data)
  
  ##gam
  gam_formula <- as.formula(paste("QERET ~", paste("s(", setdiff(vars, "QERET"), ")", sep = "", collapse = " + ")))
  gam_pred <- tryCatch({
    gam_model <- mgcv::gam(gam_formula, data = train_data)
    mgcv::predict.gam(gam_model, test_data)
  }, error = function(e) {
    cat("GAM failed for time", t, ":", e$message, "\n")
    NA
  })
  
  ##Rf
  rf1 <- randomForest(QERET ~ ., data=train_data, importance=TRUE,ntree = 500)
  rf1_pred<- predict(rf1, test_data)
  
  rf2 <- randomForest(QERET ~ ., data=train_data, importance=TRUE,ntree = 10000)
  rf2_pred<- predict(rf2, test_data)
  

  ##brt
  brt1 <- gbm(QERET ~ ., data=train_data, distribution = "gaussian",
             n.trees = 1000, shrinkage = 0.001, interaction.depth = 4)
  brt1_pred <- predict(brt1, test_data)

  brt2 <- gbm(QERET ~ ., data=train_data, distribution = "gaussian",
              n.trees = 10000, shrinkage = 0.001, interaction.depth = 2)
  brt2_pred <- predict(brt2, test_data)
  
  ##NN
  # Initialize prediction matrices for each NN architecture
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
      nn_model <- neuralnet(QERET ~ ., data = train_data, hidden = nn_hidden[[i]],linear.output = TRUE)
      all_pred_list[[i]][,s] <- neuralnet::compute(nn_model, test_data)$net.result
    }
  }
  
  # Calculate average predictions
  modnn_predavg1 <- rowMeans(all_pred_list[[1]], na.rm = TRUE)
  modnn_predavg2 <- rowMeans(all_pred_list[[2]], na.rm = TRUE)
  modnn_predavg3 <- rowMeans(all_pred_list[[3]], na.rm = TRUE)
  
  # Store results
  fin_pred[[t]] <- data.frame(
    Yvar = test_data$QERET,
    pred_int = mod_int_pred,
    pred_lin = mod_lin_pred,
    pred_gam = gam_pred,
    pred_rf1 = rf1_pred,
    pred_rf2 = rf2_pred,
    pred_brt1 = brt1_pred,
    pred_brt2 = brt2_pred,
    pred_nn1 = modnn_predavg1,
    pred_nn2 = modnn_predavg2,
    pred_nn3 = modnn_predavg3
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
model_names <- gsub("pred_", "", prediction_cols[-1])  
oos_results <- matrix(0, nrow = N, ncol = length(model_names))
colnames(oos_results) <- paste0("OOS_", toupper(model_names))

for (i in 1:N) {
  for (j in seq_along(model_names)) {
    model_name <- model_names[j]
    oos_results[i, j] <- 1 - (sum(nn_df[[paste0("fcsq_", model_name)]][i:N]) / 
                                sum(nn_df$fcsq_int[i:N]))
  }
}

# Combine results
outr <- data.frame(oos_results)
print(t(as.matrix(outr[1,])))



