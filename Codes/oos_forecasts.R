###############################################################################
#  Out-of-sample forecasts: Figure 1 , 2, and Table 6 input                   #
#                                                                             #
#  Recursive forecasts from 1965Q1 for two samples (ending 2020Q4, 2025Q4)    #
#  and two predictor sets (SVAR, LPE, INFL; all predictors).                  #                      #
#                                                                             #
#  Input :  Datasets/qrtly_data_2020.xlsx, Datasets/qrtly_data_2025.xlsx      #
#  Output:  output/oos_forecasts.xlsx                                         #
#           <sample>_<set>       out-of-sample R^2 by start quarter           #
#           <sample>_<set>_pred  forecasts and squared errors by quarter      #
#           sample = data2020, data2025;  set = pred3, predAll                #
###############################################################################


rm(list = ls())
start.time <- Sys.time()

library(readxl)
library(leaps)
library(mgcv)
library(randomForest)
library(gbm)
library(neuralnet)
library(pls)
library(writexl)

dir.create("output", showWarnings = FALSE)

n.cores <- parallel::detectCores() - 1
seeds     <- 1:10                                 
window    <- 100                                  
oos.start <- as.Date("1964-12-01")                

predictors3   <- c("SVAR_LAG", "LPE_LAG", "INFL_LAG")
predictorsAll <- c("SVAR_LAG", "LPE_LAG", "INFL_LAG", "NTIS_LAG", "LDP_LAG", "LDY_LAG",
                   "DFY_LAG", "DFR_LAG", "TMS_LAG", "RREL_LAG", "BM_LAG", "LTR_LAG",
                   "TBL_LAG", "IK_LAG")


###############################################################################
#  1. Functions                                                               #
###############################################################################
# BIC best subset; max 3
bic.subset <- function(train) {
  sub  <- regsubsets(QERET ~ ., data = train, nbest = 2, nvmax = 3, really.big = TRUE)
  sm   <- summary(sub)
  best <- which(sm$bic == min(sm$bic))[1]
  names(which(sm$which[best, ]))[-1]
}

# Mean combination forecast
combination <- function(train, test, predictors) {
  f <- numeric(length(predictors))
  for (j in seq_along(predictors))
    f[j] <- predict(lm(as.formula(paste("QERET ~", predictors[j])), data = train), test)
  mean(f)
}

# Forecasts for quarter t+1.
forecast.one.quarter <- function(t, data, dates, predictors) {

  vars  <- c("QERET", predictors)
  train <- data[data$Date <= dates[t],     vars]
  test  <- data[data$Date == dates[t + 1], vars]
  x.train <- as.matrix(train[, predictors])
  x.test  <- as.matrix(test[,  predictors])
  select  <- length(predictors) > 3        

  out <- data.frame(Date = dates[t + 1], Yvar = test$QERET)

  # historical average and linear models
  out$pred_int  <- predict(lm(QERET ~ 1, data = train), test)
  out$pred_mult <- predict(lm(QERET ~ SVAR_LAG + LPE_LAG + INFL_LAG, data = train), test)
  out$pred_lin  <- predict(lm(QERET ~ ., data = train), test)

  # BIC best-subset 
  if (select) {
    chosen <- bic.subset(train)
    train.sel <- train[, c("QERET", chosen)]
    out$pred_linsel <- predict(lm(QERET ~ ., data = train.sel), test[, chosen, drop = FALSE])
  }

  # generalized additive model
  gam.formula  <- as.formula(paste("QERET ~", paste0("s(", predictors, ")", collapse = " + ")))
  out$pred_gam <- tryCatch(predict(gam(gam.formula, data = train), test),
                           error = function(e) NA)

  # RF, BRT, NN
  rf1 <- rf2 <- brt1 <- brt2 <- nn1 <- nn2 <- nn3 <- numeric(length(seeds))
  for (s in seq_along(seeds)) {
    set.seed(seeds[s])
    rf1[s]  <- predict(randomForest(QERET ~ ., data = train, importance = TRUE, ntree = 500), test)
    set.seed(seeds[s])
    rf2[s]  <- predict(randomForest(QERET ~ ., data = train, importance = TRUE, ntree = 10000), test)
    set.seed(seeds[s])
    brt1[s] <- predict(gbm(QERET ~ ., data = train, distribution = "gaussian",
                           n.trees = 1000, shrinkage = 0.001, interaction.depth = 4),
                       test, n.trees = 1000)
    set.seed(seeds[s])
    brt2[s] <- predict(gbm(QERET ~ ., data = train, distribution = "gaussian",
                           n.trees = 10000, shrinkage = 0.001, interaction.depth = 2),
                       test, n.trees = 10000)
    set.seed(seeds[s])
    nn1[s] <- tryCatch(compute(neuralnet(QERET ~ ., data = train, hidden = 5,
                                         linear.output = TRUE, stepmax = 1e6),
                               test[, predictors])$net.result, error = function(e) NA)
    nn2[s] <- tryCatch(compute(neuralnet(QERET ~ ., data = train, hidden = c(5, 3),
                                         linear.output = TRUE, stepmax = 1e6),
                               test[, predictors])$net.result, error = function(e) NA)
    nn3[s] <- tryCatch(compute(neuralnet(QERET ~ ., data = train, hidden = c(32, 16, 8),
                                         linear.output = TRUE, stepmax = 1e6),
                               test[, predictors])$net.result, error = function(e) NA)
  }
  out$pred_rf1  <- mean(rf1)
  out$pred_rf2  <- mean(rf2)
  out$pred_brt1 <- mean(brt1)
  out$pred_brt2 <- mean(brt2)
  out$pred_nn1  <- mean(nn1, na.rm = TRUE)
  out$pred_nn2  <- mean(nn2, na.rm = TRUE)
  out$pred_nn3  <- mean(nn3, na.rm = TRUE)

  # principal components and partial least squares
  nc <- min(3, length(predictors))
  out$pred_pc  <- drop(predict(pcr(train$QERET ~ x.train,  ncomp = nc, scale = FALSE), ncomp = nc, newdata = x.test))
  out$pred_pls <- drop(predict(plsr(train$QERET ~ x.train, ncomp = nc, scale = FALSE), ncomp = nc, newdata = x.test))
  train.df <- data.frame(QERET = train$QERET, x.train)
  out$pred_pcplus  <- drop(predict(pcr(QERET ~ ., data = train.df,  ncomp = nc, scale = TRUE), newdata = data.frame(x.test), ncomp = nc))
  out$pred_plsplus <- drop(predict(plsr(QERET ~ ., data = train.df, ncomp = nc, scale = TRUE), newdata = data.frame(x.test), ncomp = nc))

  # combination forecast
  out$pred_comb <- combination(train, test, predictors)

  # rolling-window
  if (nrow(train) >= window) {
    roll <- train[(nrow(train) - window + 1):nrow(train), ]
    out$pred_rollint  <- predict(lm(QERET ~ 1, data = roll), test)
    out$pred_rollmult <- predict(lm(QERET ~ SVAR_LAG + LPE_LAG + INFL_LAG, data = roll), test)
    out$pred_rolllin  <- predict(lm(QERET ~ ., data = roll), test)
    out$pred_rollcomb <- combination(roll, test, predictors)
  } else {
    out$pred_rollint <- out$pred_rollmult <- out$pred_rolllin <- out$pred_rollcomb <- NA
  }
  out
}

oos.exercise <- function(file, end.date, predictors) {

  data <- as.data.frame(read_excel(file))
  data$Date     <- as.Date(data$Date)
  data$Date_lag <- as.Date(data$Date_lag)
  data <- data[data$Date_lag >= as.Date("1947-03-01") & data$Date_lag <= as.Date(end.date), ]
  data <- data[order(data$Date), ]
  for (v in c("QERET", predictors)) data[[v]] <- as.numeric(data[[v]])

  dates <- sort(unique(data$Date[data$Date >= oos.start & data$Date <= as.Date(end.date)]))
  T1    <- length(dates) - 1                

  # forecasts, quarter by quarter
  if (n.cores > 1) {
    cl <- parallel::makeCluster(n.cores)
    parallel::clusterEvalQ(cl, { library(leaps); library(mgcv); library(randomForest)
                                 library(gbm); library(neuralnet); library(pls) })
    parallel::clusterExport(cl, c("bic.subset", "combination", "seeds", "window"))
    forecasts <- parallel::parLapply(cl, 1:T1, forecast.one.quarter, data, dates, predictors)
    parallel::stopCluster(cl)
    forecasts <- do.call(rbind, forecasts)
  } else {
    forecasts <- NULL
    for (t in 1:T1) {
      if (t %% 20 == 1) cat("     forecast", t, "of", T1, "\n")
      forecasts <- rbind(forecasts, forecast.one.quarter(t, data, dates, predictors))
    }
  }

  # sq. forecast errors
  models <- sub("pred_", "", grep("^pred_", names(forecasts), value = TRUE))
  for (m in models) forecasts[[paste0("fcsq_", m)]] <- (forecasts$Yvar - forecasts[[paste0("pred_", m)]])^2

  # R2_OOS 
  N      <- nrow(forecasts)
  models <- setdiff(models, c("int", "rollint"))
  R2     <- matrix(NA, N, length(models), dimnames = list(NULL, paste0("OOS_", toupper(models))))
  for (j in seq_along(models)) {
    bench <- if (substr(models[j], 1, 4) == "roll") "rollint" else "int"
    e2  <- forecasts[[paste0("fcsq_", models[j])]]
    b2  <- forecasts[[paste0("fcsq_", bench)]]
    ok  <- !is.na(e2) & !is.na(b2)
    for (s in 1:N) {
      if (!ok[s]) next
      idx <- s:N
      idx <- idx[ok[idx]]
      R2[s, j] <- 1 - sum(e2[idx]) / sum(b2[idx])
    }
  }
  list(R2 = data.frame(Date = forecasts$Date, R2), forecasts = forecasts)
}

###############################################################################
#  2. Output                                                                  #
###############################################################################

sheets <- list()

for (sample in c("data2020", "data2025")) {
  if (sample == "data2020") { file <- "Datasets/qrtly_data_2020.xlsx"; end.date <- "2020-12-01" }
  if (sample == "data2025") { file <- "Datasets/qrtly_data_2025.xlsx"; end.date <- "2025-12-01" }

  for (set in c("pred3", "predAll")) {
    predictors <- if (set == "pred3") predictors3 else predictorsAll
    cat("\n", sample, set, format(Sys.time(), "%H:%M:%S"), "\n")
    res <- oos.exercise(file, end.date, predictors)
    sheets[[paste(sample, set, sep = "_")]]          <- res$R2
    sheets[[paste(sample, set, "pred", sep = "_")]]  <- res$forecasts
    cat("     R2_OOS from 1965Q1:\n")
    print(round(res$R2[1, -1], 4))
  }
}

write_xlsx(sheets, "output/oos_forecasts.xlsx")
writeLines(capture.output(sessionInfo()), "output/sessionInfo_oos.txt")

cat("\nForecasts written to output/oos_forecasts.xlsx\n")
cat("Elapsed time:", round(as.numeric(difftime(Sys.time(), start.time, units = "hours")), 2), "hours\n")
