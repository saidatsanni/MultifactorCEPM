# Code for "Searching for the Best Conditional Equity Premium Model"
**Authors:** Hui Guo, Saidat Abidemi Sanni, and Yan Yu.

This is the ReadMe document for obtaining the main results in the paper.

## 1. Datasets
The **`Datasets`** folder contains the data files used for analysis.

- **`qrtly_data_2020.xlsx`**
  
  Quarterly stock returns and predictor variables.
  
- **`G7_International_data.xlsx`**
  
  Quarterly returns and predictor variables for G7 countries.

- **`daily_stock_return.xlsx`**
  
  Daily stock return data.

- **`crsp_dailyret_1926.xlsx`**

  CRSP daily stock return data starting in 1926.

  

## 2. Codes
The **`Codes`** folder contains the contains scripts used to construct main result tables.

- **`VarSelection.R`**

  Script for the variable selection analysis.

- **`Insample_prediction_main.R`**

  In-sample predictive regressions for one-quarter-ahead returns using the full sample, first-half sample, and second-half sample.

- **`Insample_prediction_qeret4.R`**

  In-sample predictive regressions for four-quarter-ahead returns.

- **`Insample_prediction_inter.R`**

  In-sample predictive regressions for G7 country returns using U.S. forecasting variables.

- **`OOS_Compare.R`**
  
  Out-of-sample forecast evaluation using the HLN, ENC-NEW, and MSE-F tests.
  
- **`Nonlinear_ML_Models.R`**

  Out-of-sample prediction using non-linear and machine learning models.

- **`MIDAS_VarianceModel.R`**

  MIDAS variance construction from daily returns and in-sample prediction using MIDAS variance only.

- **`MIDAS_MultifactorModel.R`**
  
  MIDAS variance construction from daily returns and in-sample prediction using MIDAS variance, LPE, and INFL.

- **`MIDAS_Model_GD.R`**
  
  MIDAS variance construction from daily returns and in-sample prediction using LPE, INFL, and a Great Depression dummy.
