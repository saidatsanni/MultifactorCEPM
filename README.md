# Code for "Searching for the Best Conditional Equity Premium Model by Guo Hui, Saidat Abidemi Sanni, and Yan Yu"

This is the ReadMe document for obtaining the main results in the paper.

## 1. Datasets
The **`Datasets`** folder contains the data files used for analysis.

- **`qrtly_data_2020.xlsx`**
  Contains quarterly returns and predictors.
  
- **`G7_International_data.xlsx`**
  Contains quarterly returns and predictors for G7 countries.

- **`daily_stock_return.xlsx`**
  Contains daily returns data.

- **`crsp_dailyret_1926.xlsx`**
  Contains daily returns data from 1926.

  

## 2. Codes
The **`Codes`** folder contains the contains scripts used to construct main result tables.

- **`VarSelection.R`**
Contains the R codes for the variable selection results.

- **`Insample_prediction_main.R`**
Contains the R codes for the in-sample prediction of one-quarter ahead returns using the full, first-half, or second-half samples.

- **`Insample_prediction_qeret4`**
Contains the R codes for the in-sample prediction of four-quarter ahead returns.

- **`Insample_prediction_inter.R`**
Contains the R codes for the in-sample prediction for G7 countries using the US forecasting variables. 

- **`OOS_Compare.R`**
Contains the R codes for the out-of-sample tests - HLN, ENC-NEW, and MSEF.

- **`Nonlinear_ML_Models.R`**
Contains the R codes for the out-of-sample analysis using non-linear and machine learning models.

- **`MIDAS_VarianceModel.R`**
Contains the R codes for MIDAS variance generation using daily returns, and the in-sample prediction with the MIDAS variance only.

- **`MIDAS_MultifactorModel.R`**
Contains the R codes for MIDAS variance generation using daily returns, and the in-sample prediction with MIDAS variance, LPE, and INFL.

- **`MIDAS_Model_GD.R`**
Contains the R codes for MIDAS variance generation using daily returns, the in-sample prediction with LPE, INFL and Dummy for the Great Depression Period.
