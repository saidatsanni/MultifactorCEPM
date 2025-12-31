
rm(list=ls())

pkgs <- c("readxl", "numDeriv", "lubridate", "dplyr")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

##Load the daily returns data
day_data <- read_excel("./Datasets/daily_stock_return.xlsx")

##load quarterly data
data_qtr <- read_excel("./Datasets/qrtly_data_2020.xlsx")
data_qtr$Date_lag <- as.Date(data_qtr$Date_lag, format= "%Y-%m-%d")
full_q <- subset(data_qtr, data_qtr$Date_lag >= "1947-03-01" & data_qtr$Date_lag <= "2020-12-01") 


# ##SUB SAMPLE 1 - 1947-1983:4
# day_data$DailyReturnDate <- as.Date(strptime(day_data$DailyReturnDate, "%Y%m%d"))
# full_first <- subset(data_qtr, data_qtr$Date_lag >= "1947-03-01" & data_qtr$Date_lag <= "1983-12-01")
# day_first <- subset(day_data, day_data$DailyReturnDate >= "1946-05-07" & day_data$DailyReturnDate<= "1983-12-30")

# ##SUB SAMPLE 2 - 1984:1-2020:4
# day_data$DailyReturnDate <- as.Date(strptime(day_data$DailyReturnDate, "%Y%m%d"))
# full_sec <- subset(data_qtr, data_qtr$Date_lag >= "1984-03-01" & data_qtr$Date_lag <= "2020-12-01")
# day_sec <- subset(day_data, day_data$DailyReturnDate >= "1983-04-05" & day_data$DailyReturnDate<= "2020-12-31")


##Assign the data sets to be used going forward, indicating the sample periods.
full_q <- full_q
day_data <- day_data


##CREATE THE INDEX NEEDED TO CONSTRUCT THE QUARTERLY
#1. Add an index to the daily data
day_data$index <- seq(1,nrow(day_data))

#2. Convert the daily return date to a date format
day_data$DailyReturnDate  <- as.Date(ymd(day_data$DailyReturnDate))

#3. Identify the last day of every month
day_data$yrmon <- substr(day_data$DailyReturnDate, start = 1, stop = 7)
day_data$mon <- substr(day_data$DailyReturnDate, start = 6, stop = 7)
day_data$dayy <- substr(day_data$DailyReturnDate, start = 9, stop = 10)

#4. Identify quarters in the daily day_data
day_data$dailyqt <- ifelse(day_data$mon %in% c("03","06","09","12"),1,0)
list <-day_data %>% group_by (yrmon)%>% 
  filter(dailyqt==1) %>% 
  filter(dayy == max(dayy))

##5. Use the list and index to filter the data rows with the last day of every quarter in the daily returns
T_idx <- list$index[-c(1:3)]



##DEFINE THE MIDAS FUNCTION
r <- day_data$Dailyreturn           #daily return
Y <- as.matrix(full_q$QERET)        #quarterly return
D <- 252
D1 <-252
n <- length(T_idx)

w=vt=matrix(0,nrow=length(T_idx));denom<-0
t=d=i=1

midas <- function(r,k1,k2){
  denom=sum(exp(((1:D1)-1)*k1+k2*((1:D1)-1)^2))
  for (t in 1:n){
    w <- exp(k1*(1:D-1) + k2*(1:D-1)^2)/denom
    vt[t,] <- 66*sum(w*((r[T_idx[t]:(T_idx[t]-(D-1))]^2)))
  }
  list(vt,w)
}


##STEPS TO SELECTING k1 and k2 values
#1. Grid search to find the initial starting values
#A. Iterative Algorithm
log_lik_iter<- function(k,y,r,beta) {
  vtm <-midas(r,k[1],k[2])[[1]]
  mu <- beta[1] + beta[2]*vtm
  logl_vec2 <- -(1/2)*sum(log(vtm)) - (1/2)*sum((y - mu)^2/vtm)
  return(-logl_vec2)
}

#B. Search for k1 and k2 starting values using a random search of values.
grid1 <- seq(-0.00001,-0.005, -0.0001)
grid2 <- seq (-0.00001,-0.005, -0.0001)

ll_save<-matrix(NA, nrow = length(grid1), ncol = length(grid2))
for (l in 1:length(grid1)){
  for (m in 1:length(grid2)){
    K <- c(grid1[l], grid2[m])
    k.old = K
    epsilon = 10^(-6)
    delta=1
    step=0
    logL.old=1
    
    while((delta>=epsilon)&(step<50)){
      step=step+1
      vtm <-midas(r,K[1],K[2])[[1]]
      wts <- 1/vtm
      vt_wls <- lm(Y ~ vtm,weights=wts)
      beta = vt_wls$coefficients
      
      #maximize log-likelihood;
      optim <- optim(k.old, log_lik_iter, y=Y,r=r, beta=beta)
      k.new = optim$par
      logL = optim$value
      delta = max(abs(k.new-k.old))
      delta2 = abs(logL-logL.old)
      k.old = k.new
      logL.old = logL
      ll_save[l,m] = -logL
    }
  }
}

indices<-which(ll_save == max(ll_save), arr.ind=TRUE)
k_1<-grid1[indices[1]]
k_2<- grid2[indices[2]]


#2. Use the optimal values above as starting values in the iterative algorithm
##Using these values in the one-step algorithm yields similar values.
K= c(k_1,k_2) 
k.old = K
epsilon = 10^(-6)
delta=1
step=0
logL.old=1

while((delta>=epsilon)&(step<50)){
  step=step+1
  vtm <-midas(r,K[1],K[2])[[1]]
  wts <- 1/vtm
  vt_wls <- lm(Y ~ vtm,weights=wts)
  beta = vt_wls$coefficients
  
  #maximize log-likelihood;
  optim <- optim(k.old, log_lik_iter, y=Y,r=r, beta=beta)
  k.new = optim$par
  logL = optim$value
  delta = max(abs(k.new-k.old))
  delta2 = abs(logL-logL.old)
  k.old = k.new
  logL.old = logL
}     #end of iteration;

step; delta; k.new; -logL
k1_it <- k.new[1]
k2_it <- k.new[2]


##1. ONE-STEP JOINT OPTIMIZATION
log_lik <- function(vec,y,r) {
  b0 <- vec[1]
  b1 <- vec[2]
  k1 <-vec[3]
  k2 <- vec[4]
  vtm <-midas(r,k1,k2)[[1]]
  mu <- b0 + b1*vtm
  logl <- - (1/2)*sum(log(vtm)) - (1/2)*sum((y - mu)^2/vtm)
  return(-logl)
}



##2. Obtain the initial values of the parameters via weighted least squares
k1 = k1_it
k2 = k2_it
midasvar <- midas(r, k1,k2)[[1]]
wts <- 1/midasvar
mod_wls <- lm(QERET ~ midasvar, data= full_q, weights=wts)
zeta0 <- c(coef(mod_wls),k1,k2) 


##3. Optimize
er<-optim(zeta0,log_lik,y=Y, r=r,hessian=TRUE)
beta <- er$par
beta
k1_opt <- er$par[3]
k2_opt <- er$par[4]

##4. Obtain the midas variance using the optimal k1 and k2 values
vtmidas_opt <- midas(r,k1 = k1_opt, k2 = k2_opt)[[1]]
full_q$vtmidas_opt <- as.numeric(vtmidas_opt)

##Obtain the wls regression
# wts <- 1/vtmidas_opt
# vt_wls <- lm(QERET ~ vtmidas_opt,weights=wts, data= full_q)
# summary(vt_wls)

##5. Compute the standard error of beta using the sandwich formula
vtm <- vtmidas_opt
mu <- er$par[1] + er$par[2]*vtm

log_lik_vec <- function(vec,y,v) {
  b0 <- vec[1]
  b1 <- vec[2]
  k1 <-vec[3]
  k2 <- vec[4]
  vtm <-midas(v,k1,k2)[[1]]
  mu <- b0 + b1*vtm
  logl_vec <- -(1/2)*log(vtm) - (1/2)*((y - mu)^2/vtm)
  return(logl_vec)
}

p = 2 # number of regression parameters

G <- jacobian(log_lik_vec, er$par, y = Y, v=r)
Omega <- (t(G) %*% G)/n

v_cov1 <- (1/n)* solve(er$hessian[1:p,1:p]/n) %*% Omega[1:p,1:p]%*% t(solve(er$hessian[1:p,1:p]/n))
sand_se1 <- sqrt(abs(diag(v_cov1[1:p,1:p])))
summary(vt_wls); sand_se1

se <- sand_se1
tvals <- beta[1:p]/se
tvals

p0 <- 2*pt(-abs(tvals[1]), df=dim(full_q)[1]-p)
p1 <- 2*pt(-abs(tvals[2]), df=dim(full_q)[1]-p)
p0; p1
