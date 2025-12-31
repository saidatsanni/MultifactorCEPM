
rm(list=ls())

pkgs <- c("readxl", "numDeriv", "lubridate", "dplyr", "Hmisc")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(c(pkgs), library, character.only = TRUE))

##load the data
day_data <- read_excel("./Datasets/crsp_dailyret_1926.xlsx")
day_data$Date <- as.Date(day_data$Date, format= "%Y-%m-%d")
full_q <- subset(day_data, day_data$Date >= "1928-06-30" & day_data$Date <= "2021-03-31")
DATE <- full_q$Date

##Step 1: Effects of Great Depression
D1 <- ifelse((full_q$Date >= "1930-09-01" & full_q$Date < "1933-12-01"),1,0)
full_q$D1_LAG <- as.numeric(Lag(D1,shift = 1))
full_q$D1_LAG[1]<- 0
full_q$LPEGD_LAG <- full_q$LPE_LAG*full_q$D1_LAG
full_q$INFLGD_LAG <- full_q$INFL_LAG*full_q$D1_LAG
full_q$DUMMY_LAG <- full_q$D1_LAG
DUMMY_LAG <- full_q$D1_LAG

##Specify the variables
X<- data.frame(full_q$LPE_LAG,full_q$INFL_LAG,full_q$LPEGD_LAG,full_q$INFLGD_LAG)
Y <- as.matrix(full_q$QERET)


##Step 2: OBTAIN THE MIDAS VARIANCE AND MIDAS WITH DUMMY
day_data$index <- seq(1,nrow(day_data))
day_data$DailyReturnDate_504  <- as.Date(ymd(day_data$DailyReturnDate_504))
day_data$yrmon <- substr(day_data$DailyReturnDate_504, start = 1, stop = 7)
day_data$mon <- substr(day_data$DailyReturnDate_504, start = 6, stop = 7)
day_data$dayy <- substr(day_data$DailyReturnDate_504, start = 9, stop = 10)
day_data$dailyqt <- ifelse(day_data$mon %in% c("03","06","09","12"),1,0)
list <-day_data %>% 
  group_by (yrmon)%>% 
  filter(dailyqt==1) %>% 
  filter(dayy == max(dayy))

T_idx <- list$index[-c(1:6)] 

##MIDAS FUNCTION
r <- day_data$Dailyreturn_504         
D <- 504
D1 <- 504
n <- length(T_idx)


w=matrix(0,nrow=length(T_idx)); vt=matrix(0,nrow=length(T_idx)); denom<-0
t=d=i=1

midas <- function(r,k1,k2){
  denom = sum(exp(((1:D1)-1)*k1+k2*((1:D1)-1)^2))
  for (t in 1:n){
    w <- exp(k1*(1:D-1) + k2*(1:D-1)^2)/denom
    vt[t,] <- 66*sum(w*((r[T_idx[t]:(T_idx[t]-(D-1))]^2)))
  }
  list(vt,w)
}

##ITERATIVE ALGORITHM
log_lik_iter<- function(k,x,y,r,beta) {
  vtm <-midas(r,k[1],k[2])[[1]]
  vtm_D <- vtm*DUMMY_LAG
  mu <- beta[1] + beta[2]*vtm + beta[3]*x[,1]+beta[4]*x[,2] + beta[5]*vtm_D+beta[6]*x[,3] + beta[7]*x[,4]
  logl_vec2 <- -(1/2)*sum(log(vtm)) - (1/2)*sum((y - mu)^2/vtm)
  return(-logl_vec2)
}

###GRID SEARCH FOR K1 AND K2 STARTING VALUES
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
      vtm_D <- vtm*DUMMY_LAG
      wts <- 1/vtm
      vt_wls <- lm(Y ~ vtm+ X[,1]+X[,2]+ vtm_D +X[,3]+X[,4],weights=wts)
      beta = vt_wls$coefficients
      
      #maximize log-likelihood;
      optim <- optim(k.old, log_lik_iter, x=X, y=Y,r=r, beta=beta)
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


##Use the the k.new value from the iterative algorithm as the starting values in the one-step algorithm
K= c(k_1,k_2)
k.old = K
epsilon = 10^(-6)
delta=1
step=0
logL.old=1

while((delta>=epsilon)&(step<50)){
  step=step+1
  vtm <-midas(r,K[1],K[2])[[1]]
  vtm_D <- vtm*DUMMY_LAG
  wts <- 1/vtm
  vt_wls <- lm(Y ~ vtm+ X[,1]+X[,2]+ vtm_D +X[,3]+X[,4],weights=wts)
  beta = vt_wls$coefficients
  
  #maximize log-likelihood;
  optim <- optim(k.old, log_lik_iter, x=X, y=Y,r=r, beta=beta)
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
k1_it;k2_it


##ONE-STEP JOINT OPTIMIZATION
log_lik <- function(vec,x,y,r) {
  b0 <- vec[1]
  b1 <- vec[2]
  b2 <- vec[3]
  b3 <- vec[4]
  b4 <- vec[5]
  b5 <- vec[6]
  b6 <- vec[7]
  k1 <-vec[8]
  k2 <- vec[9]
  vtm <-midas(r,k1,k2)[[1]]
  vtm_D <- vtm*DUMMY_LAG
  mu <- b0[1] + b1*vtm + b2*x[,1] + b3*x[,2] + b4*vtm_D + b5*x[,3] + b6*x[,4]
  logl <- - (1/2)*sum(log(vtm)) - (1/2)*sum((y - mu)^2/vtm)
  return(-logl)
}

##Initial values of the parameters
k1 = k1_it
k2 = k2_it

midasvar <- midas(r, k1,k2)[[1]]
wts <- 1/midasvar
midasvar_D <- midasvar*DUMMY_LAG
mod_wls <- lm(QERET ~ midasvar+LPE_LAG+INFL_LAG + midasvar_D + LPEGD_LAG + INFLGD_LAG , data= full_q, weights=wts)
zeta0 <- c(coef(mod_wls),k1,k2) 

##optimize
er<-optim(zeta0,log_lik,x=X,y=Y, r=r,hessian=TRUE)
beta <- er$par
beta
k1_opt <- er$par[8]
k2_opt <- er$par[9]
k1_opt;k2_opt

##Obtain Midas Vt using the optimal k1 and k2 values
vtmidas_opt <- midas(r,k1 = k1_opt, k2 = k2_opt)[[1]]
full_q$vtmidas_opt <- as.numeric(vtmidas_opt)

##weighted least square
wts <- 1/vtmidas_opt
vt_wls <- lm(QERET ~ vtmidas_opt+LPE_LAG+INFL_LAG + midasvar_D + LPEGD_LAG + INFLGD_LAG , data= full_q, weights=wts)
summary(vt_wls)

##SE using the sandwich formula
vtm <- vtmidas_opt
vtm_D <- vtmidas_opt * DUMMY_LAG
mu <- er$par[1] + er$par[2]*vtm + er$par[3]*X[,1] + er$par[4]*X[,2] + er$par[5]*vtm_D + er$par[6]*X[,3] + er$par[7]*X[,4]

##step 1: create a vector of individual likelihood for each i obs.
log_lik_vec <- function(vec,u,y,v) {
  b0 <- vec[1]
  b1 <- vec[2]
  b2 <- vec[3]
  b3 <- vec[4]
  b4 <- vec[5]
  b5 <- vec[6]
  b6 <- vec[7]
  k1 <- vec[8]
  k2 <- vec[9]
  vtm <-midas(v,k1,k2)[[1]]
  vtm_D <- vtm*DUMMY_LAG
  mu <- b0[1] + b1*vtm + b2*u[,1] + b3*u[,2] + b4*vtm_D + b5*u[,3] + b6*u[,4]
  logl_vec <- -(1/2)*log(vtm) - (1/2)*((y - mu)^2/vtm)
  return(logl_vec)
}

##step 2: Calculate the m by n numerical approximation of the gradient of a real m-vector valued function with n-vector argument.
p = 7
G <- jacobian(log_lik_vec, er$par, u=X, y = Y, v=r)
Omega <- (t(G) %*% G)/n

v_cov1 <- (1/n)* solve(er$hessian[1:p,1:p]/n) %*% Omega[1:p,1:p]%*% t(solve(er$hessian[1:p,1:p]/n))
sand_se1 <- sqrt(abs(diag(v_cov1[1:p,1:p])))
summary(vt_wls); sand_se1

se <- sand_se1
##t-stat
est_coefs <- beta[1:p]
tvals <- est_coefs/se
tvals

##p-values
p0 <- 2*pt(-abs(tvals[1]), df=dim(full_q)[1]-p)
p1 <- 2*pt(-abs(tvals[2]), df=dim(full_q)[1]-p)
p2 <- 2*pt(-abs(tvals[3]), df=dim(full_q)[1]-p)
p3 <- 2*pt(-abs(tvals[4]), df=dim(full_q)[1]-p)
p4 <- 2*pt(-abs(tvals[5]), df=dim(full_q)[1]-p)
p5 <- 2*pt(-abs(tvals[6]), df=dim(full_q)[1]-p)
p6 <- 2*pt(-abs(tvals[7]), df=dim(full_q)[1]-p)
p0; p1; p2;p3;p4;p5;p6
