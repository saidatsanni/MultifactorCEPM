###############################################################################
#  Figures 1 and 2; Table 6                                                   #
#  Drawn from output/oos_forecasts.xlsx written by oos_forecasts.R            #                                                #
###############################################################################

library(readxl)

file <- "output/oos_forecasts.xlsx"
read.sheet <- function(sheet) { d <- as.data.frame(read_excel(file, sheet = sheet)); d$Date <- as.Date(d$Date); d }

r25.3   <- read.sheet("data2025_pred3")
r25.all <- read.sheet("data2025_predAll")
r20.3   <- read.sheet("data2020_pred3")
f20.all <- read.sheet("data2020_predAll_pred")

###############################################################################
#  Figure 1: out-of-sample R^2                                                #
###############################################################################

cut.years <- 10
last.start <- function(d) seq(max(d$Date), by = paste0("-", cut.years, " years"), length.out = 2)[2]

keep25 <- r25.3$Date <= last.start(r25.3)
keep20 <- r20.3$Date <= last.start(r20.3)

pdf("output/fig_1_oos_r2.pdf", width = 7, height = 4.5)
par(mar = c(3, 4.5, 1, 1))
plot(r25.3$Date[keep25], r25.3$OOS_MULT[keep25], type = "n", ylim = c(-0.4, 0.4), las = 1,
     xlab = "", ylab = expression("Out-of-sample " * R^2), xaxt = "n")
axis.Date(1, at = seq(as.Date("1965-01-01"), max(r25.3$Date), by = "10 years"), format = "%Y")
abline(h = 0, col = "grey70")
lines(r25.3$Date[keep25],   r25.3$OOS_MULT[keep25],     lwd = 3, lty = 1)
lines(r25.all$Date[keep25], r25.all$OOS_LINSEL[keep25], lwd = 1, lty = 2)
lines(r25.all$Date[keep25], r25.all$OOS_COMB[keep25],   lwd = 1, lty = 1)
lines(r20.3$Date[keep20],   r20.3$OOS_MULT[keep20],     lwd = 3, lty = 3)
legend("topleft", bty = "n", cex = 0.8, lty = c(3, 1, 1, 2), lwd = c(3, 3, 1, 1),
       legend = c("Multifactor model, sample ending 2020Q4",
                  "Multifactor model, sample ending 2025Q4",
                  "Combination forecast, sample ending 2025Q4",
                  "Recursive BIC best-subset model, sample ending 2025Q4"))

invisible(dev.off())

###############################################################################
#  Figure 2: cumulative difference in squared forecast errors                 #
###############################################################################

starts <- as.Date(c("1965-03-01", "1976-03-01", "2000-03-01", "2010-03-01"))
quarter.label <- function(d) paste0(format(d, "%Y"), "Q", (as.numeric(format(d, "%m")) - 1) %/% 3 + 1)

pdf("output/fig_2_cumulative_sse.pdf", width = 9, height = 7)
par(mfrow = c(2, 2), mar = c(3, 4.5, 2.5, 1))
for (i in 1:4) {
  d   <- f20.all[f20.all$Date >= starts[i], ]
  cum <- cumsum(d$fcsq_comb - d$fcsq_mult)       # positive: multifactor model has the lower cumulative SSE
  plot(d$Date, cum, type = "l", lwd = 2, las = 1, xlab = "", cex.main = 0.9,
       ylab = if (i %in% c(1, 3)) "Cumulative SSE difference" else "",
       main = paste0(LETTERS[i], ": ", quarter.label(d$Date[1]), "-", quarter.label(max(d$Date))))
}
invisible(dev.off())

cat("Figures written to output/fig_1_oos_r2.pdf and output/fig_2_cumulative_sse.pdf\n")


###############################################################################
#  Table 6: out-of-sample R^2 by forecasting method                           #
###############################################################################
first.row <- function(sheet) as.data.frame(read_excel(file, sheet = sheet))[1, ]

methods <- c("Linear Regression"          = "OOS_LIN",
             "Generalized Additive Model" = "OOS_GAM",
             "RF1"  = "OOS_RF1",  "RF2"  = "OOS_RF2",
             "BRT1" = "OOS_BRT1", "BRT2" = "OOS_BRT2",
             "NN1"  = "OOS_NN1",  "NN2"  = "OOS_NN2",  "NN3" = "OOS_NN3",
             "PC"   = "OOS_PC",   "PC+"  = "OOS_PCPLUS",
             "PLS"  = "OOS_PLS",  "PLS+" = "OOS_PLSPLUS",
             "Combination Forecast" = "OOS_COMB")

columns <- c("data2020_predAll", "data2020_pred3", "data2025_predAll", "data2025_pred3")

table.6 <- matrix(NA, length(methods), length(columns),
                  dimnames = list(names(methods), columns))
for (j in columns) {
  r <- first.row(j)
  table.6[, j] <- as.numeric(r[methods])
}
print(round(table.6, 4))
write.csv(data.frame(Method = names(methods), round(table.6, 4)), "output/table_6.csv", row.names = FALSE)

# LaTeX
num <- function(x) gsub("-", "$-$", sprintf("%.4f", x), fixed = TRUE)
tex <- c("\\begin{table}[htbp]", "\\centering",
         "\\begin{tabular}{lcccc}", "\\hline\\hline",
         " & \\multicolumn{2}{c}{Panel A: 1965Q1--2020Q4} & \\multicolumn{2}{c}{Panel B: 1965Q1--2025Q4} \\\\",
         "\\cline{2-3} \\cline{4-5}",
         " & All & Selected & All & Selected \\\\",
         "Methods & Predictors & Predictors & Predictors & Predictors \\\\", "\\hline")
for (i in seq_len(nrow(table.6)))
  tex <- c(tex, sprintf("%s & %s \\\\", rownames(table.6)[i], paste(num(table.6[i, ]), collapse = " & ")))
tex <- c(tex, "\\hline\\hline", "\\end{tabular}", "\\end{table}")
writeLines(tex, "output/table_6.tex")

cat("Table written to output/table_6.tex and output/table_6.csv\n")
