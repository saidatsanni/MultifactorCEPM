## =============================================================================
## Variable selection tables: Tables 2 and 3 
## =============================================================================
rm(list = ls())
pkgs <- c("readxl", "leaps", "writexl")
install.packages(setdiff(pkgs, rownames(installed.packages())))
invisible(lapply(pkgs, library, character.only = TRUE))

## ---- Settings ---------------------------------------------------------------
data_file <- "./Datasets/qrtly_data_2020.xlsx"
win_start <- 100                                 
nboot     <- 1000; boot_seed <- 10
price_vars <- c("LPE", "LDY", "BM")              
freq_cols  <- c("SVAR", "LPE", "LDY", "BM", "PRICE", "INFL", "DFR")

## samples
samples <- list(
  "Full sample" = list(start = "1947-03", drop = c("UNEM", "CAY", "CC", "NOS")),
  "UNEM sample" = list(start = "1949-03", drop = c("CAY", "CC", "NOS")),
  "CAY sample"  = list(start = "1952-03", drop = c("CC", "NOS")),
  "CC sample"   = list(start = "1954-03", drop = "NOS"),
  "NOS sample"  = list(start = "1958-03", drop = character(0))
)
end_ym <- "2020-12"
all_preds <- c("SVAR", "LPE", "INFL", "NTIS", "LDP", "LDY", "DFY", "DFR", "TMS", "RREL",
               "BM", "LTR", "TBL", "IK", "UNEM", "CAY", "CC", "NOS")

## ---- Data -------------------------------------------------------------------
d <- read_excel(data_file)
d$Date_lag <- as.Date(d$Date_lag); d <- d[order(d$Date_lag), ]
ym <- format(d$Date_lag, "%Y-%m")
for (v in c("QERET", paste0(all_preds, "_LAG"))) d[[v]] <- as.numeric(as.character(d[[v]]))
names(d) <- sub("_LAG$", "", names(d))            

get_sample <- function(s) {
  keep <- ym >= s$start & ym <= end_ym
  as.data.frame(d[keep, c("QERET", setdiff(all_preds, s$drop))])
}
dat <- lapply(samples, get_sample)
period <- sapply(samples, function(s) sprintf("%sQ%d-2020Q4", substr(s$start, 1, 4),
                                              ceiling(as.integer(substr(s$start, 6, 7)) / 3)))

## ---- BIC best subset --------------------------------------------------------
bic_subset <- function(dd, y = "QERET", nbest = 1) {
  z  <- regsubsets(as.formula(paste(y, "~ .")), data = dd, nbest = nbest, nvmax = ncol(dd) - 1)
  sm <- summary(z)
  names(which(sm$which[which.min(sm$bic), ]))[-1]
}

## =============================================================================
## Table A, Panel A: 25-year backward expanding-window selection frequencies
## =============================================================================
expanding_sel <- function(dd) {
  rev <- dd[nrow(dd):1, ]                        
  lapply(seq(win_start + 1, nrow(dd)), function(i) bic_subset(rev[1:i, ], nbest = 2))
}
freq_row <- function(sel) {
  n <- length(sel)
  f <- sapply(c("SVAR", "LPE", "LDY", "BM", "INFL", "DFR"), function(v) mean(sapply(sel, function(s) v %in% s)))
  f["PRICE"] <- mean(sapply(sel, function(s) any(price_vars %in% s)))
  f[freq_cols]
}
sel_lists <- lapply(dat, expanding_sel)
panelA <- t(sapply(sel_lists, freq_row))
panelA_fmt <- data.frame(Sample = names(samples), Period = period,
                         apply(panelA, 2, function(x) paste0(round(100 * x), "%")), check.names = FALSE)
for (v in setdiff(freq_cols, price_vars)) panelA_fmt[panelA[, v] < 0.5, v] <- ""
print(panelA_fmt, row.names = FALSE)

## =============================================================================
## Table A, Panel B: best-subset selection on each sample's full period
## =============================================================================
full_sel <- lapply(dat, bic_subset)
selB_vars <- unique(unlist(full_sel))
selB_vars <- selB_vars[order(match(selB_vars, all_preds))]
panelB <- data.frame(Sample = names(samples), Period = period,
                     t(sapply(seq_along(full_sel), function(i) ifelse(selB_vars %in% full_sel[[i]], "X", ""))),
                     check.names = FALSE)
names(panelB)[-(1:2)] <- selB_vars
for (i in seq_along(samples)) for (v in intersect(selB_vars, samples[[i]]$drop)) panelB[i, v] <- "-"
print(panelB, row.names = FALSE)

## =============================================================================
## Table B: bootstrap BIC selection frequencies, CAY sample
## =============================================================================
cay <- dat[["CAY sample"]]
boot_bic <- function(fit, dd) {
  set.seed(boot_seed)
  e <- resid(fit); fv <- fitted(fit); n <- length(e)
  X <- dd[, setdiff(names(dd), "QERET")]
  sel <- lapply(seq_len(nboot), function(b) {
    bd <- cbind(QERET_SIM = fv + sample(e, n, replace = TRUE), X)
    bic_subset(bd, y = "QERET_SIM", nbest = 2)
  })
  tab <- sort(table(unlist(sel)), decreasing = TRUE)
  data.frame(Variable = names(tab), Frequency = as.integer(tab),
             Proportion = paste0(round(100 * as.integer(tab) / nboot), "%"), row.names = NULL)
}
bootA <- boot_bic(lm(QERET ~ SVAR + LPE + INFL, data = cay), cay)   # return predictability
bootB <- boot_bic(lm(QERET ~ 1, data = cay), cay)                   # no predictability
n_show <- max(nrow(bootA), nrow(bootB))
pad <- function(x) { x[seq_len(n_show), ] |> (\(z) { z[is.na(z)] <- ""; z })() }
tableB <- cbind(Rank = seq_len(n_show), pad(bootA), pad(bootB))
names(tableB) <- c("Rank", "A_Variable", "A_Frequency", "A_Proportion", "B_Variable", "B_Frequency", "B_Proportion")
print(tableB, row.names = FALSE)

## ---- Export -----------------------------------------------------------------
dir.create("Output", showWarnings = FALSE)
write_xlsx(list(selection_freq = panelA_fmt, best_subset = panelB, bootstrap_bic = tableB),
           "output/varsel_tables.xlsx")

tex_rows <- function(df) apply(df, 1, function(r) paste(paste(r, collapse = " & "), "\\\\"))
texA <- c("\\begin{tabular}{llrrrrrrr}", "\\toprule",
          sprintf("\\multicolumn{%d}{c}{Panel A: Selection frequencies in 25-year backward expanding-window estimations} \\\\ \\midrule", ncol(panelA_fmt)),
          " & Sample period & SVAR & LPE & LDY & BM & PRICE (INDICATOR) & INFL & DFR \\\\ \\midrule",
          tex_rows(panelA_fmt), "\\midrule",
          sprintf("\\multicolumn{%d}{c}{Panel B: Best-subset selections using each specification's full available period} \\\\ \\midrule", ncol(panelA_fmt)),
          paste(" & Sample period &", paste(selB_vars, collapse = " & "), "\\\\ \\midrule"),
          tex_rows(panelB), "\\bottomrule", "\\end{tabular}")
writeLines(texA, "output/table_selection.tex")

texB <- c("\\begin{tabular}{rlrrlrr}", "\\toprule",
          " & \\multicolumn{3}{c}{Panel A: Return Predictability} & \\multicolumn{3}{c}{Panel B: No Return Predictability} \\\\",
          " & Selected variables & Frequency & Proportion & Selected variables & Frequency & Proportion \\\\ \\midrule",
          tex_rows(tableB), "\\bottomrule", "\\end{tabular}")
writeLines(texB, "output/table_bootstrap.tex")
