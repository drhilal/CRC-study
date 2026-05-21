

# Clear workspace
rm(list = ls())

################################################################################
# INSTALL AND LOAD REQUIRED PACKAGES
################################################################################

required_packages <- c(
  "tidyverse",      # Data manipulation
  "pROC",           # ROC analysis
  "readxl",         # Read Excel files
  "writexl",        # Write Excel files
  "VIM",            # KNN imputation
  "ggplot2",        # Visualization
  "reshape2",       # Data reshaping
  "RColorBrewer"    # Color palettes
)

# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
  install.packages(new_packages, dependencies = TRUE)
}

# Load packages quietly
invisible(lapply(required_packages, library, character.only = TRUE, quietly = TRUE))

# Set seed for reproducibility
set.seed(42)

################################################################################
# 1. LOAD DATA
################################################################################

cat("\n================================================================================\n")
cat("CRC METABOLOMICS - UNIVARIATE ANALYSIS 
cat("================================================================================\n\n")

cat("PREPROCESSING PIPELINE:\n")
cat("  1. KNN imputation (k=5)\n")
cat("  2. Log2 transformation\n")
cat("  3. Pareto scaling\n")
cat("  4. Mann-Whitney U test\n")
cat("  5. FDR correction (Benjamini-Hochberg)\n\n")



data_file <- "df__4_.xlsx"  # CHANGE THIS to your file path

if (!file.exists(data_file)) {
  cat("ERROR: Data file not found:", data_file, "\n")
  cat("Please provide an Excel file with:\n")
  cat("  - Metabolite columns (numeric values)\n")
  cat("  - 'Group' or 'Factors' column indicating NC (control) or CC (CRC)\n\n")
  stop("Data file not found")
}

# Read data
cat("Loading data from:", data_file, "\n")
df_raw <- read_excel(data_file)

# Extract group information
if ("Group" %in% colnames(df_raw)) {
  group_col <- df_raw$Group
} else if ("Factors" %in% colnames(df_raw)) {
  # Extract from Factors column (e.g., "Group:NC")
  group_col <- gsub(".*Group:(\\w+).*", "\\1", df_raw$Factors)
} else {
  stop("ERROR: No 'Group' or 'Factors' column found in data")
}

# Convert to binary: NC=0, CC=1
y <- ifelse(group_col == "NC", 0, 1)

# Get metabolite columns (exclude non-metabolite columns)
exclude_cols <- c("Samples", "Factors", "Group")
metabolite_cols <- setdiff(colnames(df_raw), exclude_cols)

# Extract metabolite data
X_raw <- df_raw[, metabolite_cols]

# Convert to numeric (in case some columns are character)
X_raw <- as.data.frame(lapply(X_raw, as.numeric))
colnames(X_raw) <- metabolite_cols

cat("\nData loaded successfully!\n")
cat("  Total samples:", nrow(X_raw), "\n")
cat("  Controls (NC):", sum(y == 0), "\n")
cat("  CRC cases (CC):", sum(y == 1), "\n")
cat("  Metabolites:", ncol(X_raw), "\n\n")

################################################################################
# 2. PREPROCESSING PIPELINE
################################################################################

cat("================================================================================\n")
cat("STEP 1: MISSING VALUE HANDLING\n")
cat("================================================================================\n")

# Check missing values
missing_pct <- colSums(is.na(X_raw)) / nrow(X_raw) * 100
cat("Metabolites with missing values:", sum(missing_pct > 0), "\n")

# Filter out metabolites with >50% missing values
keep_metabolites <- missing_pct <= 50
X_filtered <- X_raw[, keep_metabolites]
metabolite_cols_filtered <- metabolite_cols[keep_metabolites]

cat("Metabolites retained:", length(metabolite_cols_filtered), "\n")
cat("Metabolites excluded:", sum(!keep_metabolites), "\n")

# KNN imputation for remaining missing values
if (any(is.na(X_filtered))) {
  cat("\nPerforming KNN imputation (k=5)...\n")
  X_imputed <- kNN(X_filtered, k = 5, imp_var = FALSE)
  cat("Imputation complete.\n")
} else {
  X_imputed <- X_filtered
  cat("\nNo missing values - imputation not needed.\n")
}


# Log2 transformation
X_log2 <- log2(X_imputed + 1)  # Add 1 to avoid log(0)



# Pareto scaling function
pareto_scale <- function(x) {
  mean_x <- mean(x, na.rm = TRUE)
  sd_x <- sd(x, na.rm = TRUE)
  sqrt_sd <- sqrt(sd_x)
  
  # Avoid division by zero
  if (sqrt_sd == 0 || is.na(sqrt_sd)) sqrt_sd <- 1
  
  return((x - mean_x) / sqrt_sd)
}

# Apply Pareto scaling
X_scaled <- as.data.frame(lapply(X_log2, pareto_scale))
colnames(X_scaled) <- metabolite_cols_filtered

cat("Pareto scaling complete.\n")

# Save processed data for ML analysis
processed_data <- X_scaled
processed_data$Group <- y
write.csv(processed_data, "metabolomics_data_processed.csv", row.names = FALSE)
cat("Processed data saved: metabolomics_data_processed.csv\n")

################################################################################
# 3. NORMALITY TESTING
################################################################################

cat("\n================================================================================\n")
cat("STEP 4: NORMALITY TESTING (SHAPIRO-WILK)\n")
cat("================================================================================\n")

normality_results <- data.frame(
  Metabolite = character(),
  Control_p = numeric(),
  CRC_p = numeric(),
  Control_Normal = logical(),
  CRC_Normal = logical(),
  stringsAsFactors = FALSE
)

for (feature in metabolite_cols_filtered) {
  # Use RAW data for normality test (before scaling)
  control_vals <- X_log2[[feature]][y == 0]
  crc_vals <- X_log2[[feature]][y == 1]
  
  # Shapiro-Wilk test
  shapiro_control <- shapiro.test(control_vals)
  shapiro_crc <- shapiro.test(crc_vals)
  
  normality_results <- rbind(normality_results, data.frame(
    Metabolite = feature,
    Control_p = shapiro_control$p.value,
    CRC_p = shapiro_crc$p.value,
    Control_Normal = shapiro_control$p.value > 0.05,
    CRC_Normal = shapiro_crc$p.value > 0.05
  ))
}

normal_count <- sum(normality_results$Control_Normal & normality_results$CRC_Normal)
cat("Normally distributed metabolites:", normal_count, "out of", nrow(normality_results), "\n")
cat("Non-parametric test (Mann-Whitney U) will be used for all metabolites.\n")

################################################################################
# 4. UNIVARIATE STATISTICAL ANALYSIS
################################################################################

cat("\n================================================================================\n")
cat("STEP 5: MANN-WHITNEY U TEST\n")
cat("================================================================================\n")

results <- data.frame(
  Metabolite = character(),
  Median_Control = numeric(),
  Median_CRC = numeric(),
  FC = numeric(),
  Log2FC = numeric(),
  p_value = numeric(),
  neg_log10_p = numeric(),
  Regulation = character(),
  stringsAsFactors = FALSE
)

for (feature in metabolite_cols_filtered) {
  # Mann-Whitney U test on SCALED data
  control_scaled <- X_scaled[[feature]][y == 0]
  crc_scaled <- X_scaled[[feature]][y == 1]
  
  mw_test <- wilcox.test(crc_scaled, control_scaled, 
                         alternative = "two.sided", 
                         exact = FALSE)
  
  # Calculate FC from IMPUTED (not scaled) data
  control_raw <- X_imputed[[feature]][y == 0]
  crc_raw <- X_imputed[[feature]][y == 1]
  
  median_control <- median(control_raw, na.rm = TRUE)
  median_crc <- median(crc_raw, na.rm = TRUE)
  
  # Fold change
  if (median_control == 0) {
    fc <- ifelse(median_crc > 0, Inf, 1.0)
    log2fc <- ifelse(median_crc > 0, Inf, 0.0)
  } else {
    fc <- median_crc / median_control
    log2fc <- log2(fc)
  }
  
  # Regulation direction
  regulation <- ifelse(fc > 1, "Up", "Down")
  
  # -log10(p)
  neg_log10_p <- ifelse(mw_test$p.value > 0, -log10(mw_test$p.value), Inf)
  
  results <- rbind(results, data.frame(
    Metabolite = feature,
    Median_Control = median_control,
    Median_CRC = median_crc,
    FC = fc,
    Log2FC = log2fc,
    p_value = mw_test$p.value,
    neg_log10_p = neg_log10_p,
    Regulation = regulation
  ))
}

cat("Mann-Whitney U test completed for", nrow(results), "metabolites.\n")

################################################################################
# 5. FDR CORRECTION
################################################################################

cat("\n================================================================================\n")
cat("STEP 6: FDR CORRECTION (BENJAMINI-HOCHBERG)\n")
cat("================================================================================\n")

# FDR correction
results$Adj_p <- p.adjust(results$p_value, method = "fdr")
results$Significant <- results$Adj_p < 0.05

n_sig <- sum(results$Significant)
n_up <- sum(results$Significant & results$Regulation == "Up")
n_down <- sum(results$Significant & results$Regulation == "Down")

cat("Significant metabolites (FDR < 0.05):", n_sig, "\n")
cat("  Upregulated:", n_up, "\n")
cat("  Downregulated:", n_down, "\n")

################################################################################
# 6. ROC ANALYSIS
################################################################################

cat("\n================================================================================\n")
cat("STEP 7: ROC ANALYSIS WITH BOOTSTRAP CI\n")
cat("================================================================================\n")

roc_results <- data.frame(
  Metabolite = character(),
  AUC = numeric(),
  CI_lower = numeric(),
  CI_upper = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  Threshold = numeric(),
  stringsAsFactors = FALSE
)

for (feature in metabolite_cols_filtered) {
  # Use IMPUTED data for ROC (not scaled, not transformed)
  metabolite_values <- X_imputed[[feature]]
  
  # ROC curve
  roc_obj <- roc(y, metabolite_values, 
                 levels = c(0, 1), 
                 direction = "<",
                 quiet = TRUE)
  
  # AUC
  auc_value <- as.numeric(auc(roc_obj))
  
  # Bootstrap CI
  ci_obj <- ci.auc(roc_obj, method = "bootstrap", boot.n = 2000, quiet = TRUE)
  ci_lower <- ci_obj[1]
  ci_upper <- ci_obj[3]
  
  # Optimal cutoff (Youden index)
  coords_obj <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"),
                       best.method = "youden", quiet = TRUE)
  
  roc_results <- rbind(roc_results, data.frame(
    Metabolite = feature,
    AUC = auc_value,
    CI_lower = ci_lower,
    CI_upper = ci_upper,
    Sensitivity = coords_obj$sensitivity,
    Specificity = coords_obj$specificity,
    Threshold = coords_obj$threshold
  ))
}

cat("ROC analysis completed.\n")

################################################################################
# 7. COMBINE RESULTS
################################################################################

cat("\n================================================================================\n")
cat("STEP 8: FINAL RESULTS TABLE\n")
cat("================================================================================\n")

# Merge results
final_results <- merge(results, roc_results, by = "Metabolite")

# Add CI as string
final_results$CI_95 <- sprintf("%.3f-%.3f", final_results$CI_lower, final_results$CI_upper)

# Sort by adjusted p-value
final_results <- final_results[order(final_results$Adj_p), ]

# Select columns for output
output_cols <- c("Metabolite", "FC", "Log2FC", "neg_log10_p", "Adj_p", 
                 "Regulation", "AUC", "CI_95", "Sensitivity", "Specificity")
final_table <- final_results[, output_cols]

# Round numerical columns
final_table$FC <- round(final_table$FC, 3)
final_table$Log2FC <- round(final_table$Log2FC, 3)
final_table$neg_log10_p <- round(final_table$neg_log10_p, 3)
final_table$Adj_p <- ifelse(final_table$Adj_p < 0.001, "<0.001", 
                             sprintf("%.3f", final_table$Adj_p))
final_table$AUC <- round(final_table$AUC, 3)
final_table$Sensitivity <- round(final_table$Sensitivity, 3)
final_table$Specificity <- round(final_table$Specificity, 3)

# Save results
write.csv(final_table, "Table1_Univariate_Results.csv", row.names = FALSE)
write_xlsx(list(Table1 = final_table), "Table1_Univariate_Results.xlsx")

cat("\nResults saved:\n")
cat("  - Table1_Univariate_Results.csv\n")
cat("  - Table1_Univariate_Results.xlsx\n\n")

# Print Table 1
cat("TABLE 1 - SIGNIFICANT METABOLITES (FDR < 0.05):\n")
cat("================================================================================\n")
significant_metabolites <- final_table[final_results$Significant, ]
print(significant_metabolites, row.names = FALSE)

cat("\n================================================================================\n")
cat("SUMMARY\n")
