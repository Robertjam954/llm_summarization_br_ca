# ============================================
#  SHAP analysis for XGBoost AFT model
#  Memorial Sloan Kettering | Goel Lab
# ============================================

# --- 1. Install + load required packages ---
if (!requireNamespace("SHAPforxgboost", quietly = TRUE))
  install.packages("SHAPforxgboost")

if (!requireNamespace("devtools", quietly = TRUE))
  install.packages("devtools")

if (!requireNamespace("xgboost", quietly = TRUE))
  install.packages("xgboost")

# For SHAPforxgboost (GitHub version)
if (!requireNamespace("SHAPforxgboost", quietly = TRUE)) {
  devtools::install_github("liuyanguu/SHAPforxgboost")
}

library(SHAPforxgboost)
library(xgboost)
library(dplyr)
library(gridExtra)

# --- 2. Load your dataset ---
# Adjust path if needed (use forward slashes for Windows)
data_path <- "C:/Users/jamesr4/OneDrive - Memorial Sloan Kettering Cancer Center/Documents/Research/Projects/genomics_brain_mets_genie_bpc/tcga_gene_alteration_dfs_os_xgb/merged_tcga_df.csv"

dataXY_df <- read.csv(data_path)

# --- 3. Define your survival outcome + features ---
# XGBoost AFT requires two survival columns: lower-bound (start) and upper-bound (end)
# Here we’ll assume “PFS_MONTHS” = time and “PFS_STATUS” = event (1 = event, 0 = censored)
# Edit these if your dataset uses different names.

time_col  <- "PFS_MONTHS"
event_col <- "PFS_STATUS"

# Create interval-censored survival labels
y_lower <- dataXY_df[[time_col]]
y_upper <- ifelse(dataXY_df[[event_col]] == 1, y_lower, Inf)  # right-censored cases get Inf

# --- 4. Prepare data for model ---
# Drop ID columns or other non-numeric fields
exclude_cols <- c(time_col, event_col, "patient_id", "mrn")
X_df <- dataXY_df[ , !(names(dataXY_df) %in% exclude_cols)]
X_df <- X_df[sapply(X_df, is.numeric)]  # only numeric features for XGBoost
X_mat <- as.matrix(X_df)

# --- 5. Train AFT model ---
params <- list(
  objective = "survival:aft",
  eval_metric = "aft-nloglik",
  aft_loss_distribution = "normal",
  aft_loss_distribution_scale = 1.0,
  eta = 0.03,
  max_depth = 6,
  subsample = 0.9,
  colsample_bytree = 0.9
)

dtrain <- xgb.DMatrix(X_mat, label_lower_bound = y_lower, label_upper_bound = y_upper)

set.seed(2025)
aft_mod <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  verbose = 1
)

# --- 6. Compute SHAP values ---
# SHAPforxgboost works with survival models as long as features are numeric
shap_values <- shap.values(xgb_model = aft_mod, X_train = X_mat)

# Rank features by mean |SHAP|
feature_ranking <- shap_values$mean_shap_score
print(head(feature_ranking, 15))

# --- 7. Prepare long-format SHAP data ---
shap_long <- shap.prep(xgb_model = aft_mod, X_train = X_mat)

# --- 8. SHAP Summary plots ---
pdf("shap_summary_plots.pdf", width = 9, height = 6)
shap.plot.summary(shap_long)
dev.off()

# Lighter version for large datasets
pdf("shap_summary_diluted.pdf", width = 9, height = 6)
shap.plot.summary(shap_long, x_bound = 1.5, dilute = 10)
dev.off()

# --- 9. SHAP Dependence plots ---
# Identify top features
top_features <- names(feature_ranking)[1:6]

# Dependence plots for top features
pdf("shap_dependence_top6.pdf", width = 10, height = 8)
fig_list <- lapply(top_features, shap.plot.dependence, data_long = shap_long, dilute = 5)
gridExtra::grid.arrange(grobs = fig_list, ncol = 2)
dev.off()

# Example: single-feature dependence with coloring
pdf("shap_dependence_colored.pdf", width = 7, height = 5)
shap.plot.dependence(data_long = shap_long,
                     x = top_features[1],
                     color_feature = top_features[2])
dev.off()

# --- 10. SHAP Interaction plots ---
# Warning: interaction calculation is very slow for many features
# Run only on subset or small data
message("Calculating SHAP interactions — may take a while on full dataset...")

try({
  shap_int <- shap.prep.interaction(xgb_mod = aft_mod, X_train = X_mat)
  pdf("shap_interaction_plot.pdf", width = 7, height = 5)
  shap.plot.dependence(
    data_long = shap_long,
    data_int = shap_int,
    x = top_features[1],
    y = top_features[2],
    color_feature = top_features[2]
  )
  dev.off()
}, silent = TRUE)

# --- 11. SHAP Force plots ---
# Show top 4 features, grouped into 6 clusters
plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 4, n_groups = 6)

pdf("shap_force_plots.pdf", width = 9, height = 6)
shap.plot.force_plot(plot_data, zoom_in_location = 300, y_parent_limit = c(-1, 1))
shap.plot.force_plot_bygroup(plot_data)
dev.off()

message("✅ All SHAP plots have been saved as PDFs in the working directory.")
