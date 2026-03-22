library(dplyr)
library(tidyverse)
library(readxl)

DATA_PRIVATE_DIR <- Sys.getenv("DATA_PRIVATE_DIR", "/Users/robertjames/data_private")
PROJECT_ROOT     <- Sys.getenv("PROJECT_ROOT",     "/Users/robertjames/Documents/GitHub/llm_summarization_br_ca")

df1 <- read_excel(file.path(DATA_PRIVATE_DIR, "raw", "LLM_Summary_Validation_Table_v2.xlsx"))
df2 <- read_excel(file.path(DATA_PRIVATE_DIR, "raw", "ai_summary_data_collection_sheet.xlsx"))

df2 |>
  select(
    "surgeon",
    "mrn",
    "patient_initials",
    "tumor_invasion",
    "complex_case_status"
  ) -> df3

df1 <- df1 |>
  mutate(mrn = as.character(mrn))

merged_llm_summary_validation_datasheet <- full_join(df3, df1, by = c("mrn"))
write.csv(merged_llm_summary_validation_datasheet,
          file = file.path(PROJECT_ROOT, "data", "processed", "merged_llm_summary_validation_datasheet.csv"),
          row.names = FALSE)

