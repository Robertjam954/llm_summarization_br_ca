library(dplyr)
library(tidyverse)
library(readxl)

df1 <- read_excel("C:/Users/jamesr4/OneDrive - Memorial Sloan Kettering Cancer Center/Documents/Research/Projects/moo/llm_summary/validaton/LLM_Summary_Validation_Table_v2.xlsx") 
df2 <- read_excel("Research/Projects/moo/llm_summary/data/ai_summary_data_collection_sheet.xlsx")

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
write.csv(merged_llm_summary_validation_datasheet, file = "merged_llm_summary_validation_datasheet.csv")

