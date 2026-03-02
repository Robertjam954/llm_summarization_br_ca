# Before we start, if you don't have deepchecks installed yet, run:
import sys


# or install using pip from your python environment

from deepchecks.tabular import datasets

# load data
dirty_df = pd.read_excel(
    "data/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx")

from deepchecks.tabular.suites import data_integrity

# Run Suite:
integ_suite = data_integrity()
suite_result = integ_suite.run(ds)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result.show()