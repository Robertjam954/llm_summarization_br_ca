import streamlit as st
from summarizer import LuciSummarizer
import os
from dotenv import load_dotenv
from analyzer import EvaluationMetrics
from visualization import ScoreVisualizer

load_dotenv()


class LuciSummarizationApp:
    def __init__(self):
        self.model_options = [
            "GPT-3",
            "GPT-4",
            "Claude-3",
            "Llama",
            "Cohere",
            "Luci-FT-AM",
        ]

    def render_sidebar(self):
        st.sidebar.header("Upload Files")
        document_upload = st.sidebar.file_uploader("Upload Document", type=["pdf"])
        summary_upload = st.sidebar.file_uploader(
            "Upload 'Golden Summary'", type=["pdf"]
        )

        return document_upload, summary_upload

    def select_models(self):
        selected_models = st.sidebar.multiselect(
            "Choose models to benchmark:", self.model_options
        )
        return selected_models

    def execution(self):
        run_button = st.sidebar.button("Run Benchmark")
        return run_button

    def select_task(self):
        task = st.sidebar.radio("Choose Task:", ["Summarize", "Write Code"])
        return task

    def run(self):
        st.title("LLM Benchmarking Tool")
        st.subheader("Summarization & Code Generation")
        logo = "../assets/color_lucidate.png"
        st.image(logo, width=120)

        document_upload, summary_upload = self.render_sidebar()
        selected_models = self.select_models()
        task = self.select_task()
        run_button = self.execution()

        if run_button:
            if document_upload is not None and summary_upload is not None:
                st.success("Files uploaded successfully!")
                st.write(
                    "Selected models for benchmarking:", ", ".join(selected_models)
                )

                # Convert the uploaded files to bytes for processing
                document_bytes = document_upload.getvalue()
                summary_bytes = summary_upload.getvalue()

                # Create an instance of the summarizer
                # summarizer = LuciSummarizer(document_bytes, summary_bytes)
                summarizer = LuciSummarizer(document_upload, summary_upload)

                # Process the documents and store text in vector stores (this functionality has to be implemented)
                document_text, summary_text_golden = summarizer.process_documents()

                # Here you can now use the text and vector stores to perform summarization
                # and comparison which will be added in future development steps.

                # For now, we can display the extracted text to confirm the process works
                # st.subheader("Extracted Document Text:")
                # st.write(document_text)  # Assuming we have a property to get the processed text
                #
                if task == "Summarize":
                    st.subheader("'Golden Source' Target Summary Text:")
                else:
                    st.subheader("'Golden Source' Target Code:")
                with st.expander("Click for 'Golden Source' text"):
                    st.write(
                        summary_text_golden
                    )  # Assuming we have a property to get the processed text

                all_scores = {}
                for model in selected_models:
                    summarizer = LuciSummarizer(document_upload, summary_upload)
                    document_text, summary_text_golden = summarizer.process_documents()

                    generated_summary = summarizer.summarise_documents(model, task)
                    # print(f"Generated summary for {model}: {generated_summary}")
                    st.subheader(model)
                    with st.expander(f"Click to see {model} generated summary"):
                        st.write(generated_summary)

                    evaluation_metrics = EvaluationMetrics()

                    # Assuming 'generated_summary' and 'summary_text_golden' are available from earlier steps
                    if task == "Summarize":
                        bleu_score = evaluation_metrics.calculate_bleu_score(
                            summary_text_golden, generated_summary
                        )
                        rouge_scores = evaluation_metrics.calculate_rouge_scores(
                            summary_text_golden, generated_summary
                        )
                        semantic_similarity = (
                            evaluation_metrics.calculate_semantic_similarity(
                                summary_text_golden, generated_summary
                            )
                        )
                        all_scores[model] = {
                            "bleu_score": bleu_score,
                            "rouge_scores": rouge_scores,
                            "semantic_similarity": semantic_similarity,
                        }

                    else:
                        bleu_score = evaluation_metrics.calculate_code_bleu_score(
                            summary_text_golden, generated_summary
                        )
                        all_scores[model] = {
                            "codebleu": bleu_score["codebleu"],
                            "ngram_match_score": bleu_score["ngram_match_score"],
                            "weighted_ngram_match_score": bleu_score[
                                "weighted_ngram_match_score"
                            ],
                            "syntax_match_score": bleu_score["syntax_match_score"],
                            "dataflow_match_score": bleu_score["dataflow_match_score"],
                        }

                    # Display the scores
                    # st.subheader("Evaluation Metrics:")
                    # st.write(f"BLEU Score: {bleu_score}")
                    # st.write("ROUGE Scores:", rouge_scores)
                    # st.write(f"Semantic Similarity (Cosine): {semantic_similarity}")

                visualizer = ScoreVisualizer(all_scores, task)
                visualizer.plot_scores()
                visualizer.render_score_table()
                visualizer.plot_radar_chart()


if __name__ == "__main__":
    app = LuciSummarizationApp()
    app.run()
