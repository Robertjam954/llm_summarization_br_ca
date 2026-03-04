import plotly.graph_objs as go
import streamlit as st
import numpy as np
import pandas as pd
import os


class ScoreVisualizer:
    def __init__(self, all_scores, task):
        self.all_scores = all_scores
        self.task = task
        self.color_map = {
            "GPT-3": "red",
            "GPT-4": "purple",
            "Cohere": "blue",
            "Llama": "green",
            "Claude-3": "orange",
            "Luci-FT-AM": "cyan",
        }

    def plot_scores(self):
        data = []
        BAR_WIDTH = 0.9 / (len(self.all_scores) + 1)

        for model, scores in self.all_scores.items():
            model_color = self.color_map.get(model, "gray")
            showlegend = True

            if self.task == "Summarize":
                bleu_trace = go.Bar(
                    name=model,
                    x=["BLEU"],
                    y=[scores["bleu_score"]],
                    marker_color=model_color,
                    width=BAR_WIDTH,
                    legendgroup=model,
                    showlegend=showlegend,
                )
                data.append(bleu_trace)

                showlegend = False

                for rouge_key in ["rouge-1", "rouge-2", "rouge-l"]:
                    rouge_trace = go.Bar(
                        name=model,
                        x=[rouge_key.upper()],
                        y=[scores["rouge_scores"][rouge_key]["f"]],
                        marker_color=model_color,
                        width=BAR_WIDTH,
                        legendgroup=model,
                        showlegend=showlegend,
                    )
                    data.append(rouge_trace)

                semantic_trace = go.Bar(
                    name=model,
                    x=["Semantic Similarity"],
                    y=[scores["semantic_similarity"]],
                    marker_color=model_color,
                    width=BAR_WIDTH,
                    legendgroup=model,
                    showlegend=showlegend,
                )
                data.append(semantic_trace)
            else:
                for metric in [
                    "codebleu",
                    "ngram_match_score",
                    "weighted_ngram_match_score",
                    "syntax_match_score",
                    "dataflow_match_score",
                ]:
                    trace = go.Bar(
                        name=model,
                        x=[metric.replace("_", " ").title()],
                        y=[scores[metric]],
                        marker_color=model_color,
                        width=BAR_WIDTH,
                        legendgroup=model,
                        showlegend=showlegend,
                    )
                    data.append(trace)
                    showlegend = False

        layout = go.Layout(
            title="Code Evaluation Scores"
            if self.task == "Write code"
            else "Summary Evaluation Scores",
            barmode="group",
            yaxis=dict(title="Score"),
            xaxis=dict(title="Metric"),
            legend=dict(groupclick="toggleitem"),
        )

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)

    def plot_radar_chart(self):
        if self.task == "Summarize":
            categories = [
                "BLEU",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "Semantic Similarity",
            ]
        else:
            categories = [
                "CodeBLEU",
                "N-gram Match",
                "Weighted N-gram Match",
                "Syntax Match",
                "Dataflow Match",
            ]

        data = []
        for model, scores in self.all_scores.items():
            if self.task == "Summarize":
                values = [
                    scores["bleu_score"],
                    scores["rouge_scores"]["rouge-1"]["f"],
                    scores["rouge_scores"]["rouge-2"]["f"],
                    scores["rouge_scores"]["rouge-l"]["f"],
                    scores["semantic_similarity"],
                ]
            else:
                values = [
                    scores["codebleu"],
                    scores["ngram_match_score"],
                    scores["weighted_ngram_match_score"],
                    scores["syntax_match_score"],
                    scores["dataflow_match_score"],
                ]
            data.append(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    line=dict(width=3),
                    fillcolor="rgba(0,0,0,0)",
                    opacity=0.7,
                    name=model,
                    marker=dict(color=self.color_map.get(model, "gray")),
                )
            )

        layout = go.Layout(
            title="Model Performance Metrics",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color="white"),
                bgcolor="rgba(0,0,0,0)",
                angularaxis=dict(color="white"),
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(font=dict(color="white")),
        )

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)

    def render_score_table(self):
        table_data = {}
        for model, metrics in self.all_scores.items():
            table_data[model] = {"Model": model}
            if self.task == "Summarize":
                for metric, score in metrics.items():
                    if metric == "rouge_scores":
                        for rouge_metric, rouge_scores in score.items():
                            table_data[model][f"{rouge_metric.upper()} F1"] = round(
                                rouge_scores["f"], 3
                            )
                    else:
                        metric_name = (
                            "BLEU" if metric == "bleu_score" else "Semantic Similarity"
                        )
                        table_data[model][metric_name] = round(score, 3)
            else:
                for metric, score in metrics.items():
                    table_data[model][metric.replace("_", " ").title()] = round(
                        score, 3
                    )

        score_df = pd.DataFrame.from_dict(table_data, orient="index").reset_index(
            drop=True
        )

        def highlight_max(s, props=""):
            return np.where(s == np.nanmax(s.to_numpy()), props, "")

        styled_df = score_df.style.apply(
            highlight_max,
            props="background-color:yellow;",
            axis=0,
            subset=pd.IndexSlice[:, "BLEU":"Semantic Similarity"]
            if self.task == "Summarize"
            else pd.IndexSlice[:, "Codebleu":"Dataflow Match Score"],
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


class ScoreVisualizerOld:
    def __init__(self, all_scores):
        self.all_scores = all_scores
        self.color_map = {
            "GPT-3": "red",
            "GPT-4": "purple",
            "Cohere": "blue",
            "Llama": "green",
            "Claude-3": "orange",
            "Luci-FT-AM": "cyan",
        }

    def plot_scores(self):
        data = []
        BAR_WIDTH = 0.9 / (len(self.all_scores) + 1)
        print(len(self.all_scores))

        for model, scores in self.all_scores.items():
            # Use the same color for all traces of the same model
            model_color = self.color_map.get(
                model, "gray"
            )  # Default to gray if model not found

            # Initialize showlegend as True for the first metric (BLEU) only
            showlegend = True

            # Create a bar for the BLEU score
            bleu_trace = go.Bar(
                name=model,
                x=["BLEU"],
                y=[scores["bleu_score"]],
                marker_color=model_color,
                width=BAR_WIDTH,
                legendgroup=model,
                showlegend=showlegend,
            )
            data.append(bleu_trace)

            # After the first trace, set showlegend to False for subsequent metrics
            showlegend = False

            # Create bars for each ROUGE score
            for rouge_key in ["rouge-1", "rouge-2", "rouge-l"]:
                rouge_trace = go.Bar(
                    name=model,
                    x=[rouge_key.upper()],
                    y=[scores["rouge_scores"][rouge_key]["f"]],
                    marker_color=model_color,
                    width=BAR_WIDTH,
                    legendgroup=model,
                    showlegend=showlegend,
                )
                data.append(rouge_trace)

            # Create a bar for the Semantic Similarity
            semantic_trace = go.Bar(
                name=model,
                x=["Semantic Similarity"],
                y=[scores["semantic_similarity"]],
                marker_color=model_color,
                width=BAR_WIDTH,
                legendgroup=model,
                showlegend=showlegend,
            )
            data.append(semantic_trace)

        layout = go.Layout(
            title="Summary Evaluation Scores",
            barmode="group",
            yaxis=dict(title="Score"),
            xaxis=dict(title="Metric"),
            # Update the legend so that it only shows one entry per legendgroup
            legend=dict(groupclick="toggleitem"),
        )

        fig = go.Figure(data=data, layout=layout)

        st.plotly_chart(fig)

    def plot_radar_chart(self):
        # Categories
        categories = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Semantic Similarity"]

        # Extracting data for each model
        data = []
        for model, scores in self.all_scores.items():
            values = [
                scores["bleu_score"],
                scores["rouge_scores"]["rouge-1"]["f"],
                scores["rouge_scores"]["rouge-2"]["f"],
                scores["rouge_scores"]["rouge-l"]["f"],
                scores["semantic_similarity"],
            ]
            # Add data for radar plot
            data.append(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    line=dict(width=3),  # Adjust line width here
                    fillcolor="rgba(0,0,0,0)",  # Remove fill / make it fully transparent
                    opacity=0.7,
                    name=model,
                    marker=dict(
                        color=self.color_map.get(model, "gray")
                    ),  # Use the same color map for consistency
                )
            )

        # Layout of the radar chart
        layout = go.Layout(
            title="Model Performance Metrics",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],  # Adjust range based on your actual score values
                    color="white",  # Axis color to white
                ),
                bgcolor="rgba(0,0,0,0)",  # Transparent background for the polar area
                angularaxis=dict(
                    color="white"  # Angular axis color to white
                ),
            ),
            paper_bgcolor="black",  # Set the background color around the polar area to black
            font=dict(color="white"),  # Set the font color to white
            showlegend=True,
            legend=dict(font=dict(color="white")),  # Set legend font color to white
        )

        # Plot the radar chart
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)

    def render_score_table(self):
        # Prepare data for the table
        table_data = {}
        for model, metrics in self.all_scores.items():
            table_data[model] = {"Model": model}  # Adding model name as a column
            for metric, score in metrics.items():
                if metric == "rouge_scores":
                    for rouge_metric, rouge_scores in score.items():
                        # Assuming 'f' is the F1 score, adding ROUGE scores
                        table_data[model][f"{rouge_metric.upper()} F1"] = round(
                            rouge_scores["f"], 3
                        )
                else:
                    # Directly round the numerical score values for BLEU and Semantic Similarity
                    metric_name = (
                        "BLEU" if metric == "bleu_score" else "Semantic Similarity"
                    )
                    table_data[model][metric_name] = round(score, 3)

        # Create DataFrame and adjust the structure if necessary
        score_df = pd.DataFrame.from_dict(table_data, orient="index").reset_index(
            drop=True
        )

        # # Render the table with Streamlit
        # st.dataframe(score_df.style.highlight_max(axis=0), hide_index=True, use_container_width=True)

        def highlight_max(s, props=""):
            return np.where(s == np.nanmax(s.to_numpy()), props, "")

        # Apply the highlighting function only to data columns
        styled_df = score_df.style.apply(
            highlight_max,
            props="background-color:yellow;",
            axis=0,
            subset=pd.IndexSlice[:, "BLEU":"Semantic Similarity"],
        )
        # Render the table with Streamlit
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
