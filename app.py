#!/usr/bin/env python
import os
import datetime
import socket
import base64
from threading import Thread

import gradio as gr
import pandas as pd
import time
from apscheduler.schedulers.background import BackgroundScheduler

from huggingface_hub import snapshot_download

from src.display.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    LLM_BENCHMARKS_DETAILS,
    FAQ_TEXT,
    TITLE,
    ACKNOWLEDGEMENT_TEXT,
)

from src.display.css_html_js import custom_css

from src.display.utils import (
    BENCHMARK_COLS,
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    TYPES,
    AutoEvalColumn,
    ModelType,
    InferenceFramework,
    fields,
    WeightType,
    Precision,
    GPUType
)

from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, H4_TOKEN, IS_PUBLIC, \
    QUEUE_REPO, REPO_ID, RESULTS_REPO, DEBUG_QUEUE_REPO, DEBUG_RESULTS_REPO
from src.populate import get_evaluation_queue_df, get_leaderboard_df
from src.submission.submit import add_new_eval
from src.utils import get_dataset_summary_table

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run the LLM Leaderboard")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()

args = get_args()
if args.debug:
    print("Running in debug mode")
    QUEUE_REPO = DEBUG_QUEUE_REPO
    RESULTS_REPO = DEBUG_RESULTS_REPO

def ui_snapshot_download(repo_id, local_dir, repo_type, tqdm_class, etag_timeout):
    try:
        print(local_dir)
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type=repo_type, tqdm_class=tqdm_class, etag_timeout=etag_timeout
        )
    except Exception as e:
        restart_space()


def restart_space():
    API.restart_space(repo_id=REPO_ID, token=H4_TOKEN)


def init_space():
    dataset_df = get_dataset_summary_table(file_path="blog/Hallucination-Leaderboard-Summary.csv")

    if socket.gethostname() not in {"neuromancer"}:
        # sync model_type with open-llm-leaderboard
        ui_snapshot_download(
            repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
        )
        ui_snapshot_download(
            repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
        )
    raw_data, original_df = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, "", COLS, BENCHMARK_COLS)

    finished_eval_queue_df, running_eval_queue_df, pending_eval_queue_df = get_evaluation_queue_df(
        EVAL_REQUESTS_PATH, EVAL_COLS
    )
    return dataset_df, original_df, finished_eval_queue_df, running_eval_queue_df, pending_eval_queue_df
    
    
def add_benchmark_columns(shown_columns):
    benchmark_columns = []
    for benchmark in BENCHMARK_COLS:
        if benchmark in shown_columns:
            for c in COLS:
                if benchmark in c and benchmark != c:
                    benchmark_columns.append(c)
    return benchmark_columns


# Searching and filtering
def update_table(
    hidden_df: pd.DataFrame, columns: list, type_query: list, precision_query: list, size_query: list, query: str
):
    filtered_df = filter_models(hidden_df, type_query, size_query, precision_query)
    filtered_df = filter_queries(query, filtered_df)
    benchmark_columns = add_benchmark_columns(columns)
    df = select_columns(filtered_df, columns + benchmark_columns)
    return df


def search_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df[(df[AutoEvalColumn.dummy.name].str.contains(query, case=False))]


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    # always_here_cols = [AutoEvalColumn.model_type_symbol.name, AutoEvalColumn.model.name]

    always_here_cols = [c.name for c in fields(AutoEvalColumn) if c.never_hidden]
    dummy_col = [AutoEvalColumn.dummy.name]

    # We use COLS to maintain sorting
    filtered_df = df[
        # always_here_cols + [c for c in COLS if c in df.columns and c in columns] + [AutoEvalColumn.dummy.name]
        always_here_cols
        + [c for c in COLS if c in df.columns and c in columns]
        + dummy_col
    ]
    return filtered_df


def filter_queries(query: str, filtered_df: pd.DataFrame):
    final_df = []
    if query != "":
        queries = [q.strip() for q in query.split(";")]
        for _q in queries:
            _q = _q.strip()
            if _q != "":
                temp_filtered_df = search_table(filtered_df, _q)
                if len(temp_filtered_df) > 0:
                    final_df.append(temp_filtered_df)
        if len(final_df) > 0:
            filtered_df = pd.concat(final_df)
            subset = [AutoEvalColumn.model.name, AutoEvalColumn.precision.name, AutoEvalColumn.revision.name]
            filtered_df = filtered_df.drop_duplicates(subset=subset)
    return filtered_df


def filter_models(df: pd.DataFrame, type_query: list, size_query: list, precision_query: list) -> pd.DataFrame:
    # Show all models
    filtered_df = df

    type_emoji = [t[0] for t in type_query]
    filtered_df = filtered_df.loc[df[AutoEvalColumn.model_type_symbol.name].isin(type_emoji)]
    filtered_df = filtered_df.loc[df[AutoEvalColumn.precision.name].isin(precision_query + ["None"])]

    # numeric_interval = pd.IntervalIndex(sorted([NUMERIC_INTERVALS[s] for s in size_query]))
    # params_column = pd.to_numeric(df[AutoEvalColumn.params.name], errors="coerce")
    # mask = params_column.apply(lambda x: any(numeric_interval.contains(x)))
    # filtered_df = filtered_df.loc[mask]

    return filtered_df

shown_columns = None
dataset_df, original_df, finished_eval_queue_df, running_eval_queue_df, pending_eval_queue_df = init_space()
leaderboard_df = original_df.copy()

# def update_leaderboard_table():
#     global leaderboard_df, shown_columns
#     print("Updating leaderboard table")
#     return leaderboard_df[
#                 [c.name for c in fields(AutoEvalColumn) if c.never_hidden]
#                 + shown_columns.value
#                 + [AutoEvalColumn.dummy.name]
#             ] if not leaderboard_df.empty else leaderboard_df
        

# def update_hidden_leaderboard_table():
#     global original_df
#     return original_df[COLS] if original_df.empty is False else original_df

# def update_dataset_table():
#     global dataset_df
#     return dataset_df

# def update_finish_table():
#     global finished_eval_queue_df
#     return finished_eval_queue_df

# def update_running_table():
#     global running_eval_queue_df
#     return running_eval_queue_df

# def update_pending_table():
#     global pending_eval_queue_df
#     return pending_eval_queue_df

# def update_finish_num():
#     global finished_eval_queue_df
#     return len(finished_eval_queue_df)

# def update_running_num():
#     global running_eval_queue_df
#     return len(running_eval_queue_df)

# def update_pending_num():
#     global pending_eval_queue_df
#     return len(pending_eval_queue_df)

# triggered only once at startup => read query parameter if it exists
def load_query(request: gr.Request):
    query = request.query_params.get("query") or ""
    return query


def get_image_html(url, image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'<a href="{url}" target="_blank"><img src="data:image/jpg;base64,{encoded_string}" alt="NetMind.AI Logo" style="width:100pt;"></a>'


# Prepare the HTML content with the image
image_html = get_image_html("https://netmind.ai/home", "./src/display/imgs/Netmind.AI_LOGO.jpg")


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
    gr.HTML(ACKNOWLEDGEMENT_TEXT.format(image_html=image_html))

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("open-moe-llm-leaderboard", elem_id="llm-benchmark-tab-table", id=0):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Model search (separate multiple queries with `;`)",
                            show_label=False,
                            elem_id="search-bar"
                        )
                    with gr.Row():
                        shown_columns = gr.CheckboxGroup(
                            choices=[
                                c.name
                                for c in fields(AutoEvalColumn)
                                if not c.hidden and not c.never_hidden and not c.dummy
                            ],
                            value=[
                                c.name
                                for c in fields(AutoEvalColumn)
                                if c.displayed_by_default and not c.hidden and not c.never_hidden
                            ],
                            label="Select columns to show",
                            elem_id="column-select",
                            interactive=True,
                        )

                with gr.Column(min_width=320):
                    filter_columns_size = gr.CheckboxGroup(
                        label="Inference frameworks",
                        choices=[t.to_str() for t in InferenceFramework],
                        value=[t.to_str() for t in InferenceFramework],
                        interactive=True,
                        elem_id="filter-columns-size",
                    )

                    filter_columns_type = gr.CheckboxGroup(
                        label="Model types",
                        choices=[t.to_str() for t in ModelType],
                        value=[t.to_str() for t in ModelType],
                        interactive=True,
                        elem_id="filter-columns-type",
                    )

                    filter_columns_precision = gr.CheckboxGroup(
                        label="Precision",
                        choices=[i.value.name for i in Precision],
                        value=[i.value.name for i in Precision],
                        interactive=True,
                        elem_id="filter-columns-precision",
                    )

                    # filter_columns_size = gr.CheckboxGroup(
                    #     label="Model sizes (in billions of parameters)",
                    #     choices=list(NUMERIC_INTERVALS.keys()),
                    #     value=list(NUMERIC_INTERVALS.keys()),
                    #     interactive=True,
                    #     elem_id="filter-columns-size",
                    # )

            # breakpoint()
            benchmark_columns = add_benchmark_columns(shown_columns.value)
            leaderboard_table = gr.components.Dataframe(
                value=(
                    leaderboard_df[
                        [c.name for c in fields(AutoEvalColumn) if c.never_hidden]
                        + shown_columns.value
                        + benchmark_columns
                        + [AutoEvalColumn.dummy.name]
                    ]
                    if leaderboard_df.empty is False
                    else leaderboard_df
                ),
                headers=[c.name for c in fields(AutoEvalColumn) if c.never_hidden] + shown_columns.value + benchmark_columns,
                datatype=TYPES,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )  # column_widths=["2%", "20%"]

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=original_df[COLS] if original_df.empty is False else original_df,
                headers=COLS,
                datatype=TYPES,
                visible=False,
            )

            search_bar.submit(
                update_table,
                [
                    hidden_leaderboard_table_for_search,
                    shown_columns,
                    filter_columns_type,
                    filter_columns_precision,
                    filter_columns_size,
                    search_bar,
                ],
                leaderboard_table
            )

            # Check query parameter once at startup and update search bar
            demo.load(load_query, inputs=[], outputs=[search_bar])

            for selector in [shown_columns, filter_columns_type, filter_columns_precision, filter_columns_size]:
                selector.change(
                    update_table,
                    [
                        hidden_leaderboard_table_for_search,
                        shown_columns,
                        filter_columns_type,
                        filter_columns_precision,
                        filter_columns_size,
                        search_bar,
                    ],
                    leaderboard_table,
                    queue=True,
                )

        with gr.TabItem("About", elem_id="llm-benchmark-tab-table", id=2):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

            dataset_table = gr.components.Dataframe(
                value=dataset_df,
                headers=list(dataset_df.columns),
                datatype=["str", "markdown", "str", "str", "str"],
                elem_id="dataset-table",
                interactive=False,
                visible=True,
                column_widths=["15%", "20%"],
            )

            gr.Markdown(LLM_BENCHMARKS_DETAILS, elem_classes="markdown-text")
            gr.Markdown(FAQ_TEXT, elem_classes="markdown-text")

        with gr.TabItem("Submit a model ", elem_id="llm-benchmark-tab-table", id=3):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")

                with gr.Column():
                    with gr.Accordion(f"✅ Finished Evaluations ({len(finished_eval_queue_df)})", open=False):
                        with gr.Row():
                            finished_eval_table = gr.components.Dataframe(
                                value=finished_eval_queue_df, headers=EVAL_COLS, datatype=EVAL_TYPES, row_count=5
                            )

                    with gr.Accordion(f"🔄 Running Evaluation Queue ({len(running_eval_queue_df)})", open=False):
                        with gr.Row():
                            running_eval_table = gr.components.Dataframe(
                                value=running_eval_queue_df, headers=EVAL_COLS, datatype=EVAL_TYPES, row_count=5
                            )

                    with gr.Accordion(f"⏳ Scheduled Evaluation Queue ({len(pending_eval_queue_df)})", open=False):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df, headers=EVAL_COLS, datatype=EVAL_TYPES, row_count=5
                            )

            with gr.Row():
                gr.Markdown("# Submit your model here", elem_classes="markdown-text")

            with gr.Row():
                inference_framework = gr.Dropdown(
                    choices=[t.to_str() for t in InferenceFramework],
                    label="Inference framework",
                    multiselect=False,
                    value=None,
                    interactive=True,
                )
                
                gpu_type = gr.Dropdown(
                    choices=[t.to_str() for t in GPUType],
                    label="GPU type",
                    multiselect=False,
                    value="NVIDIA-A100-PCIe-80GB",
                    interactive=True,
                )
                

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="Model name")
                    revision_name_textbox = gr.Textbox(label="Revision commit", placeholder="main")
                    private = gr.Checkbox(False, label="Private", visible=not IS_PUBLIC)
                    model_type = gr.Dropdown(
                        choices=[t.to_str(" : ") for t in ModelType if t != ModelType.Unknown],
                        label="Model type",
                        multiselect=False,
                        value=None,
                        interactive=True,
                    )

                with gr.Column():
                    precision = gr.Dropdown(
                        choices=[i.value.name for i in Precision if i != Precision.Unknown],
                        label="Precision",
                        multiselect=False,
                        value="float32",
                        interactive=True,
                    )

                    weight_type = gr.Dropdown(
                        choices=[i.value.name for i in WeightType],
                        label="Weights type",
                        multiselect=False,
                        value="Original",
                        interactive=True,
                    )

                    base_model_name_textbox = gr.Textbox(label="Base model (for delta or adapter weights)")

            submit_button = gr.Button("Submit Eval")
            submission_result = gr.Markdown()
            debug = gr.Checkbox(value=args.debug, label="Debug", visible=False)
            submit_button.click(
                add_new_eval,
                [
                    model_name_textbox,
                    base_model_name_textbox,
                    revision_name_textbox,
                    precision,
                    private,
                    weight_type,
                    model_type,
                    inference_framework,
                    debug,
                    gpu_type
                ],
                submission_result,
            )

    with gr.Row():
        with gr.Accordion("Citing this leaderboard", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=20,
                elem_id="citation-button",
                show_copy_button=True,
            )

scheduler = BackgroundScheduler()

scheduler.add_job(restart_space, "interval", hours=6)

def launch_backend():
    import subprocess
    from src.backend.envs import DEVICE

    if DEVICE not in {"cpu"}:
        _ = subprocess.run(["python", "backend-cli.py"])

# Thread(target=periodic_init, daemon=True).start()
# scheduler.add_job(launch_backend, "interval", seconds=120)
if __name__ == "__main__":
    scheduler.start()
    demo.queue(default_concurrency_limit=40).launch()
    
