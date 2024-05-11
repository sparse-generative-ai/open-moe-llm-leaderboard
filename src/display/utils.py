from dataclasses import dataclass, make_dataclass
from enum import Enum

import pandas as pd


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]

E2Es = "E2E(s)" #"End-to-end time (s)"
PREs = "PRE(s)" #"Prefilling time (s)"
TS = "T/s" #Decoding throughput (tok/s)
InFrame = "Method" #"Inference framework"
MULTIPLE_CHOICEs = ["mmlu"]

GPU_TEMP = 'Temp(C)'
GPU_Power = 'Power(W)'
GPU_Mem = 'Mem(G)'
GPU_Name = "GPU"
GPU_Util = 'Util(%)'
BATCH_SIZE = 'bs'
PRECISION = "Precision"
system_metrics_to_name_map = {
    "end_to_end_time": f"{E2Es}",
    "prefilling_time": f"{PREs}",
    "decoding_throughput": f"{TS}",
}

gpu_metrics_to_name_map = {
    GPU_Util: GPU_Util,
    GPU_TEMP: GPU_TEMP,
    GPU_Power: GPU_Power,
    GPU_Mem: GPU_Mem,
    "batch_size": BATCH_SIZE,
    "precision": PRECISION,
    GPU_Name: GPU_Name,
}

@dataclass
class Task:
    benchmark: str
    metric: str
    col_name: str


class Tasks(Enum):
    # XXX include me back at some point
    # nqopen = Task("nq8", "em", "NQ Open/EM")
    # triviaqa = Task("tqa8", "em", "TriviaQA/EM")

    # truthfulqa_mc1 = Task("truthfulqa_mc1", "acc", "TruthQA MC1/Acc")
    # truthfulqa_mc2 = Task("truthfulqa_mc2", "acc", "TruthQA MC2/Acc")
    # truthfulqa_gen = Task("truthfulqa_gen", "rougeL_acc", "TruthQA Gen/ROUGE")

    # xsum_r = Task("xsum_v2", "rougeL", "XSum/ROUGE")
    # xsum_f = Task("xsum_v2", "factKB", "XSum/factKB")
    # xsum_b = Task("xsum_v2", "bertscore_precision", "XSum/BERT-P")

    # cnndm_r = Task("cnndm_v2", "rougeL", "CNN-DM/ROUGE")
    # cnndm_f = Task("cnndm_v2", "factKB", "CNN-DM/factKB")
    # cnndm_b = Task("cnndm_v2", "bertscore_precision", "CNN-DM/BERT-P")

    # race = Task("race", "acc", "RACE/Acc")
    # squadv2 = Task("squadv2", "exact", "SQUaDv2/EM")

    # memotrap = Task("memo-trap_v2", "acc", "MemoTrap/Acc")
    # ifeval = Task("ifeval", "prompt_level_strict_acc", "IFEval/Acc")

    # faithdial = Task("faithdial_hallu_v2", "acc", "FaithDial/Acc")

    # halueval_qa = Task("halueval_qa", "acc", "HaluQA/Acc")
    # halueval_summ = Task("halueval_summarization", "acc", "HaluSumm/Acc")
    # halueval_dial = Task("halueval_dialogue", "acc", "HaluDial/Acc")

    # # XXX include me back at some point
    selfcheck = Task("selfcheckgpt", "max-selfcheckgpt", "SelfCheckGPT")
    mmlu = Task("mmlu", "acc", "MMLU") #MMLU/Acc (5-shot)
    gsm8k = Task("gsm8k_custom", "em", "GSM8K") #GSM8K/EM (8-shot)


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modif is needed
@dataclass
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False
    dummy: bool = False


auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(["model_type_symbol", ColumnContent, ColumnContent("T", "str", True, never_hidden=True)])
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])

# #Scores
# # auto_eval_column_dict.append(["average", ColumnContent, ColumnContent("Avg", "number", True)])

# Inference framework
auto_eval_column_dict.append(["inference_framework", ColumnContent, ColumnContent(f"{InFrame}", "str", True)])

for task in Tasks:
    auto_eval_column_dict.append([task.name, ColumnContent, ColumnContent(task.value.col_name, "number", True)])
    # System performance metrics
    auto_eval_column_dict.append([f"{task.name}_end_to_end_time", ColumnContent, ColumnContent(f"{task.value.col_name} {E2Es}", "number", True, hidden=True)])
    auto_eval_column_dict.append([f"{task.name}_batch_size", ColumnContent, ColumnContent(f"{task.value.col_name} {BATCH_SIZE}", "number", True, hidden=True)])
    # auto_eval_column_dict.append([f"{task.name}_precision", ColumnContent, ColumnContent(f"{task.value.col_name} {PRECISION}", "str", True, hidden=True)])
    auto_eval_column_dict.append([f"{task.name}_gpu_mem", ColumnContent, ColumnContent(f"{task.value.col_name} {GPU_Mem}", "number", True, hidden=True)])
    auto_eval_column_dict.append([f"{task.name}_gpu", ColumnContent, ColumnContent(f"{task.value.col_name} {GPU_Name}", "str", True, hidden=True)])
    auto_eval_column_dict.append([f"{task.name}_gpu_util", ColumnContent, ColumnContent(f"{task.value.col_name} {GPU_Util}", "number", True, hidden=True)])
    if task.value.benchmark in MULTIPLE_CHOICEs:
        continue
    # auto_eval_column_dict.append([f"{task.name}_prefilling_time", ColumnContent, ColumnContent(f"{task.value.col_name} {PREs}", "number", False, hidden=True)])
    auto_eval_column_dict.append([f"{task.name}_decoding_throughput", ColumnContent, ColumnContent(f"{task.value.col_name} {TS}", "number", True, hidden=True)])


# Model information
auto_eval_column_dict.append(["model_type", ColumnContent, ColumnContent("Type", "str", False)])
auto_eval_column_dict.append(["architecture", ColumnContent, ColumnContent("Architecture", "str", False)])
auto_eval_column_dict.append(["weight_type", ColumnContent, ColumnContent("Weight type", "str", False, True)])
auto_eval_column_dict.append(["precision", ColumnContent, ColumnContent("Precision", "str", True)])
auto_eval_column_dict.append(["license", ColumnContent, ColumnContent("Hub License", "str", False)])
auto_eval_column_dict.append(["params", ColumnContent, ColumnContent("#Params (B)", "number", False)])
auto_eval_column_dict.append(["likes", ColumnContent, ColumnContent("Hub ❤️", "number", False)])
auto_eval_column_dict.append(["still_on_hub", ColumnContent, ColumnContent("Available on the hub", "bool", False)])
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Model sha", "str", False, False)])
# Dummy column for the search bar (hidden by the custom CSS)
auto_eval_column_dict.append(["dummy", ColumnContent, ColumnContent("model_name_for_query", "str", False, dummy=True)])

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    private = ColumnContent("private", "bool", True)
    precision = ColumnContent("precision", "str", True)
    weight_type = ColumnContent("weight_type", "str", "Original")
    model_framework = ColumnContent("inference_framework", "str", True)
    status = ColumnContent("status", "str", True)


@dataclass
class ModelDetails:
    name: str
    symbol: str = ""  # emoji, only for the model type


class ModelType(Enum):
    PT = ModelDetails(name="pretrained", symbol="🟢")
    FT = ModelDetails(name="fine-tuned on domain-specific datasets", symbol="🔶")
    chat = ModelDetails(name="chat models (RLHF, DPO, IFT, ...)", symbol="💬")
    merges = ModelDetails(name="base merges and moerges", symbol="🤝")
    Unknown = ModelDetails(name="", symbol="?")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if any([k in type for k in ["instruction-tuned", "RL-tuned", "chat", "🟦", "⭕", "💬"]]):
            return ModelType.chat
        if "merge" in type or "🤝" in type:
            return ModelType.merges
        return ModelType.Unknown


class InferenceFramework(Enum):
    # "moe-infinity", hf-chat
    MoE_Infinity = ModelDetails("moe-infinity")
    HF_Chat = ModelDetails("hf-chat")
    Unknown = ModelDetails("?")

    def to_str(self):
        return self.value.name

    @staticmethod
    def from_str(inference_framework: str):
        if inference_framework in ["moe-infinity"]:
            return InferenceFramework.MoE_Infinity
        if inference_framework in ["hf-chat"]:
            return InferenceFramework.HF_Chat
        return InferenceFramework.Unknown

class GPUType(Enum):
    H100_pcie = ModelDetails("NVIDIA-H100-PCIe-80GB")
    A100_pcie = ModelDetails("NVIDIA-A100-PCIe-80GB")
    A5000 = ModelDetails("NVIDIA-RTX-A5000-24GB")
    Unknown = ModelDetails("?")

    def to_str(self):
        return self.value.name

    @staticmethod
    def from_str(gpu_type: str):
        if gpu_type in ["NVIDIA-H100-PCIe-80GB"]:
            return GPUType.A100_pcie
        if gpu_type in ["NVIDIA-A100-PCIe-80GB"]:
            return GPUType.H100_pcie
        if gpu_type in ["NVIDIA-A5000-24GB"]:
            return GPUType.A5000
        return GPUType.Unknown
    
class WeightType(Enum):
    Adapter = ModelDetails("Adapter")
    Original = ModelDetails("Original")
    Delta = ModelDetails("Delta")


class Precision(Enum):
    float32 = ModelDetails("float32")
    float16 = ModelDetails("float16")
    bfloat16 = ModelDetails("bfloat16")
    qt_8bit = ModelDetails("8bit")
    qt_4bit = ModelDetails("4bit")
    qt_GPTQ = ModelDetails("GPTQ")
    Unknown = ModelDetails("?")

    @staticmethod
    def from_str(precision: str):
        if precision in ["torch.float32", "float32"]:
            return Precision.float32
        if precision in ["torch.float16", "float16"]:
            return Precision.float16
        if precision in ["torch.bfloat16", "bfloat16"]:
            return Precision.bfloat16
        if precision in ["8bit"]:
            return Precision.qt_8bit
        if precision in ["4bit"]:
            return Precision.qt_4bit
        if precision in ["GPTQ", "None"]:
            return Precision.qt_GPTQ
        return Precision.Unknown


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn)]
TYPES = [c.type for c in fields(AutoEvalColumn)]
COLS_LITE = [c.name for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden]
TYPES_LITE = [c.type for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Tasks]

# NUMERIC_INTERVALS = {
#     "?": pd.Interval(-1, 0, closed="right"),
#     "~1.5": pd.Interval(0, 2, closed="right"),
#     "~3": pd.Interval(2, 4, closed="right"),
#     "~7": pd.Interval(4, 9, closed="right"),
#     "~13": pd.Interval(9, 20, closed="right"),
#     "~35": pd.Interval(20, 45, closed="right"),
#     "~60": pd.Interval(45, 70, closed="right"),
#     "70+": pd.Interval(70, 10000, closed="right"),
# }
