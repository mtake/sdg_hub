# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Install SDG
# ```bash 
# git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
# cd sdg_hub
# pip install .[examples]
# ```
# **⚠️ If you haven't already, run the document pre-processing notebook to create the seed data.**

# %%
# Third Party
from datasets import load_dataset

# First Party
from sdg_hub import Flow, FlowRegistry

# %%
# Required to run the flow with async mode
import nest_asyncio

nest_asyncio.apply()  

# %% [markdown]
# #### Configure timestamp

# %%
from datetime import datetime

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S')

# %% [markdown]
# #### Configure seed data

# %%
# data_name = ""
data_name = "teigaku-genzei"
# data_name = "teigaku-genzei-ibm-v0"
# data_name = "teigaku-genzei-ibm-v2"
# data_name = "teigaku-genzei-ibm-v3"
# data_name = "teigaku-genzei-ibm-v4"
# data_name = "teigaku-genzei-ibm-v5"
# data_name = "teigaku-genzei-ibm-v6"
# data_name = "ibm-newsroom"
# data_name = "ibm-newsroom-en"
# data_name = "jfe-technical-report"

if data_name.endswith(("_ja", "-ja")):
    data_lang = "ja"
elif data_name.endswith(("_en", "-en")):
    data_lang = ""
elif data_name.startswith(("teigaku-genzei", "ibm-newsroom", "jfe-technical-report")):
    data_lang = "ja"
else:
    data_lang = ""

repeat_times = 1
# repeat_times = 5

# %%
sdg_demo_output = "sdg_demo_output"

_data_name = f"_{data_name}" if data_name is not None and len(data_name) > 0 else ""
# _data_lang = f"_{data_lang}" if data_lang is not None and len(data_lang) > 0 else ""

_data_name_repeat = f"{_data_name}_r{repeat_times}" if repeat_times > 1 else _data_name

seed_data_dir = f"{sdg_demo_output}{_data_name}"
seed_data_path = f"{seed_data_dir}/seed_data.jsonl"
output_dir_prefix = f"{sdg_demo_output}{_data_name_repeat}"

# %% [markdown]
# ### Run SDG
# - This will create knowledge flow from provided yaml file
# - We will run this on small dataset for demo purposes
# - For large scale generation, please use the python command provided in the next cell
# - You can analyze the generated data to ensure the quality is similar to proivded QnA pairs

# %% [markdown]
# #### Discover the available generation flows

# %%
# Auto-discover all available flows (no setup needed!)
FlowRegistry.discover_flows()

# List available flows
flows = FlowRegistry.list_flows()
print(f"Available flows: {flows}")

# You can also search the flows by tag
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"QA flows: {qa_flows}")

# %% [markdown]
# #### Configure processing mode

# %%
# @@@ahoaho XXX
async_mode = True  # original
# async_mode = False  # for test

# %% [markdown]
# #### Determine the generation flow to use

# %%
# We will use the "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning" flow.
# For loading the flow simply use the fullname to load it
# @@@ahoaho XXX
# flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
for qa_flow in qa_flows:
    print(f"XXX qa_flow: {qa_flow}")
    flow_name = qa_flow["name"]
    metadata = FlowRegistry.get_flow_metadata(flow_name)

    has_japanese = "japanese" in metadata.tags
    if data_lang == "ja":
        if has_japanese:
            break
    else:
        if not has_japanese:
            break

print(f"XXX flow_name: {flow_name}")
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# %% [markdown]
# #### Identify the recommended model and set the model config

# %%
flow.get_default_model()

# %%
flow.get_model_recommendations()

# %% [markdown]
# #### Configure model server
# 
# [RITS](https://rits.fmaas.res.ibm.com/) is a model server for internal use.
# 
# This notebook still works without it by hosting the model using vLLM.

# %%
use_rits = True

# %% [markdown]
# #### Configure models

# %%
# HuggingFace model name
phi4_model_name_hf = "microsoft/phi-4"
gptoss20_model_name_hf = "openai/gpt-oss-20b"
gptoss_model_name_hf = "openai/gpt-oss-120b"
# llama3_model_name_hf = "meta-llama/Llama-3.3-70B-Instruct"
llama3_model_name_hf = "unsloth/Llama-3.3-70B-Instruct"
# llama4_model_name_hf = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
llama4_model_name_hf = "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"
mistral_model_name_hf = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
mixtral_model_name_hf = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if use_rits:
    # Served model name (RITS)
    phi4_model_name = "hosted_vllm/microsoft/phi-4"
    gptoss20_model_name = "hosted_vllm/openai/gpt-oss-20b"
    gptoss_model_name = "hosted_vllm/openai/gpt-oss-120b"
    llama3_model_name = "hosted_vllm/meta-llama/llama-3-3-70b-instruct"
    llama4_model_name = "hosted_vllm/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    mistral_model_name = "hosted_vllm/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    mixtral_model_name = "hosted_vllm/mistralai/mixtral-8x7B-instruct-v0.1"
else:
    # Served model name (self hosting via vLLM)
    phi4_model_name = f"hosted_vllm/{phi4_model_name_hf}"
    gptoss20_model_name = f"hosted_vllm/{gptoss20_model_name_hf}"
    gptoss_model_name = f"hosted_vllm/{gptoss_model_name_hf}"
    llama3_model_name = f"hosted_vllm/{llama3_model_name_hf}"
    llama4_model_name = f"hosted_vllm/{llama4_model_name_hf}"
    mistral_model_name = f"hosted_vllm/{mistral_model_name_hf}"
    mixtral_model_name = f"hosted_vllm/{mixtral_model_name_hf}"

# Model short name
phi4_short_name = "phi4"
gptoss20_short_name = "gptoss20"
gptoss_short_name = "gptoss"
llama3_short_name = "llama3"
llama4_short_name = "llama4"
mistral_short_name = "mistral"
mixtral_short_name = "mixtral"

# %%
use_phi4 = True
use_gptoss20 = False
use_gptoss = False
use_llama3 = False
use_llama4 = False
use_mistral = False
use_mixtral = False

# %%
if use_phi4:
    # @@@ahoaho XXX
    # model_name_hf = phi4_model_name_hf
    model_name = phi4_model_name
    short_name = phi4_short_name
elif use_gptoss20:
    # @@@ahoaho XXX
    # model_name_hf = gptoss20_model_name_hf
    model_name = gptoss20_model_name
    short_name = gptoss20_short_name
elif use_gptoss:
    # @@@ahoaho XXX
    # model_name_hf = gptoss_model_name_hf
    model_name = gptoss_model_name
    short_name = gptoss_short_name
elif use_llama3:
    # @@@ahoaho XXX
    # model_name_hf = llama3_model_name_hf
    model_name = llama3_model_name
    short_name = llama3_short_name
elif use_llama4:
    # @@@ahoaho XXX
    # model_name_hf = llama4_model_name_hf
    model_name = llama4_model_name
    short_name = llama4_short_name
elif use_mistral:
    # @@@ahoaho XXX
    # model_name_hf = mistral_model_name_hf
    model_name = mistral_model_name
    short_name = mistral_short_name
elif use_mixtral:
    # @@@ahoaho XXX
    # model_name_hf = mixtral_model_name_hf
    model_name = mixtral_model_name
    short_name = mixtral_short_name

# output_dir = f"{output_dir_prefix}_{short_name}"
output_dir = f"{output_dir_prefix}_{short_name}_{timestamp}"

# %%
import os
import requests

if use_rits:
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    default_headers = {"RITS_API_KEY": RITS_API_KEY}

    url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
    res = requests.get(url=url, headers=default_headers)
    assert res.status_code == 200
    model_list: list[dict[str, str]] = res.json()
    model_dict = { m["model_name"]: m["endpoint"] for m in model_list }
    # avoid crashes in model_name
    model_dict[phi4_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4"
    model_dict[gptoss20_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-20b"
    model_dict[gptoss_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b"
    model_dict[llama3_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
    model_dict[llama4_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8"
    model_dict[mistral_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-small-3-2-24b-2506"
    model_dict[mixtral_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01"
else:
    default_headers: dict[str, str] = {}
    model_dict: dict[str, str] = {}

def get_base_url(model_name: str) -> str:
    endpoint = model_dict.get(model_name, "http://0.0.0.0:8000")  # fall back to vllm
    return f"{endpoint}/v1"

# %%
# You can dynamically change the model without having to change the flow yaml file.
# Configure the flow to use a vllm model hosted at localhost:8000/v1. 
# @@@ahoaho XXX
# flow.set_model_config(
#     model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
#     api_base="http://localhost:8000/v1",
#     api_key="EMPTY",
# )
flow.set_model_config(
    model=model_name,
    api_base=get_base_url(model_name),
    api_key="EMPTY",
    async_mode=async_mode,
    timeout=3600,
    extra_headers=default_headers,
)

# %% [markdown]
# #### Load and prepare seed data

# %%
# Load the seed data
# number_of_samples = 2
# ds = load_dataset('json', data_files=seed_data_path, split='train')
# ds = ds.shuffle(seed=42).select(range(number_of_samples))

# %%
# Load the seed data
ds = load_dataset('json', data_files=seed_data_path, split='train')

# %% [markdown]
# Repeat (duplicate) seed data

# %%
if repeat_times > 1:
    ds = ds.repeat(repeat_times)

# %% [markdown]
# Shuffle seed data

# %%
# ds = ds.shuffle(seed=42)

# %% [markdown]
# (Optional) sample seed data

# %%
# number_of_samples = 2
# ds = ds.select(range(number_of_samples))

# %% [markdown]
# Add seed id

# %%
# Add seed_id column to preserve repetition in seed data
# See https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/42650f1340a2d3576818d68e05508dfe2a8d04bd/src/sdg_hub/checkpointer.py#L103
ds = ds.add_column("seed_id", list(range(len(ds))))

# %%
print(f"Loaded {len(ds)} seed data", flush=True)

# %% [markdown]
# Configure checkpointing

# %%
# checkpoint_dir = None
# save_freq = None  # Save the last checkpoint when everything is finished. Not useful.
# max_concurrency = None

checkpoint_dir = f"{output_dir}_ckpt"

# See https://github.com/mtake/sdg_hub/blob/main/docs/blocks/llm-blocks.md#async-processing--concurrency-control

# save_freq = 1000  # Request timed out. - timeout value=600.0, time taken=1802.91 seconds
# save_freq = 100  # Creates 200 checkpoints
# save_freq = 10  # Creates 2000+ checkpoints
# max_concurrency = 20
# max_concurrency = 10

# MEM=100G, sdg_hub.core.utils.error_handling.FlowValidationError: Block 'gen_detailed_summary' execution failed: litellm.NotFoundError: NotFoundError: Hosted_vllmException - Error getting active endpoint: revision.serving.knative.dev "microsoft-phi-4-predictor-00062" not found
# save_freq = None
# max_concurrency = None

# MEM=100G, Finished in 14.0 hours, 18449 QA pairs. samples_processed: 20424, checkpoint_counter: 205
# save_freq = 100
# max_concurrency = None

# MEM=100G, sdg_hub.core.utils.error_handling.FlowValidationError: Block 'gen_detailed_summary' execution failed: litellm.NotFoundError: NotFoundError: Hosted_vllmException - Error getting active endpoint: revision.serving.knative.dev "microsoft-phi-4-predictor-00054" not found
# save_freq = 1000
# max_concurrency = 30

# MEM=100G, sdg_hub.core.utils.error_handling.FlowValidationError: Block 'gen_detailed_summary' execution failed: litellm.NotFoundError: NotFoundError: Hosted_vllmException - Error getting active endpoint: revision.serving.knative.dev "microsoft-phi-4-predictor-00064" not found
# save_freq = 500
# max_concurrency = 30

# MEM=100G, Finished in 8.3 hours, 18524 QA pairs. samples_processed: 20524, checkpoint_counter: 42
# save_freq = 200
# max_concurrency = 30

# MEM=100G, Finished in 7.5 hours, 18347 QA pairs. samples_processed: 20344, checkpoint_counter: 204
# save_freq = 100
# max_concurrency = 30

# [v0.2 20250907-034856] MEM=100G, Finished in 6.9 hours (24679 secs), 18656 QA pairs. samples_processed: 20668, checkpoint_counter: 1
# save_freq = None
# max_concurrency = 30

# [v0.2 best 20250908-130424] MEM=100G, Finished in 4.5 hours (16275 secs), 18336 QA pairs. samples_processed: 20306, checkpoint_counter: 1
save_freq = None
max_concurrency = 40

# [v0.2 20250909-015619] MEM=100G, Unclosed client session. client_session: <aiohttp.client.ClientSession object at 0x14dc52560b60>
# save_freq = None
# max_concurrency = 50

# MEM=100G, async_mode: false. Very slow. Killed
# -n 64, MEM=200G, async_mode: false. Very slow. Killed
# save_freq = None
# max_concurrency = 30

# MEM=100G, async_mode: false. Very slow. Killed
# save_freq = 100
# max_concurrency = 30

# MEM=100G, async_mode: false. Very slow. Killed
# -n 16, MEM=100G, async_mode: false. Slow. Killed
# save_freq = 8
# max_concurrency = 8

# MEM=100G, async_mode: false. Very slow. Killed
# save_freq = None
# max_concurrency = None

# %%
# Generate data
generated_data = flow.generate(ds, checkpoint_dir=checkpoint_dir, save_freq=save_freq, max_concurrency=max_concurrency)

# %% [markdown]
# ### Converting the generated data into training format

# %%
from datasets import Dataset

def create_simple_qa_dataset(generated_data: Dataset) -> Dataset:
    seen = set()
    messages_list: list[dict[str, any]] = []
    for generated_data_i in generated_data:
        user = generated_data_i['question']
        assistant = generated_data_i['response']
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        # deduplicate messages
        key = tuple([frozenset(d.items()) for d in messages])
        if key not in seen:
            seen.add(key)
            messages_list.append({"messages": messages})
    messages_data = Dataset.from_list(messages_list)
    return messages_data

# %%
messages_data = create_simple_qa_dataset(generated_data)

messages_data.to_json(f"{output_dir}/messages_data.jsonl", orient="records", lines=True)

# %% [markdown]
# ### (Original code) Converting the generated data into training format

# %%
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
# from knowledge_utils import create_knowledge_regular_ds, create_knowledge_pretraining_ds

# from datasets import concatenate_datasets

# # Create Pretraining Knowledge Dataset (Also known as Phase 0.7/Phase 7)
# instructlab_phase_1_ds = create_knowledge_pretraining_ds(generated_data)
# instructlab_phase_1_ds.to_json(f'{output_dir}/instructlab_phase_1_ds.jsonl', orient='records', lines=True)

# # Create Regular Knowledge Dataset (Also known as Phase 1.0/Phase 10)
# instructlab_phase_2_ds = create_knowledge_regular_ds(generated_data)

# # Mix the pre-computed skills with the regular knowledge dataset. If more than one dataset were generated simply add those in this concatenation stage.
# # If you have any generated instruction data, that can be also mixed in this stage. If you only have generated skills phase 07 generation and training can be skipped.
# instructlab_phase_2_ds.to_json(f'{output_dir}/instructlab_phase_2_ds.jsonl', orient='records', lines=True)

# %%
# # If you have any other instruction tuning datasets you can mix with phase 2 dataset.
# instruction_tuning_dataset_path = "<Your instruction tuning dataset path>"
# instruction_tuning_dataset = load_dataset('json', data_files=instruction_tuning_dataset_path, split='train')
# instructlab_phase_2_ds = concatenate_datasets([instructlab_phase_2_ds, instruction_tuning_dataset])
# instructlab_phase_2_ds.to_json(f'{output_dir}/instructlab_phase_2_ds.jsonl', orient='records', lines=True)


