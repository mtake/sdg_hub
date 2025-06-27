# %% [markdown]
# # Synthetic Data Generation Tutorial using phi4, llama3, llama4, and mixtral
# 
# This tutorial demonstrates how to use SDG repository to generate synthetic question-answer pairs from documents using large language models like phi4. We will also generate data using llama3, llama4, and mixtral models for comparison. We'll cover:
# 
# 1. Setting up the environment
# 2. Connecting to LLM servers
# 3. Configuring the data generation pipeline
# 4. Generating data with different models
# 5. Comparing results

# %%
# Enable auto-reloading of modules - useful during development
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Setup Instructions
# 
# Before running this notebook, you'll need to:
# 
# ```bash 
# pip install sdg-hub==0.1.0a4
# ```

# %%
# %%capture
# %pip install transformers

# %%
# Import required libraries
# datasets: For handling our data
# OpenAI: For interfacing with the LLM servers
# SDG components: For building our data generation pipeline
from datasets import load_dataset, Dataset
from openai import OpenAI
from transformers import AutoTokenizer

from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
from sdg_hub.registry import PromptRegistry

# %%
from datetime import datetime

now = datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S')

# %% [markdown]
# ### Configure Output

# %%
force_ascii = True

# %% [markdown]
# ### Configure Flow

# %%
flow_config = "synth_knowledge1.5"
# flow_config = "synth_knowledge1.5_0617"

# %% [markdown]
# ### Configure Parallelism

# %%
# src/sdg_hub/flow_runner.py
num_workers = 32   # Number of worker processes to use, by default 32.
batch_size = 8     # Batch size for processing, by default 8.
save_freq = 2      # Frequency (in batches) at which to save checkpoints, by default 2.

# For test
# num_workers = 1    # Number of worker processes to use, by default 32.
# batch_size = 1     # Batch size for processing, by default 8.
# save_freq = 1000   # Frequency (in batches) at which to save checkpoints, by default 2.

# %% [markdown]
# ### Configure Models

# %%
# Served model name
phi4_model_name = "microsoft/phi-4"
llama3_model_name = "meta-llama/llama-3-3-70b-instruct"
llama4_model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
mixtral_model_name = "mistralai/mixtral-8x7B-instruct-v0.1"

# HuggingFace model name
phi4_model_name_hf = "microsoft/phi-4"
# llama3_model_name_hf = "meta-llama/Llama-3.3-70B-Instruct"
llama3_model_name_hf = "unsloth/Llama-3.3-70B-Instruct"
# llama4_model_name_hf = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
llama4_model_name_hf = "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"
mixtral_model_name_hf = "mistralai/Mixtral-8x7B-Instruct-v0.1"

phi4_short_name = "phi4"
llama3_short_name = "llama3"
llama4_short_name = "llama4"
mixtral_short_name = "mixtral"

# %%
use_phi4 = True
use_llama3 = False
use_llama4 = False
use_mixtral = False

# %% [markdown]
# ### Configure Model Server
# 
# [RITS](https://rits.fmaas.res.ibm.com/) is a model server for researchers at IBM.

# %%
use_rits = True

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
    model_dict[llama3_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
    model_dict[llama4_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8"
    model_dict[mixtral_model_name] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01"
else:
    default_headers: dict[str, str] = {}
    model_dict: dict[str, str] = {}

def get_base_url(model_name: str)-> str:
    endpoint = model_dict.get(model_name, "http://0.0.0.0:8000")  # fall back to vllm
    return f"{endpoint}/v1"

# %% [markdown]
# ### Configure Seed Data

# %%
# data_name = "20250411_en_2"
# data_name = "20250411_ja"
# data_name = "teigaku-genzei"
# data_name = "teigaku-genzei-ibm-v0"
# data_name = "teigaku-genzei-ibm-v2"
# data_name = "teigaku-genzei-ibm-v3"
data_name = "teigaku-genzei-ibm-v4"

if "20250411_ja" in data_name or "teigaku-genzei" in data_name:
    data_lang = "_ja"
else:
    data_lang = ""

seed_data_name = f"seed_data_{data_name}"
seed_data_path = f"{seed_data_name}.jsonl"

# %%
duplicate_times = 1
# duplicate_times = 5

data_name_duplicate = f"{data_name}-d{duplicate_times}" if duplicate_times > 1 else data_name

# %% [markdown]
# ### Load and Prepare Seed Data
# 
# We'll load our seed data (documents) that will be used to generate question-answer pairs.

# %%
# Load the seed data from JSON file
ds = load_dataset('json', data_files=seed_data_path, split='train')
orig_ds = ds

# %% [markdown]
# (Optional) Sample Seed Data

# %%
# For testing, we'll use just one example
# ds = ds.select(range(1))

# %% [markdown]
# (Optional) Duplicate Seed Data

# %%
if duplicate_times > 1:
    ds = ds.repeat(duplicate_times)

# %% [markdown]
# Add Seed ID

# %%
# Add seed_id column to preserve repetition in seed data
# See https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/42650f1340a2d3576818d68e05508dfe2a8d04bd/src/sdg_hub/checkpointer.py#L103
ds = ds.add_column("seed_id", list(range(len(ds))))

# %%
print(f"Loaded {len(ds)} seed data", flush=True)

# %% [markdown]
# ### Utilities for Generated Data

# %%
def to_messages(generated_data: Dataset) -> Dataset:
    seen = set()
    messages_list: list[dict[str, any]] = []
    for generated_data_i in generated_data:
        user = generated_data_i['question']
        assistant = generated_data_i['response']
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        # NOTE deduplicate messages
        # messages_list.append({"messages": messages})
        key = tuple([frozenset(d.items()) for d in messages])
        if key not in seen:
            seen.add(key)
            messages_list.append({"messages": messages})
    messages_data = Dataset.from_list(messages_list)
    return messages_data

def document_type(generated_data_i: dict[str, any]) -> str:
    dataset_type = generated_data_i.get('dataset_type', None)
    if dataset_type is not None:
        _document_type = f" ({dataset_type})"
    else:
        _document_type = ""
    return _document_type

def print_seed_data(f, generated_data_i) -> None:
    icl_document = generated_data_i.get('icl_document', None)
    if icl_document is not None:
        f.write(f"### ICL example\n\n")
        f.write(f"#### icl_document\n")
        f.write(icl_document + "\n\n")
    icl_query_1 = generated_data_i.get('icl_query_1', None)
    if icl_query_1 is not None:
        f.write(f"#### icl_query_1\n")
        f.write(icl_query_1 + "\n\n")
    icl_response_1 = generated_data_i.get('icl_response_1', None)
    if icl_response_1 is not None:
        f.write(f"#### icl_response_1\n")
        f.write(icl_response_1 + "\n\n")
    icl_query_2 = generated_data_i.get('icl_query_2', None)
    if icl_query_2 is not None:
        f.write(f"#### icl_query_2\n")
        f.write(icl_query_2 + "\n\n")
    icl_response_2 = generated_data_i.get('icl_response_2', None)
    if icl_response_2 is not None:
        f.write(f"#### icl_response_2\n")
        f.write(icl_response_2 + "\n\n")
    icl_query_3 = generated_data_i.get('icl_query_3', None)
    if icl_query_3 is not None:
        f.write(f"#### icl_query_3\n")
        f.write(icl_query_3 + "\n\n")
    icl_response_3 = generated_data_i.get('icl_response_3', None)
    if icl_response_3 is not None:
        f.write(f"#### icl_response_3\n")
        f.write(icl_response_3 + "\n\n")
    document_outline = generated_data_i.get('document_outline', None)
    if document_outline is not None:
        f.write(f"### document_outline\n")
        f.write(document_outline + "\n\n")
    raw_document = generated_data_i.get('raw_document', None)
    if raw_document is not None:
        f.write(f"### raw_document (not used for Q&A generation)\n")
        f.write(raw_document + "\n\n")

def print_generated_data(f, generated_data_i, short_name: str) -> None:
    print_seed_data(f, generated_data_i)
    f.write(f"### document{document_type(generated_data_i)} from {short_name}\n")
    f.write(generated_data_i['document'] + "\n\n")
    f.write(f"### question from {short_name}\n")
    f.write(generated_data_i['question'] + "\n\n")
    f.write(f"### response from {short_name}\n")
    f.write(generated_data_i['response'] + "\n\n")

# %% [markdown]
# ## SDG with phi4

# %% [markdown]
# ### Setting up phi4 Model
# 
# Unless `use_rits` is True and the model is hosted on RITS, we need to host the model using vLLM.
# 
# Start the vLLM server (run in terminal):
# ```bash
# vllm serve ${phi4_model_name_hf} --served-model-name ${phi4_model_name}
# ```

# %%
if use_phi4:
    # Configure OpenAI client
    phi4_base_url = get_base_url(phi4_model_name)

    phi4_client = OpenAI(
        api_key="EMPTY",
        base_url=phi4_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {phi4_model_name}", flush=True)

# %% [markdown]
# ### Configure phi4 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_phi4 and phi4_model_name not in PromptRegistry.get_registry():
    # Load the tokenizer and get the chat template
    # phi4_tokenizer = AutoTokenizer.from_pretrained(phi4_model_name_hf)
    # _phi4_chat_template = phi4_tokenizer.chat_template

    # Copy the chat template
    from sdg_hub.prompts import microsoft_phi_chat_template
    _phi4_chat_template = microsoft_phi_chat_template()

    # Register the chat template
    @PromptRegistry.register(phi4_model_name)
    def phi4_chat_template():
        return _phi4_chat_template

# %% [markdown]
# ### Configure phi4 Pipeline
# 
# Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
# 1. SDG Flow configuration from YAML
# 2. SDG Pipeline setup
# 3. SDG configuration with batch processing, number of workers, and save frequency parameters

# %%
if use_phi4:
    # Load the flow configuration from YAML file
    flow_phi4 = Flow(phi4_client).get_flow_from_file(f"{flow_config}{data_lang}_{phi4_short_name}_rits.yaml")

    # Initialize the SDG pipeline with processing parameters
    sdg_phi4 = SDG(
        [flow_phi4],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )

# %% [markdown]
# ### Generate Data with phi4
# 
# Now we'll use our configured pipeline to generate synthetic question-answer pairs.

# %%
if use_phi4:
    # Generate data and save checkpoints
    generated_data_phi4 = sdg_phi4.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_{phi4_short_name}")

    generated_data_path_phi4 = f"generated_data_{data_name_duplicate}_{timestamp}_{phi4_short_name}.jsonl"
    generated_data_phi4.to_json(generated_data_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_phi4}", flush=True)

    # Save generated data in messages format for training
    messages_data_phi4 = to_messages(generated_data_phi4)

    messages_data_path_phi4 = f"messages_data_{data_name_duplicate}_{timestamp}_{phi4_short_name}.jsonl"
    messages_data_phi4.to_json(messages_data_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_phi4}", flush=True)

# %% [markdown]
# ### Compare Generated Data with phi4

# %%
if use_phi4:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_{phi4_short_name}.md"

    if 'generated_data_phi4' not in locals():
        generated_data_phi4 = []

    with open(model_comparison_path, "w") as f:
        num_generated_data_phi4 = len(generated_data_phi4)

        # Number of examples to compare
        k = num_generated_data_phi4

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_phi4:
                # phi4 results
                generated_data_i = generated_data_phi4[i]
                short_name = phi4_short_name
                print_generated_data(f, generated_data_i, short_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) SDG with llama3

# %% [markdown]
# ### Setting up llama3 Model
# 
# Unless `use_rits` is True and the model is hosted on RITS, we need to host the model using vLLM.
# 
# Start the vLLM server (run in terminal):
# ```bash
# vllm serve ${llama3_model_name_hf} --served-model-name ${llama3_model_name} --tensor-parallel-size 8
# ```

# %%
if use_llama3:
    # Configure OpenAI client
    llama3_base_url = get_base_url(llama3_model_name)

    llama3_client = OpenAI(
        api_key="EMPTY",
        base_url=llama3_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {llama3_model_name}", flush=True)

# %% [markdown]
# ### Configure llama3 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_llama3 and llama3_model_name not in PromptRegistry.get_registry():
    # Load the tokenizer and get the chat template
    # llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_model_name_hf)
    # _llama3_chat_template = llama3_tokenizer.chat_template

    # Copy the chat template
    from sdg_hub.prompts import meta_llama_chat_template
    _llama3_chat_template = meta_llama_chat_template()

    # Register the chat template
    @PromptRegistry.register(llama3_model_name)
    def llama3_chat_template():
        return _llama3_chat_template

# %% [markdown]
# ### Configure llama3 Pipeline
# 
# Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
# 1. SDG Flow configuration from YAML
# 2. SDG Pipeline setup
# 3. SDG configuration with batch processing, number of workers, and save frequency parameters

# %%
if use_llama3:
    # Load the flow configuration from YAML file
    flow_llama3 = Flow(llama3_client).get_flow_from_file(f"{flow_config}{data_lang}_{llama3_short_name}_rits.yaml")

    # Initialize the SDG pipeline with processing parameters
    sdg_llama3 = SDG(
        [flow_llama3],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )

# %% [markdown]
# ### Generate Data with llama3
# 
# Now we'll use our configured pipeline to generate synthetic question-answer pairs.

# %%
if use_llama3:
    # Generate data and save checkpoints
    generated_data_llama3 = sdg_llama3.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_{llama3_short_name}")

    generated_data_path_llama3 = f"generated_data_{data_name_duplicate}_{timestamp}_{llama3_short_name}.jsonl"
    generated_data_llama3.to_json(generated_data_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_llama3}", flush=True)

    # Save generated data in messages format for training
    messages_data_llama3 = to_messages(generated_data_llama3)

    messages_data_path_llama3 = f"messages_data_{data_name_duplicate}_{timestamp}_{llama3_short_name}.jsonl"
    messages_data_llama3.to_json(messages_data_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_llama3}", flush=True)

# %% [markdown]
# ### Compare Generated Data with llama3

# %%
if use_llama3:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_{llama3_short_name}.md"

    if 'generated_data_llama3' not in locals():
        generated_data_llama3 = []

    with open(model_comparison_path, "w") as f:
        num_generated_data_llama3 = len(generated_data_llama3)

        # Number of examples to compare
        k = num_generated_data_llama3

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_llama3:
                # llama3 results
                generated_data_i = generated_data_llama3[i]
                short_name = llama3_short_name
                print_generated_data(f, generated_data_i, short_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) SDG with llama4

# %% [markdown]
# ### Setting up llama4 Model
# 
# Unless `use_rits` is True and the model is hosted on RITS, we need to host the model using vLLM.
# 
# Start the vLLM server (run in terminal):
# ```bash
# vllm serve ${llama4_model_name_hf} --served-model-name ${llama4_model_name} --tensor-parallel-size 8
# ```

# %%
if use_llama4:
    # Configure OpenAI client
    llama4_base_url = get_base_url(llama4_model_name)

    llama4_client = OpenAI(
        api_key="EMPTY",
        base_url=llama4_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {llama4_model_name}", flush=True)

# %% [markdown]
# ### Configure llama4 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_llama4 and llama4_model_name not in PromptRegistry.get_registry():
    # Load the tokenizer and get the chat template
    llama4_tokenizer = AutoTokenizer.from_pretrained(llama4_model_name_hf)
    _llama4_chat_template = llama4_tokenizer.chat_template

    # Register the chat template
    @PromptRegistry.register(llama4_model_name)
    def llama4_chat_template():
        return _llama4_chat_template

# %% [markdown]
# ### Configure llama4 Pipeline
# 
# Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
# 1. SDG Flow configuration from YAML
# 2. SDG Pipeline setup
# 3. SDG configuration with batch processing, number of workers, and save frequency parameters

# %%
if use_llama4:
    # Load the flow configuration from YAML file
    flow_llama4 = Flow(llama4_client).get_flow_from_file(f"{flow_config}{data_lang}_{llama4_short_name}_rits.yaml")

    # Initialize the SDG pipeline with processing parameters
    sdg_llama4 = SDG(
        [flow_llama4],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )

# %% [markdown]
# ### Generate Data with llama4
# 
# Now we'll use our configured pipeline to generate synthetic question-answer pairs.

# %%
if use_llama4:
    # Generate data and save checkpoints
    generated_data_llama4 = sdg_llama4.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_{llama4_short_name}")

    generated_data_path_llama4 = f"generated_data_{data_name_duplicate}_{timestamp}_{llama4_short_name}.jsonl"
    generated_data_llama4.to_json(generated_data_path_llama4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_llama4}", flush=True)

    # Save generated data in messages format for training
    messages_data_llama4 = to_messages(generated_data_llama4)

    messages_data_path_llama4 = f"messages_data_{data_name_duplicate}_{timestamp}_{llama4_short_name}.jsonl"
    messages_data_llama4.to_json(messages_data_path_llama4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_llama4}", flush=True)

# %% [markdown]
# ### Compare Generated Data with llama4

# %%
if use_llama4:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_{llama4_short_name}.md"

    if 'generated_data_llama4' not in locals():
        generated_data_llama4 = []

    with open(model_comparison_path, "w") as f:
        num_generated_data_llama4 = len(generated_data_llama4)

        # Number of examples to compare
        k = num_generated_data_llama4

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_llama4:
                # llama4 results
                generated_data_i = generated_data_llama4[i]
                short_name = llama4_short_name
                print_generated_data(f, generated_data_i, short_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) SDG with mixtral

# %% [markdown]
# ### Setting up mixtral Model
# 
# Unless `use_rits` is True and the model is hosted on RITS, we need to host the model using vLLM.
# 
# Start the vLLM server (run in terminal):
# ```bash
# vllm serve ${mixtral_model_name_hf} --served-model-name ${mixtral_model_name} --tensor-parallel-size 8
# ```

# %%
if use_mixtral:
    # Configure OpenAI client
    mixtral_base_url = get_base_url(mixtral_model_name)

    mixtral_client = OpenAI(
        api_key="EMPTY",
        base_url=mixtral_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {mixtral_model_name}", flush=True)

# %% [markdown]
# ### Configure mixtral Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_mixtral and mixtral_model_name not in PromptRegistry.get_registry():
    # Load the tokenizer and get the chat template
    # mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_model_name_hf)
    # _mixtral_chat_template = mixtral_tokenizer.chat_template

    # Copy the chat template
    from sdg_hub.prompts import mistral_chat_template
    _mixtral_chat_template = mistral_chat_template()

    # Register the chat template
    @PromptRegistry.register(mixtral_model_name)
    def mixtral_chat_template():
        return _mixtral_chat_template

# %% [markdown]
# ### Configure mixtral Pipeline
# 
# Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
# 1. SDG Flow configuration from YAML
# 2. SDG Pipeline setup
# 3. SDG configuration with batch processing, number of workers, and save frequency parameters

# %%
if use_mixtral:
    # Load the flow configuration from YAML file
    flow_mixtral = Flow(mixtral_client).get_flow_from_file(f"{flow_config}{data_lang}_{mixtral_short_name}_rits.yaml")

    # Initialize the SDG pipeline with processing parameters
    sdg_mixtral = SDG(
        [flow_mixtral],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )

# %% [markdown]
# ### Generate Data with mixtral
# 
# Now we'll use our configured pipeline to generate synthetic question-answer pairs.

# %%
if use_mixtral:
    # Generate data and save checkpoints
    generated_data_mixtral = sdg_mixtral.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_{mixtral_short_name}")

    generated_data_path_mixtral = f"generated_data_{data_name_duplicate}_{timestamp}_{mixtral_short_name}.jsonl"
    generated_data_mixtral.to_json(generated_data_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_mixtral}", flush=True)

    # Save generated data in messages format for training
    messages_data_mixtral = to_messages(generated_data_mixtral)

    messages_data_path_mixtral = f"messages_data_{data_name_duplicate}_{timestamp}_{mixtral_short_name}.jsonl"
    messages_data_mixtral.to_json(messages_data_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_mixtral}", flush=True)

# %% [markdown]
# ### Compare Generated Data with mixtral

# %%
if use_mixtral:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_{mixtral_short_name}.md"

    if 'generated_data_mixtral' not in locals():
        generated_data_mixtral = []

    with open(model_comparison_path, "w") as f:
        num_generated_data_mixtral = len(generated_data_mixtral)

        # Number of examples to compare
        k = num_generated_data_mixtral

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_mixtral:
                # mixtral results
                generated_data_i = generated_data_mixtral[i]
                short_name = mixtral_short_name
                print_generated_data(f, generated_data_i, short_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) Compare Generated Data
# 
# Let's compare the outputs from both models by saving them to a markdown file for easy review.

# %%
used_models = 0

if 'generated_data_phi4' not in locals():
    generated_data_phi4 = []
else:
    used_models += 1

if 'generated_data_llama3' not in locals():
    generated_data_llama3 = []
else:
    used_models += 1

if 'generated_data_llama4' not in locals():
    generated_data_llama4 = []
else:
    used_models += 1

if 'generated_data_mixtral' not in locals():
    generated_data_mixtral = []
else:
    used_models += 1

if used_models > 1:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}.md"

    with open(model_comparison_path, "w") as f:
        num_generated_data_phi4 = len(generated_data_phi4)
        num_generated_data_llama3 = len(generated_data_llama3)
        num_generated_data_llama4 = len(generated_data_llama4)
        num_generated_data_mixtral = len(generated_data_mixtral)

        # Number of examples to compare
        k = max(num_generated_data_phi4, num_generated_data_llama3, num_generated_data_llama4, num_generated_data_mixtral)

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_phi4:
                # phi4 results
                generated_data_i = generated_data_phi4[i]
                short_name = phi4_short_name
                print_generated_data(f, generated_data_i, short_name)

            if i < num_generated_data_llama3:
                # llama3 results
                generated_data_i = generated_data_llama3[i]
                short_name = llama3_short_name
                print_generated_data(f, generated_data_i, short_name)

            if i < num_generated_data_llama4:
                # llama4 results
                generated_data_i = generated_data_llama4[i]
                short_name = llama4_short_name
                print_generated_data(f, generated_data_i, short_name)

            if i < num_generated_data_mixtral:
                # mixtral results
                generated_data_i = generated_data_mixtral[i]
                short_name = mixtral_short_name
                print_generated_data(f, generated_data_i, short_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## Production Usage
# For large-scale data generation, export this notebook to a python script and execute it.
# 
# Note: The script `src/sdg_hub/flow_runner.py` doesn't pass `RITS_API_KEY` to the header.


