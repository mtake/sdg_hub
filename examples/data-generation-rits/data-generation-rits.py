# %% [markdown]
# # Synthetic Data Generation Tutorial using Phi-4, LLaMA, and Mixtral on RITS
# 
# This tutorial demonstrates how to use SDG repository to generate synthetic question-answer pairs from documents using large language models like Phi-4 and LLaMA 3.3 70B. We will also generate data using Mixtral model for comparison. We'll cover:
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
# pip install sdg-hub==0.1.0a2
# ```

# %%
# %%capture
# %pip install transformers
# %pip install protobuf sentencepiece  # for Mixtral-8x22B-Instruct-v0.1

# %%
# Import required libraries
# datasets: For handling our data
# OpenAI: For interfacing with the LLM servers
# SDG components: For building our data generation pipeline
from datasets import load_dataset, Dataset
from openai import OpenAI
from transformers import AutoTokenizer

from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.registry import PromptRegistry

# %%
import datetime

now = datetime.datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S')

# %%
force_ascii = True  # NOTE this is default
# force_ascii = False

# %%
sample_seed_data = False  # For production
# sample_seed_data = True  # For test

MAX_SEED_DATA = 1
# MAX_SEED_DATA = 3

# %% [markdown]
# ### Configure Parallelism

# %%
# For production
num_workers = 8   # Number of parallel workers
batch_size = 8    # Batch size for processing
save_freq = 1000  # How often to save checkpoints

# For test
# num_workers = 1   # Number of parallel workers
# batch_size = 1    # Batch size for processing
# save_freq = 1000  # How often to save checkpoints

# %% [markdown]
# ### Setup environments for [RITS](https://rits.fmaas.res.ibm.com/)

# %%
import os
import requests

RITS_API_KEY = os.getenv("RITS_API_KEY")
# print(f"RITS_API_KEY={RITS_API_KEY}", flush=True)

default_headers = {"RITS_API_KEY": RITS_API_KEY}

url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
res = requests.get(url=url, headers=default_headers)
assert res.status_code == 200
model_list: list[dict[str, str]] = res.json()
model_dict = { m["model_name"]: m["endpoint"] for m in model_list }

def get_base_url(model_name: str)-> str:
    endpoint = model_dict[model_name]
    return f"{endpoint}/v1"

# %% [markdown]
# ### Configure Seed Data

# %%
# data_name = "samples"
# data_name = "20250411_en_2"
# data_name = "20250411_ja"
# data_name = "20250411_ja_non_ascii"
data_name = "teigaku-genzei"

seed_data_name = f"seed_data_{data_name}"
seed_data_path = f"{seed_data_name}.jsonl"

# import pandas as pd
# df = pd.read_json(seed_data_path, orient='records', lines=True)
# seed_data_path_non_ascii = f"{seed_data_name}_non_ascii.jsonl"
# df.to_json(seed_data_path_non_ascii, orient='records', lines=True, force_ascii=False)

# %% [markdown]
# ### (Optional) Create Seed Data from a [Test Case](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/a3f788bcc36702ef09bfee4be6e569d77ea8a20b/scripts/test_knowledge.py#L25)

# %%
# samples = [
#     {
#         "icl_query_1": "what is the location of the tubal tonsils?",
#         "icl_response_1": "The location of the tubal tonsils is the roof of the pharynx.",
#         "icl_query_2": "How long does the adenoid grow?",
#         "task_description": "Teaching about human anatomy, specifically tonsils",
#         "icl_response_2": "The adenoid grows until the age of 5, starts to shrink at the age of 7 and becomes small in adulthood.",
#         "icl_query_3": "What is the immune systems first line of defense against ingested or inhaled foreign pathogens?",
#         "icl_response_3": "The tonsils are the immune systems first line of defense.",
#         "document": "The **tonsils** are a set of lymphoid organs facing into the aerodigestive tract, which is known as Waldeyer's tonsillar ring and consists of the adenoid tonsil or pharyngeal tonsil, two tubal tonsils, two palatine tonsils, and the lingual tonsils. These organs play an important role in the immune system. When used unqualified, the term most commonly refers specifically to the palatine tonsils, which are two lymphoid organs situated at either side of the back of the human throat. The palatine tonsils and the adenoid tonsil are organs consisting of lymphoepithelial tissue located near the oropharynx and nasopharynx parts of the throat",
#         "domain": "textbook",
#     }
# ]

# ds = Dataset.from_list(samples)
# ds.to_json(seed_data_path, orient="records", lines=True)

# %% [markdown]
# ### Load and Prepare Seed Data
# 
# We'll load our seed data (documents) that will be used to generate question-answer pairs.

# %%
# Load the seed data from JSON file
ds = load_dataset('json', data_files=seed_data_path, split='train')

# %% [markdown]
# ### (Optional) Reduce Seed Data for Testing

# %%
if sample_seed_data:
    num_seed_data = len(ds)
    num_seed_data = min(num_seed_data, MAX_SEED_DATA)

    ds = ds.select(range(num_seed_data))

# %% [markdown]
# ### Utilities for Generated Data

# %%
def to_messages(generated_data: Dataset) -> Dataset:
    messages_list: list[dict[str, any]] = []
    for generated_datum in generated_data:
        user = generated_datum['question']
        assistant = generated_datum['response']
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        messages_list.append({"messages": messages})
    messages_data = Dataset.from_list(messages_list)
    return messages_data

def get_dataset_type(generated_data_i: dict[str, any]) -> str:
    _dataset_type = generated_data_i.get('dataset_type', None)
    if _dataset_type is not None:
        _dataset_type = f" ({_dataset_type})"
    else:
        _dataset_type = ""
    return _dataset_type

def write_input(f, generated_data_i) -> None:
    icl_document = generated_data_i.get('icl_document', None)
    if icl_document is not None:
        f.write(f"### In-Context Learning Example\n\n")
        f.write(f"#### ICL Document\n")
        f.write(icl_document + "\n\n")
    icl_query_1 = generated_data_i.get('icl_query_1', None)
    if icl_query_1 is not None:
        f.write(f"#### ICL Query 1\n")
        f.write(icl_query_1 + "\n\n")
    icl_response_1 = generated_data_i.get('icl_response_1', None)
    if icl_response_1 is not None:
        f.write(f"#### ICL Response 1\n")
        f.write(icl_response_1 + "\n\n")
    icl_query_2 = generated_data_i.get('icl_query_2', None)
    if icl_query_2 is not None:
        f.write(f"#### ICL Query 2\n")
        f.write(icl_query_2 + "\n\n")
    icl_response_2 = generated_data_i.get('icl_response_2', None)
    if icl_response_2 is not None:
        f.write(f"#### ICL Response 2\n")
        f.write(icl_response_2 + "\n\n")
    icl_query_3 = generated_data_i.get('icl_query_3', None)
    if icl_query_3 is not None:
        f.write(f"#### ICL Query 3\n")
        f.write(icl_query_3 + "\n\n")
    icl_response_3 = generated_data_i.get('icl_response_3', None)
    if icl_response_3 is not None:
        f.write(f"#### ICL Response 3\n")
        f.write(icl_response_3 + "\n\n")
    document_outline = generated_data_i.get('document_outline', None)
    if document_outline is not None:
        f.write(f"### Document Outline\n")
        f.write(document_outline + "\n\n")
    raw_document = generated_data_i.get('raw_document', None)
    if raw_document is not None:
        f.write(f"### Raw Document (not used for Q&A generation)\n")
        f.write(raw_document + "\n\n")

# %% [markdown]
# ### Select Models

# %%
generate_data_with_phi4 = True
generate_data_with_llama3 = False
generate_data_with_mixtral = False
generate_data_with_mixtral8x22b = False

# %% [markdown]
# ## SDG with Phi-4 Model

# %% [markdown]
# ### Setting up Phi-4 Model

# %%
# Connect to Phi-4 model running on RITS
phi4_teacher_model = "microsoft/phi-4"
phi4_endpoint = get_base_url(phi4_teacher_model)

phi4_client = OpenAI(
    api_key="EMPTY",
    base_url=phi4_endpoint,
    default_headers=default_headers,
)

# Verify connection to Phi-4 model
print(f"Connected to Phi-4 model: {phi4_teacher_model}", flush=True)

# %% [markdown]
# ### Configure Phi-4 Prompt Template

# %%
# Register the Phi-4 chat template
# This ensures proper formatting of prompts for the model

phi4_teacher_model_hf = "microsoft/phi-4"

# Load the tokenizer to get the chat template
phi4_tokenizer = AutoTokenizer.from_pretrained(phi4_teacher_model_hf)

# Register the chat template in our prompt registry
@PromptRegistry.register(phi4_teacher_model)
def phi4_chat_template():
    return phi4_tokenizer.chat_template

# %% [markdown]
# ### Configure Phi-4 Pipeline

# %%
# Create flow configuration for Phi-4
flow_cfg_phi4 = Flow(phi4_client).get_flow_from_file("synth_knowledge1.5_phi4_rits.yaml")

# Initialize SDG pipeline for Phi-4
sdg_phi4 = SDG(
    [Pipeline(flow_cfg_phi4)],
    num_workers=num_workers,
    batch_size=batch_size,
    save_freq=save_freq,
)

# %% [markdown]
# ### Generate Data with Phi-4

# %%
if generate_data_with_phi4:
    # Generate data using Phi-4 model
    generated_data_phi4 = sdg_phi4.generate(ds, checkpoint_dir="Tmp-checkpoint")

    generated_path_phi4 = f"generated_data_{data_name}_{timestamp}_phi4.jsonl"
    generated_data_phi4.to_json(generated_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_path_phi4}", flush=True)

    # Save generated data in messages format for training
    messages_data_phi4 = to_messages(generated_data_phi4)

    messages_data_path_phi4 = f"messages_data_{data_name}_{timestamp}_phi4.jsonl"
    messages_data_phi4.to_json(messages_data_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_phi4}", flush=True)

# %% [markdown]
# ### Output Generated Data with Phi-4

# %%
if generate_data_with_phi4:
    # Save comparison results to markdown file
    output_file = f"model_output_{data_name}_{timestamp}_phi4.md"

    if 'generated_data_phi4' not in locals():
        generated_data_phi4 = []

    with open(output_file, "w") as f:
        num_generated_data_phi4 = len(generated_data_phi4)

        # Number of examples to compare
        k = num_generated_data_phi4

        # Compare generated Q&A pairs
        for i in range(k):
            f.write("# Example #{}\n\n".format(i+1))

            if i < num_generated_data_phi4:
                # Phi-4 results
                write_input(f, generated_data_phi4[i])
                f.write(f"### Document{get_dataset_type(generated_data_phi4[i])} from phi-4\n")
                f.write(generated_data_phi4[i]['document'] + "\n\n")
                f.write("### Result from phi-4\n")
                f.write(generated_data_phi4[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data_phi4[i]['response'] + "\n")

            f.write("\n")

    print(f"Wrote {k} examples to {output_file}", flush=True)

# %% [markdown]
# ## (Optional) SDG with LLaMA 3.3 70B Model

# %% [markdown]
# ### Setting up LLaMA 3.3 70B Model

# %%
# Configure OpenAI client to connect to RITS server
llama3_teacher_model = "meta-llama/llama-3-3-70b-instruct"
llama3_endpoint = get_base_url(llama3_teacher_model)

llama3_client = OpenAI(
    api_key="EMPTY",
    base_url=llama3_endpoint,
    default_headers=default_headers,
)

print(f"Connected to Llama-3.3 model: {llama3_teacher_model}", flush=True)

# %% [markdown]
# ### Configure LLaMA 3.3 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
# Register the LLaMA 3.3 chat template
# This ensures proper formatting of prompts for the model

# llama3_teacher_model_hf = "meta-llama/Llama-3.3-70B-Instruct"
llama3_teacher_model_hf = "unsloth/Llama-3.3-70B-Instruct"

# Load the tokenizer to get the chat template
llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_teacher_model_hf)

# Register the chat template in our prompt registry
@PromptRegistry.register(llama3_teacher_model)
def llama3_chat_template():
    return llama3_tokenizer.chat_template

# %% [markdown]
# ### Configure the Data Generation Pipeline
# 
# Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
# 1. SDG Flow configuration from YAML
# 2. SDG Pipeline setup
# 3. SDG configuration with batch processing, number of workers, and save frequency parameters

# %%
# Load the flow configuration from YAML file
flow_cfg_llama3 = Flow(llama3_client).get_flow_from_file("synth_knowledge1.5_llama3.3_rits.yaml")

# Initialize the SDG pipeline with processing parameters
sdg_llama3 = SDG(
    [Pipeline(flow_cfg_llama3)],
    num_workers=num_workers,
    batch_size=batch_size,
    save_freq=save_freq,
)

# %% [markdown]
# ### Generate Data with LLaMA 3.3
# 
# Now we'll use our configured pipeline to generate synthetic question-answer pairs.

# %%
if generate_data_with_llama3:
    # Generate synthetic data and save checkpoints
    generated_data_llama3 = sdg_llama3.generate(ds, checkpoint_dir="Tmp-checkpoint")

    generated_path_llama3 = f"generated_data_{data_name}_{timestamp}_llama3.jsonl"
    generated_data_llama3.to_json(generated_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_path_llama3}", flush=True)

    # Save generated data in messages format for training
    messages_data_llama3 = to_messages(generated_data_llama3)

    messages_data_path_llama3 = f"messages_data_{data_name}_{timestamp}_llama3.jsonl"
    messages_data_llama3.to_json(messages_data_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_llama3}", flush=True)

# %% [markdown]
# ### Output Generated Data with LLaMA 3.3

# %%
if generate_data_with_llama3:
    # Save comparison results to markdown file
    output_file = f"model_output_{data_name}_{timestamp}_llama3.md"

    if 'generated_data_llama3' not in locals():
        generated_data_llama3 = []

    with open(output_file, "w") as f:
        num_generated_data_llama3 = len(generated_data_llama3)

        # Number of examples to compare
        k = num_generated_data_llama3

        # Compare generated Q&A pairs
        for i in range(k):
            f.write("# Example #{}\n\n".format(i+1))

            if i < num_generated_data_llama3:
                # LLaMA 3.3 results
                write_input(f, generated_data_llama3[i])
                f.write(f"### Document{get_dataset_type(generated_data_llama3[i])} from llama-3.3-70b\n")
                f.write(generated_data_llama3[i]['document'] + "\n\n")
                f.write("### Result from llama-3.3-70b\n")
                f.write(generated_data_llama3[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data_llama3[i]['response'] + "\n")

            f.write("\n")

    print(f"Wrote {k} examples to {output_file}", flush=True)

# %% [markdown]
# ## (Optional) SDG with Mixtral-8x7B Model

# %% [markdown]
# ### Setting up Mixtral-8x7B Model
# 
# For comparison, we'll also generate data using the Mixtral model.

# %%
# Connect to Mixtral model running on RITS
mixtral_teacher_model = "mistralai/mixtral-8x7B-instruct-v0.1"
mixtral_endpoint = get_base_url(mixtral_teacher_model)

mixtral_client = OpenAI(
    api_key="EMPTY",
    base_url=mixtral_endpoint,
    default_headers=default_headers,
)

# Verify connection to Mixtral model
print(f"Connected to Mixtral model: {mixtral_teacher_model}", flush=True)

# %% [markdown]
# ### Configure Mixtral-8x7B Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
# Register the Mixtral chat template
# This ensures proper formatting of prompts for the model

mixtral_teacher_model_hf = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the tokenizer to get the chat template
mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_teacher_model_hf)

# Register the chat template in our prompt registry
@PromptRegistry.register(mixtral_teacher_model)
def mixtral_chat_template():
    return mixtral_tokenizer.chat_template

# %% [markdown]
# ### Configure Mixtral-8x7B Pipeline
# 
# Set up a similar pipeline for Mixtral model generation.

# %%
# Create flow configuration for Mixtral
flow_cfg_mixtral = Flow(mixtral_client).get_flow_from_file("synth_knowledge1.5_mixtral_rits.yaml")

# Initialize SDG pipeline for Mixtral
sdg_mixtral = SDG(
    [Pipeline(flow_cfg_mixtral)],
    num_workers=num_workers,
    batch_size=batch_size,
    save_freq=save_freq,
)

# %% [markdown]
# ### Generate Data with Mixtral-8x7B
# 
# Generate synthetic data using the Mixtral model for comparison.

# %%
if generate_data_with_mixtral:
    # Generate data using Mixtral model
    generated_data_mixtral = sdg_mixtral.generate(ds, checkpoint_dir="Tmp-checkpoint")

    generated_path_mixtral = f"generated_data_{data_name}_{timestamp}_mixtral.jsonl"
    generated_data_mixtral.to_json(generated_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_path_mixtral}", flush=True)

    # Save generated data in messages format for training
    messages_data_mixtral = to_messages(generated_data_mixtral)

    messages_data_path_mixtral = f"messages_data_{data_name}_{timestamp}_mixtral.jsonl"
    messages_data_mixtral.to_json(messages_data_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_mixtral}", flush=True)

# %% [markdown]
# ### Output Generated Data with Mixtral-8x7B

# %%
if generate_data_with_mixtral:
    # Save comparison results to markdown file
    output_file = f"model_output_{data_name}_{timestamp}_mixtral.md"

    if 'generated_data_mixtral' not in locals():
        generated_data_mixtral = []

    with open(output_file, "w") as f:
        num_generated_data_mixtral = len(generated_data_mixtral)

        # Number of examples to compare
        k = num_generated_data_mixtral

        # Compare generated Q&A pairs
        for i in range(k):
            f.write("# Example #{}\n\n".format(i+1))

            if i < num_generated_data_mixtral:
                # Mixtral-8x7B results
                write_input(f, generated_data_mixtral[i])
                f.write(f"### Document{get_dataset_type(generated_data_mixtral[i])} from mixtral-8x7B\n")
                f.write(generated_data_mixtral[i]['document'] + "\n\n")
                f.write("### Result from mixtral-8x7B\n")
                f.write(generated_data_mixtral[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data_mixtral[i]['response'] + "\n")

            f.write("\n")

    print(f"Wrote {k} examples to {output_file}", flush=True)

# %% [markdown]
# ## (Optional) SDG with Mixtral-8x22B Model

# %% [markdown]
# ### Setting up Mixtral-8x22B Model
# 
# For comparison, we'll also generate data using the Mixtral model.

# %%
# Connect to Mixtral model running on RITS
mixtral8x22b_teacher_model = "mistralai/mixtral-8x22B-instruct-v0.1"
mixtral8x22b_endpoint = get_base_url(mixtral8x22b_teacher_model)

mixtral8x22b_client = OpenAI(
    api_key="EMPTY",
    base_url=mixtral8x22b_endpoint,
    default_headers=default_headers,
)

# Verify connection to Mixtral model
print(f"Connected to Mixtral model: {mixtral8x22b_teacher_model}", flush=True)

# %% [markdown]
# ### Configure Mixtral-8x22B Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
# Register the Mixtral chat template
# This ensures proper formatting of prompts for the model

mixtral8x22b_teacher_model_hf = "mistralai/Mixtral-8x22B-Instruct-v0.1"

# Load the tokenizer to get the chat template
mixtral8x22b_tokenizer = AutoTokenizer.from_pretrained(mixtral8x22b_teacher_model_hf)

# Register the chat template in our prompt registry
@PromptRegistry.register(mixtral8x22b_teacher_model)
def mixtral8x22b_chat_template():
    return mixtral8x22b_tokenizer.chat_template

# %% [markdown]
# ### Configure Mixtral-8x22B Pipeline
# 
# Set up a similar pipeline for Mixtral model generation.

# %%
# Create flow configuration for Mixtral
flow_cfg_mixtral8x22b = Flow(mixtral8x22b_client).get_flow_from_file("synth_knowledge1.5_mixtral8x22b_rits.yaml")

# Initialize SDG pipeline for Mixtral
sdg_mixtral8x22b = SDG(
    [Pipeline(flow_cfg_mixtral8x22b)],
    num_workers=num_workers,
    batch_size=batch_size,
    save_freq=save_freq,
)

# %% [markdown]
# ### Generate Data with Mixtral-8x22B
# 
# Generate synthetic data using the Mixtral model for comparison.

# %%
if generate_data_with_mixtral8x22b:
    # Generate data using Mixtral model
    generated_data_mixtral8x22b = sdg_mixtral8x22b.generate(ds, checkpoint_dir="Tmp-checkpoint")

    generated_path_mixtral8x22b = f"generated_data_{data_name}_{timestamp}_mixtral8x22b.jsonl"
    generated_data_mixtral8x22b.to_json(generated_path_mixtral8x22b, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_path_mixtral8x22b}", flush=True)

    # Save generated data in messages format for training
    messages_data_mixtral8x22b = to_messages(generated_data_mixtral8x22b)

    messages_data_path_mixtral8x22b = f"messages_data_{data_name}_{timestamp}_mixtral8x22b.jsonl"
    messages_data_mixtral8x22b.to_json(messages_data_path_mixtral8x22b, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_mixtral8x22b}", flush=True)

# %% [markdown]
# ### Output Generated Data with Mixtral-8x22B

# %%
if generate_data_with_mixtral8x22b:
    # Save comparison results to markdown file
    output_file = f"model_output_{data_name}_{timestamp}_mixtral8x22b.md"

    if 'generated_data_mixtral8x22b' not in locals():
        generated_data_mixtral8x22b = []

    with open(output_file, "w") as f:
        num_generated_data_mixtral8x22b = len(generated_data_mixtral8x22b)

        # Number of examples to compare
        k = num_generated_data_mixtral8x22b

        # Compare generated Q&A pairs
        for i in range(k):
            f.write("# Example #{}\n\n".format(i+1))

            if i < num_generated_data_mixtral8x22b:
                # Mixtral-8x22B results
                write_input(f, generated_data_mixtral8x22b[i])
                f.write(f"### Document{get_dataset_type(generated_data_mixtral8x22b[i])} from mixtral-8x22B\n")
                f.write(generated_data_mixtral8x22b[i]['document'] + "\n\n")
                f.write("### Result from mixtral-8x22B\n")
                f.write(generated_data_mixtral8x22b[i]['question'] + "\n")
                f.write("*******************************\n")
                f.write(generated_data_mixtral8x22b[i]['response'] + "\n")

            f.write("\n")

    print(f"Wrote {k} examples to {output_file}", flush=True)

# %% [markdown]
# ## Compare Generated Data
# 
# Let's compare the outputs from both models by saving them to a markdown file for easy review.

# %%
# Save comparison results to markdown file
output_file = f"model_comparison_{data_name}_{timestamp}.md"

if 'generated_data_phi4' not in locals():
    generated_data_phi4 = []

if 'generated_data_llama3' not in locals():
    generated_data_llama3 = []

if 'generated_data_mixtral' not in locals():
    generated_data_mixtral = []

if 'generated_data_mixtral8x22b' not in locals():
    generated_data_mixtral8x22b = []

with open(output_file, "w") as f:
    num_generated_data_phi4 = len(generated_data_phi4)
    num_generated_data_llama3 = len(generated_data_llama3)
    num_generated_data_mixtral = len(generated_data_mixtral)
    num_generated_data_mixtral8x22b = len(generated_data_mixtral8x22b)

    # Number of examples to compare
    k = max(num_generated_data_phi4, num_generated_data_llama3, num_generated_data_mixtral, num_generated_data_mixtral8x22b)

    # Compare generated Q&A pairs
    for i in range(k):
        f.write("# Example #{}\n\n".format(i+1))

        if i < num_generated_data_phi4:
            # Phi-4 results
            write_input(f, generated_data_phi4[i])
            f.write(f"### Document{get_dataset_type(generated_data_phi4[i])} from phi-4\n")
            f.write(generated_data_phi4[i]['document'] + "\n\n")
            f.write("### Result from phi-4\n")
            f.write(generated_data_phi4[i]['question'] + "\n")
            f.write("*******************************\n")
            f.write(generated_data_phi4[i]['response'] + "\n")

        if i < num_generated_data_llama3:
            # LLaMA 3.3 results
            write_input(f, generated_data_llama3[i])
            f.write(f"### Document{get_dataset_type(generated_data_llama3[i])} from llama-3.3-70b\n")
            f.write(generated_data_llama3[i]['document'] + "\n\n")
            f.write("### Result from llama-3.3-70b\n")
            f.write(generated_data_llama3[i]['question'] + "\n")
            f.write("*******************************\n")
            f.write(generated_data_llama3[i]['response'] + "\n")

        if i < num_generated_data_mixtral:
            # Mixtral-8x7B results
            write_input(f, generated_data_mixtral[i])
            f.write(f"### Document{get_dataset_type(generated_data_mixtral[i])} from mixtral-8x7B\n")
            f.write(generated_data_mixtral[i]['document'] + "\n\n")
            f.write("### Result from mixtral-8x7B\n")
            f.write(generated_data_mixtral[i]['question'] + "\n")
            f.write("*******************************\n")
            f.write(generated_data_mixtral[i]['response'] + "\n")

        if i < num_generated_data_mixtral8x22b:
            # Mixtral-8x22B results
            write_input(f, generated_data_mixtral8x22b[i])
            f.write(f"### Document{get_dataset_type(generated_data_mixtral8x22b[i])} from mixtral-8x22B\n")
            f.write(generated_data_mixtral8x22b[i]['document'] + "\n\n")
            f.write("### Result from mixtral-8x22B\n")
            f.write(generated_data_mixtral8x22b[i]['question'] + "\n")
            f.write("*******************************\n")
            f.write(generated_data_mixtral8x22b[i]['response'] + "\n")

        f.write("\n")

print(f"Wrote {k} examples to {output_file}", flush=True)

# %% [markdown]
# ## Production Usage
# 
# For large-scale data generation, use the command-line script instead of this notebook:
# 
# ```bash
# python scripts/generate.py --ds_path seed_data.jsonl \
#     --bs 2 --num_workers 10 \
#     --save_path <your_save_path> \
#     --flow ../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml \
#     --checkpoint_dir <your_checkpoint_dir> \
#     --endpoint <your_endpoint>
# ```
# 
# Note: For LLaMA 3.3, use `synth_knowledge1.5_llama3.3.yaml` as the flow configuration file.


