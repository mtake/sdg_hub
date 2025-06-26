# %% [markdown]
# # Synthetic Data Generation Tutorial using phi4, llama3, and mixtral
# 
# This tutorial demonstrates how to use SDG repository to generate synthetic question-answer pairs from documents using large language models like phi4. We will also generate data using llama3 and mixtral models for comparison. We'll cover:
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
# ## Configurations
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
# from transformers import AutoTokenizer

from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
from sdg_hub.registry import PromptRegistry

# %%
import datetime

now = datetime.datetime.now()
timestamp = now.strftime('%Y%m%d-%H%M%S')

# %%
force_ascii = True  # NOTE this is default
# force_ascii = False

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
# ### Configure [RITS](https://rits.fmaas.res.ibm.com/) Inference Server

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
# NOTE avoid clashes in model_name
model_dict["meta-llama/llama-3-3-70b-instruct"] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
model_dict["microsoft/phi-4"] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4"
model_dict["mistralai/mixtral-8x7B-instruct-v0.1"] = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01"

def get_base_url(model_name: str)-> str:
    endpoint = model_dict.get(model_name, "http://0.0.0.0:8000")  # fall back to vllm
    return f"{endpoint}/v1"

# %% [markdown]
# ### Configure Models

# %%
use_phi4 = True
use_llama3 = False
use_mixtral = False

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
# duplicate_times = 1
duplicate_times = 5

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

def print_generated_data(f, generated_data_i, model_name: str) -> None:
    print_seed_data(f, generated_data_i)
    f.write(f"### Document{document_type(generated_data_i)} from {model_name}\n")
    f.write(generated_data_i['document'] + "\n\n")
    f.write(f"### Result from {model_name}\n")
    f.write(generated_data_i['question'] + "\n")
    f.write("***\n")
    f.write(generated_data_i['response'] + "\n")

# %% [markdown]
# ## SDG with phi4

# %% [markdown]
# ### Setting up phi4 Model

# %%
if use_phi4:
    # Configure OpenAI client
    phi4_teacher_model = "microsoft/phi-4"
    phi4_base_url = get_base_url(phi4_teacher_model)

    phi4_client = OpenAI(
        api_key="EMPTY",
        base_url=phi4_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {phi4_teacher_model}", flush=True)

# %% [markdown]
# ### Configure phi4 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_phi4:
    # Load the tokenizer and get the chat template
    # phi4_teacher_model_hf = "microsoft/phi-4"
    # phi4_tokenizer = AutoTokenizer.from_pretrained(phi4_teacher_model_hf)
    # _phi4_chat_template = phi4_tokenizer.chat_template

    # Hard code the chat template
    _phi4_chat_template = '''{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}'''

    # Register the chat template
    @PromptRegistry.register(phi4_teacher_model)
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
    flow_phi4 = Flow(phi4_client).get_flow_from_file(f"{flow_config}{data_lang}_phi4_rits.yaml")

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
    generated_data_phi4 = sdg_phi4.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_phi4")

    generated_data_path_phi4 = f"generated_data_{data_name_duplicate}_{timestamp}_phi4.jsonl"
    generated_data_phi4.to_json(generated_data_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_phi4}", flush=True)

    # Save generated data in messages format for training
    messages_data_phi4 = to_messages(generated_data_phi4)

    messages_data_path_phi4 = f"messages_data_{data_name_duplicate}_{timestamp}_phi4.jsonl"
    messages_data_phi4.to_json(messages_data_path_phi4, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_phi4}", flush=True)

# %% [markdown]
# ### Compare Generated Data with phi4

# %%
if use_phi4:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_phi4.md"

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
                model_name = "phi4"
                print_generated_data(f, generated_data_i, model_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) SDG with llama3

# %% [markdown]
# ### Setting up llama3 Model

# %%
if use_llama3:
    # Configure OpenAI client
    llama3_teacher_model = "meta-llama/llama-3-3-70b-instruct"
    llama3_base_url = get_base_url(llama3_teacher_model)

    llama3_client = OpenAI(
        api_key="EMPTY",
        base_url=llama3_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {llama3_teacher_model}", flush=True)

# %% [markdown]
# ### Configure llama3 Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_llama3:
    # Load the tokenizer and get the chat template
    # # llama3_teacher_model_hf = "meta-llama/Llama-3.3-70B-Instruct"
    # llama3_teacher_model_hf = "unsloth/Llama-3.3-70B-Instruct"
    # llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_teacher_model_hf)
    # _llama3_chat_template = llama3_tokenizer.chat_template

    # Hard code the chat template
    _llama3_chat_template = '''{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
'''

    # Register the chat template
    @PromptRegistry.register(llama3_teacher_model)
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
    flow_llama3 = Flow(llama3_client).get_flow_from_file(f"{flow_config}{data_lang}_llama3_rits.yaml")

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
    generated_data_llama3 = sdg_llama3.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_llama3")

    generated_data_path_llama3 = f"generated_data_{data_name_duplicate}_{timestamp}_llama3.jsonl"
    generated_data_llama3.to_json(generated_data_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_llama3}", flush=True)

    # Save generated data in messages format for training
    messages_data_llama3 = to_messages(generated_data_llama3)

    messages_data_path_llama3 = f"messages_data_{data_name_duplicate}_{timestamp}_llama3.jsonl"
    messages_data_llama3.to_json(messages_data_path_llama3, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_llama3}", flush=True)

# %% [markdown]
# ### Compare Generated Data with llama3

# %%
if use_llama3:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_llama3.md"

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
                model_name = "llama3"
                print_generated_data(f, generated_data_i, model_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

# %% [markdown]
# ## (Optional) SDG with mixtral

# %% [markdown]
# ### Setting up mixtral Model

# %%
if use_mixtral:
    # Configure OpenAI client
    mixtral_teacher_model = "mistralai/mixtral-8x7B-instruct-v0.1"
    mixtral_base_url = get_base_url(mixtral_teacher_model)

    mixtral_client = OpenAI(
        api_key="EMPTY",
        base_url=mixtral_base_url,
        default_headers=default_headers,
    )

    print(f"Connected to model: {mixtral_teacher_model}", flush=True)

# %% [markdown]
# ### Configure mixtral Prompt Template
# 
# We need to register the correct chat template for our model to ensure proper prompt formatting.

# %%
if use_mixtral:
    # Load the tokenizer and get the chat template
    # mixtral_teacher_model_hf = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_teacher_model_hf)
    # _mixtral_chat_template = mixtral_tokenizer.chat_template

    # Hard code the chat template
    _mixtral_chat_template = '''{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- ' [INST] ' + system_message + '\n\n' + message['content'] + ' [/INST]' }}
        {%- else %}
            {{- ' [INST] ' + message['content'] + ' [/INST]' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}
'''

    # Register the chat template
    @PromptRegistry.register(mixtral_teacher_model)
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
    flow_mixtral = Flow(mixtral_client).get_flow_from_file(f"{flow_config}{data_lang}_mixtral_rits.yaml")

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
    generated_data_mixtral = sdg_mixtral.generate(ds, checkpoint_dir=f"Tmp_{data_name_duplicate}_mixtral")

    generated_data_path_mixtral = f"generated_data_{data_name_duplicate}_{timestamp}_mixtral.jsonl"
    generated_data_mixtral.to_json(generated_data_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Data saved to {generated_data_path_mixtral}", flush=True)

    # Save generated data in messages format for training
    messages_data_mixtral = to_messages(generated_data_mixtral)

    messages_data_path_mixtral = f"messages_data_{data_name_duplicate}_{timestamp}_mixtral.jsonl"
    messages_data_mixtral.to_json(messages_data_path_mixtral, orient="records", lines=True, force_ascii=force_ascii)
    print(f"Messages data saved to {messages_data_path_mixtral}", flush=True)

# %% [markdown]
# ### Compare Generated Data with mixtral

# %%
if use_mixtral:
    # Save comparison results to markdown file
    model_comparison_path = f"model_comparison_{data_name_duplicate}_{timestamp}_mixtral.md"

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
                model_name = "mixtral"
                print_generated_data(f, generated_data_i, model_name)

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
        num_generated_data_mixtral = len(generated_data_mixtral)

        # Number of examples to compare
        k = max(num_generated_data_phi4, num_generated_data_llama3, num_generated_data_mixtral)

        # Compare generated Q&A pairs
        for i in range(k):
            f.write(f"# Example #{i+1}\n\n")

            if i < num_generated_data_phi4:
                # phi4 results
                generated_data_i = generated_data_phi4[i]
                model_name = "phi4"
                print_generated_data(f, generated_data_i, model_name)

            if i < num_generated_data_llama3:
                # llama3 results
                generated_data_i = generated_data_llama3[i]
                model_name = "llama3"
                print_generated_data(f, generated_data_i, model_name)

            if i < num_generated_data_mixtral:
                # mixtral results
                generated_data_i = generated_data_mixtral[i]
                model_name = "mixtral"
                print_generated_data(f, generated_data_i, model_name)

            f.write("\n")

    print(f"Wrote {k} examples to {model_comparison_path}", flush=True)

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


