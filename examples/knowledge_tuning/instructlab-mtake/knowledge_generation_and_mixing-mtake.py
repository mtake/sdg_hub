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
# data_name = "teigaku-genzei"
# data_name = "teigaku-genzei-ibm-v0"
# data_name = "teigaku-genzei-ibm-v2"
# data_name = "teigaku-genzei-ibm-v3"
# data_name = "teigaku-genzei-ibm-v4"
# data_name = "teigaku-genzei-ibm-v5"
data_name = "teigaku-genzei-ibm-v6"
# data_name = "akita-medical"
# data_name = "ibm-newsroom"
# data_name = "ibm-newsroom-en"
# data_name = "jfe-technical-report"

if data_name.endswith(("_ja", "-ja")):
    data_lang = "ja"
elif data_name.endswith(("_en", "-en")):
    data_lang = ""
elif data_name.startswith(("teigaku-genzei", "akita-medical", "ibm-newsroom", "jfe-technical-report")):
    data_lang = "ja"
else:
    data_lang = ""

repeat_times = 1
# repeat_times = 5

# %%
sdg_demo_output = "sdg_demo_output"

_data_name = f"_{data_name}" if data_name is not None and len(data_name) > 0 else ""
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

# %%
# We will use the "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning" flow.
# For loading the flow simply use the fullname to load it
# @@@ahoaho XXX
# flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
if data_lang == "ja":
    flow_name = "Advanced Japanese Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
else:
    flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# %% [markdown]
# #### Configure processing mode

# %%
import os

async_mode = True  # original
# async_mode = False  # single worker

DEFAULT_TIMEOUT = 3600
timeout_str = os.getenv("TIMEOUT", str(DEFAULT_TIMEOUT))
try:
    timeout = int(timeout_str)
except ValueError:
    print(f"WARNING: unsupported timeout value: {timeout_str}. fall back to {DEFAULT_TIMEOUT}")
    timeout = DEFAULT_TIMEOUT

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
phi4test_model_name_hf = "microsoft/phi-4"
gptoss_model_name_hf = "openai/gpt-oss-120b"
# llama3_model_name_hf = "meta-llama/Llama-3.3-70B-Instruct"
llama3_model_name_hf = "unsloth/Llama-3.3-70B-Instruct"
# llama4_model_name_hf = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
llama4_model_name_hf = "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"
# mistral_model_name_hf = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
mistral_model_name_hf = "mistralai/Mistral-Large-3-675B-Instruct-2512"
mixtral_model_name_hf = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if use_rits:
    # Served model name (RITS)
    phi4_model_name = "hosted_vllm/microsoft/phi-4"
    phi4test_model_name = "hosted_vllm/microsoft/phi-4"
    gptoss_model_name = "hosted_vllm/openai/gpt-oss-120b"
    llama3_model_name = "hosted_vllm/meta-llama/llama-3-3-70b-instruct"
    llama4_model_name = "hosted_vllm/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    # mistral_model_name = "hosted_vllm/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    mistral_model_name = "hosted_vllm/mistralai/Mistral-Large-3-675B-Instruct-2512"
    mixtral_model_name = "hosted_vllm/mistralai/mixtral-8x7B-instruct-v0.1"
    # Model inference endpoint (RITS)
    phi4_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4"
    phi4test_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4-test"
    gptoss_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b"
    llama3_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
    llama4_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8"
    # mistral_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-small-3-2-24b-2506"
    mistral_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-large-3-675b-2512"
    mixtral_model_endpoint = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01"
else:
    # Served model name (self hosting via vLLM)
    phi4_model_name = f"hosted_vllm/{phi4_model_name_hf}"
    phi4test_model_name = f"hosted_vllm/{phi4test_model_name_hf}"
    gptoss_model_name = f"hosted_vllm/{gptoss_model_name_hf}"
    llama3_model_name = f"hosted_vllm/{llama3_model_name_hf}"
    llama4_model_name = f"hosted_vllm/{llama4_model_name_hf}"
    mistral_model_name = f"hosted_vllm/{mistral_model_name_hf}"
    mixtral_model_name = f"hosted_vllm/{mixtral_model_name_hf}"
    # Model inference endpoint (self hosting via vLLM)
    phi4_model_endpoint = "http://0.0.0.0:8000"
    phi4test_model_endpoint = "http://0.0.0.0:8000"
    gptoss_model_endpoint = "http://0.0.0.0:8000"
    llama3_model_endpoint = "http://0.0.0.0:8000"
    llama4_model_endpoint = "http://0.0.0.0:8000"
    mistral_model_endpoint = "http://0.0.0.0:8000"
    mixtral_model_endpoint = "http://0.0.0.0:8000"

# Model short name
phi4_short_name = "phi4"
phi4test_short_name = "phi4test"
gptoss_short_name = "gptoss"
llama3_short_name = "llama3"
llama4_short_name = "llama4"
mistral_short_name = "mistral"
mixtral_short_name = "mixtral"

# %%
# @@@ahoaho XXX
DEFAULT_MODEL = phi4_short_name
# DEFAULT_MODEL = mistral_short_name
# DEFAULT_MODEL = gptoss_short_name  # FlowValidationError: Block 'parse_atomic_facts' execution failed: Block 'parse_atomic_facts' produced empty dataset

short_name = os.getenv("MODEL", DEFAULT_MODEL)

if short_name not in (phi4_short_name, phi4test_short_name, gptoss_short_name, llama3_short_name, llama4_short_name, mistral_short_name, mixtral_short_name):
    print(f"WARNING: unknown model: {short_name}. fall back to {DEFAULT_MODEL}")
    short_name = DEFAULT_MODEL

# %%
if short_name == phi4_short_name:
    model_name = phi4_model_name
    model_endpoint = phi4_model_endpoint
elif short_name == phi4test_short_name:
    model_name = phi4test_model_name
    model_endpoint = phi4test_model_endpoint
elif short_name == gptoss_short_name:
    model_name = gptoss_model_name
    model_endpoint = gptoss_model_endpoint
elif short_name == llama3_short_name:
    model_name = llama3_model_name
    model_endpoint = llama3_model_endpoint
elif short_name == llama4_short_name:
    model_name = llama4_model_name
    model_endpoint = llama4_model_endpoint
elif short_name == mistral_short_name:
    model_name = mistral_model_name
    model_endpoint = mistral_model_endpoint
elif short_name == mixtral_short_name:
    model_name = mixtral_model_name
    model_endpoint = mixtral_model_endpoint
else:
    raise ValueError(f"Invalid short_name: {short_name}")

# @@@ahoaho XXX
# output_dir = f"{output_dir_prefix}_{short_name}"  # for continued execution after failure
output_dir = f"{output_dir_prefix}_{short_name}_{timestamp}"

# %%
extra_headers: dict[str, any] = {}

if use_rits:
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    extra_headers["RITS_API_KEY"] = RITS_API_KEY

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
    api_base=f"{model_endpoint}/v1",
    api_key="EMPTY",
    async_mode=async_mode,
    timeout=timeout,
    extra_headers=extra_headers,
)

# %% [markdown]
# #### Load and prepare seed data

# %%
sample_size = 2

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

# See https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/docs/blocks/llm-blocks.md#async-processing--concurrency-control

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

# Error teigaku-genzei-ibm-v6
# [v0.2 20250907-034856] MEM=100G, Finished in 6.9 hours (24679 secs), 18656 QA pairs. samples_processed: 20668, checkpoint_counter: 1
# save_freq = None
# max_concurrency = 30

# OK for teigaku-genzei
# [v0.2 best 20250908-130424] MEM=100G, Finished in 4.5 hours (16275 secs), 18336 QA pairs. samples_processed: 20306, checkpoint_counter: 1
# save_freq = None
# max_concurrency = 40

# bkilled teigaku-genzei-ibm-v6 61.1 hours estimated
# teigaku-genzei-ibm-v6 on phi-4-test sdg_hub.core.utils.error_handling.FlowValidationError: Block 'knowledge_generation' execution failed: litellm.APIError: APIError: Hosted_vllmException - Authentication failed
# teigaku-genzei-ibm-v6 on phi-4 sdg_hub.core.utils.error_handling.FlowValidationError: Block 'verify_question' execution failed: litellm.APIError: APIError: Hosted_vllmException - Authentication failed
# save_freq = 100
# max_concurrency = 20

# [v0.2 20250923-033910] teigaku-genzei-ibm-v6 on phi-4. Finished in 23 hours (including idle time), 7 job submissions in total, 16751 QA pairs. samples_processed: 18605, checkpoint_counter: 1861
save_freq = 10
max_concurrency = 20

# bkilled teigaku-genzei-ibm-v6 didn't proceed
# teigaku-genzei-ibm-v6 on phi-4-test sdg_hub.core.utils.error_handling.FlowValidationError: Block 'knowledge_generation' execution failed: litellm.NotFoundError: NotFoundError: Hosted_vllmException - Error getting active endpoint: revision.serving.knative.dev "microsoft-phi-4-test-predictor-00074" not found
# save_freq = 100
# max_concurrency = 30

# teigaku-genzei-ibm-v6 [11:20:15] ERROR    Block 'eval_faithfulness' failed during execution: litellm.APIError: APIError: Hosted_vllmException - Authentication failed
# teigaku-genzei-ibm-v6 on phi-4-test [00:22:01] ERROR    Block 'eval_faithfulness' failed during execution: litellm.APIError: APIError: Hosted_vllmException - Authentication failed
# save_freq = 100
# max_concurrency = 40

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
if False:
    print("Dry run to test and estimate time", flush=True)

    # Test AND estimate in one call
    result = flow.dry_run(ds, sample_size=sample_size, max_concurrency=max_concurrency, enable_time_estimation=True)

    # Access dry run results
    print(f"Tested with {result['sample_size']} samples")
    print(f"Output columns: {result['final_dataset']['columns']}")

    # Time estimation is automatically displayed in a Rich table format
    # No need to access it programmatically - the table shows all estimation details

# %%
print("Now generate full data", flush=True)

# %%
# Generate data
generated_data = flow.generate(ds, checkpoint_dir=checkpoint_dir, save_freq=save_freq, max_concurrency=max_concurrency)

# %% [markdown]
# ### Converting the generated data into training format

# %%
from datasets import Dataset

def create_simple_qa_dataset(generated_data: Dataset) -> Dataset:
    seen = set()
    messages_list: list[dict] = []
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
os.makedirs(output_dir, exist_ok=True)
messages_data.to_json(f"{output_dir}/messages_data.jsonl", force_ascii=False)


