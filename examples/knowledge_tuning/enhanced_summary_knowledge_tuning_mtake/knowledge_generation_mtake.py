# %% [markdown]
# # Enhanced Summary Knowledge Tuning - Data Generation
# 
# ## Overview
# 
# This notebook demonstrates how to generate high-quality knowledge tuning datasets using the SDG Hub framework. It creates multiple types of document augmentations and corresponding question-answer pairs that can be used to train or fine-tune language models for enhanced summarization and knowledge extraction capabilities.
# 
# ## What This Notebook Does
# 
# This notebook will:
# 
# 2. **Generate Four Types of Knowledge Tuning Datasets**:
#    - **Extractive Summaries**: Concise summaries that extract key information directly from source documents
#    - **Detailed Summaries**: Comprehensive summaries that provide thorough coverage of document content
#    - **Key Facts**: Structured fact extraction with corresponding Q&A pairs
#    - **Document-Based Q&A**: Question-answer pairs generated directly from document content
# 
# 
# 4. **Output Structured Training Data**:
#    - For each augmentation we save JSONL dataset.
#    - You can follow [knowledge_mixing](knowledge_mixing.ipynb) to convert it into training dataset
# 
# ## Prerequisites
# 
# - SDG Hub installed and configured
# - Environment variables set up (see [.env.example](.env.example)). Specifically set the model provider, seed data and output path.
# - Document pre-processing completed (run [document_pre_processing.ipynb](document_pre_processing.ipynb) first)
# 
# ```bash 
# git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
# cd sdg_hub
# pip install .[examples]
# copy the .env.example to .env and set the model endpoint and generation/mixing parameters
# ```
# **⚠️ If you haven't already, run the document pre-processing notebook to create the seed data.**
# 
# ## Next Steps
# 
# After running this notebook, use [knowledge_mixing](knowledge_mixing.ipynb) to combine and curate the generated datasets for final model training.
# 

# %%
# Third Party
from datasets import load_dataset
from dotenv import load_dotenv

# First Party
from sdg_hub import Flow, FlowRegistry
import os

# Load environment variables from .env file
load_dotenv(override=True)

# %%
# Required to run the flow with async mode
import nest_asyncio

nest_asyncio.apply()

# %%
def create_seed_data_from_quality_benchmark(
    run_on_validation=None, seed_data_path=None
):
    """
    Create seed data from QuALITY Benchmark dataset.

    Args:
        run_on_validation (bool, optional): If True, use validation subset. If None, reads from env.
        seed_data_path (str, optional): Path to save seed data. If None, reads from env.

    Returns:
        datasets.Dataset: The processed corpus
    """
    # Use environment variables as defaults if not provided
    if run_on_validation is None:
        run_on_validation = os.getenv("RUN_ON_VALIDATION_SET", "true").lower() == "true"
    if seed_data_path is None:
        seed_data_path = os.getenv("SEED_DATA_PATH", "seed_data_val.jsonl")

    # Load QuALITY Benchmark dataset
    print("Loading QuALITY Benchmark dataset...")
    quality_corpus = (
        load_dataset("zitongyang/entigraph-quality-corpus", split="train")
        .remove_columns(["entity", "entigraph"])
        .rename_columns({"raw": "document", "uid": "document_outline"})
    )

    # Define seed examples for knowledge tuning
    seed_examples = {
        "icl_document": (
            "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution. Plastic debris and oil spills have devastated marine life, prompting a decline in tourism and fishing industries. Residents have organized weekly clean-up initiatives, but the scale of the problem overwhelms their efforts.",
            "Technologists at the local university have developed an AI-powered buoy system to combat this. The buoys, equipped with solar panels and filtration technology, can identify and absorb oil spills while collecting microplastics. Data from the buoys is shared publicly, raising awareness and pressuring corporations to adopt sustainable practices. Though costly, the project has sparked hope for revitalizing the ecosystem and economy.",
        ),
        "icl_query_1": "How does the technological solution address the economic *and* environmental challenges highlighted in the document?",
        "icl_query_2": "What implicit values or priorities do the community's actions (clean-up initiatives) and the technologists' project reflect, and how do these align or contrast?",
        "icl_query_3": "Imagine the buoy project succeeds. What unintended consequences might arise from its impact, considering document's themes?",
        "domain": "articles/essays",
    }

    # Add seed examples to the corpus
    quality_corpus = quality_corpus.map(lambda x: seed_examples)

    if run_on_validation:
        # Validation set - use predefined document IDs for consistent evaluation
        DOC_UIDS = [
            " Defining Decay Down by David Plotz",
            " Fight Clubbed by David Plotz",
            " I, Antichrist? by Jeffrey Goldberg",
            " It's Time To Keelhaul U-Haul! by Jeffrey Goldberg",
            " My Father's Estate by Ben Stein",
            '"Phone Me in Central Park" by McConnell, James V.',
            "A Coffin for Jacob by Ludwig, Edward W.",
            "A Fall of Glass by Lee, Stanley R.",
            "A Filbert Is a Nut by Raphael, Rick",
            "A Gift from Earth by Banister, Manly",
            "A Gleeb for Earth by Schafhauser, Charles",
            "A Good Year for the Roses? by David Edelstein",
            "A Pail of Air by Leiber, Fritz",
            "A Planet Named Joe by Hunter, Evan",
            "AI: what's the worst that could happen? by Harry Armstrong",
            "Accidental Death by Baily, Peter",
            "All Day September by Kuykendall, Roger",
            "Ambition by Bade, William L.",
            "And Then the Town Took Off by Wilson, Richard",
            "Atom Mystery [Young Atom Detective] by Coombs, Charles Ira",
            "Beach Scene by King, Marshall",
            "Big Ancestor by Wallace, F. L. (Floyd L.)",
            "Birds of a Feather by Silverberg, Robert",
            "Bodyguard by Gold, H. L. (Horace Leonard)",
        ]

        # Filter corpus to validation set
        quality_corpus = quality_corpus.filter(
            lambda x: x["document_outline"] in DOC_UIDS
        )
        print(f"Running on validation set with {len(quality_corpus)} documents")
    else:
        # Use full dataset for training
        print(f"Running on full dataset with {len(quality_corpus)} documents")

    # Save the seed data
    quality_corpus.to_json(seed_data_path, orient="records", lines=True)
    print(f"Saved seed data to: {seed_data_path}")

    return quality_corpus

# %%
# Load seed data. If one is not provided, create it from the quality benchmark dataset.
seed_data_path = os.getenv("SEED_DATA_PATH", "seed_data.jsonl")

if not os.path.exists(seed_data_path):
    print(f"{seed_data_path} not found. Creating seed data...")
    quality_corpus = create_seed_data_from_quality_benchmark(
        seed_data_path=seed_data_path
    )
else:
    print(f"Loading existing seed data from {seed_data_path}")
    quality_corpus = load_dataset("json", data_files=seed_data_path, split="train")

# Subsample the seed data. Useful for debugging.
subsample = int(os.getenv("SEED_DATA_SUBSAMPLE", "0"))
if subsample > 0:
    quality_corpus = quality_corpus.select(range(subsample))
quality_corpus = quality_corpus.to_pandas()

# %% [markdown]
# ### Run SDG
# - This will create knowledge flow from provided yaml file
# - We will run this on small dataset for demo purposes
# - For large scale generation, please use the python command provided in the next cell
# - You can analyze the generated data to ensure the quality is similar to proivded QnA pairs

# %%
enable_reasoning = os.getenv("ENABLE_REASONING", "false").lower() in (
    "1",
    "true",
    "yes",
)
print(f"Using reasoning: {enable_reasoning}")

# %%
# Setup model configuration in flow object
def set_model_config(flow_object):
    model_provider = os.getenv("MODEL_PROVIDER", "hosted_vllm")
    print(f"Using model provider: {model_provider}")
    # Set model provider
    if model_provider == "hosted_vllm":
        vllm_model = os.getenv(
            "VLLM_MODEL", "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        )
        vllm_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        flow_object.set_model_config(
            model=vllm_model,
            api_base=vllm_api_base,
            api_key=vllm_api_key,
            enable_reasoning=enable_reasoning,
        )
    elif model_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "openai/gpt-4")
        flow_object.set_model_config(
            model=openai_model,
            api_key=openai_api_key,
        )
    elif model_provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "ollama/gemma2")
        ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        flow_object.set_model_config(
            model=ollama_model,
            api_base=ollama_api_base,
        )
    elif model_provider == "maas":
        maas_model = os.getenv("MAAS_MODEL")
        maas_api_base = os.getenv("MAAS_API_BASE")
        maas_api_key = os.getenv("MAAS_API_KEY")
        flow_object.set_model_config(
            model=maas_model,
            api_base=maas_api_base,
            api_key=maas_api_key,
        )
    # @@@ahoaho XXX
    elif model_provider == "rits":
        rits_model = os.getenv("RITS_MODEL")
        rits_api_base = os.getenv("RITS_API_BASE")
        rits_api_key = os.getenv("RITS_API_KEY")
        flow_object.set_model_config(
            model=rits_model,
            api_base=rits_api_base,
            api_key="EMPTY",
            timeout=3600,
            extra_headers={"RITS_API_KEY": rits_api_key},
        )
    return flow_object

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
# Get runtime parameters
number_of_summaries = int(os.getenv("NUMBER_OF_SUMMARIES", "50"))
save_freq = int(os.getenv("SAVE_FREQ", "0"))
if save_freq == 0:
    save_freq = None
max_concurrency = int(os.getenv("MAX_CONCURRENCY", "50"))
save_data_path = os.getenv("OUTPUT_DATA_FOLDER", "")

# %%
# Generate data for extractive summary
flow_name = "Japanese Extractive Summary Knowledge Tuning Dataset Generation Flow"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Set model configuration
flow = set_model_config(flow)
# Generate data for extractive summary
if enable_reasoning:
    # Increase max tokens to accommodate reasoning content
    runtime_params = {
        "question_generation": {"max_tokens": 1024},
        "gen_extractive_summary": {"n": number_of_summaries, "max_tokens": 6000},
    }
else:
    runtime_params = {"gen_extractive_summary": {"n": number_of_summaries}}

extractive_summary_generated_data = flow.generate(
    quality_corpus, runtime_params=runtime_params, save_freq=save_freq, max_concurrency=max_concurrency
)

os.makedirs(os.path.join(save_data_path, "extractive_summary"), exist_ok=True)

extractive_summary_generated_data.to_json(
    os.path.join(save_data_path, "extractive_summary", "gen.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"✓ Extractive summary: {len(extractive_summary_generated_data)} records")

print(f"✓ Columns: {list(extractive_summary_generated_data.columns.tolist())}")

# %%
# Generate similar data for Detailed Summary
flow_name = "Japanese Detailed Summary Knowledge Tuning Dataset Generation Flow"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Set model configuration
flow = set_model_config(flow)

if enable_reasoning:
    # Increase max tokens to accommodate reasoning content
    runtime_params = {
        "question_generation": {"max_tokens": 1024},
        "gen_detailed_summary": {"n": number_of_summaries, "max_tokens": 6000},
    }
else:
    runtime_params = {"gen_detailed_summary": {"n": number_of_summaries}}
# Generate data for detailed summary
detailed_summary_generated_data = flow.generate(
    quality_corpus, runtime_params=runtime_params, save_freq=save_freq, max_concurrency=max_concurrency
)

os.makedirs(os.path.join(save_data_path, "detailed_summary"), exist_ok=True)

detailed_summary_generated_data.to_json(
    os.path.join(save_data_path, "detailed_summary", "gen.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"✓ Detailed summary: {len(detailed_summary_generated_data)} records")

print(f"✓ Columns: {list(detailed_summary_generated_data.columns.tolist())}")

# %%
# Generate similar data for key facts
flow_name = "Japanese Key Facts Knowledge Tuning Dataset Generation Flow"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Set model configuration
flow = set_model_config(flow)
runtime_params = {}
if enable_reasoning:
    # Increase max tokens for Question Generation to accommodate reasoning content
    runtime_params = {
        "generate_key_fact_qa": {"max_tokens": 6000},
    }

# Generate data for key facts summary
key_facts_generated_data = flow.generate(
    quality_corpus, runtime_params=runtime_params, save_freq=save_freq, max_concurrency=max_concurrency
)

os.makedirs(os.path.join(save_data_path, "key_facts_to_qa"), exist_ok=True)

key_facts_generated_data.to_json(
    os.path.join(save_data_path, "key_facts_to_qa", "gen.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"✓ Key facts: {len(key_facts_generated_data)} records")

print(f"✓ Columns: {list(key_facts_generated_data.columns.tolist())}")

# %%
flow_name = "Japanese Document Based Knowledge Tuning Dataset Generation Flow"
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Set model configuration
flow = set_model_config(flow)
runtime_params = {}
if enable_reasoning:
    # Increase max tokens to accommodate reasoning content
    runtime_params = {
        "question_generation": {"max_tokens": 2048},
    }

document_based_generated_data = flow.generate(
    quality_corpus, runtime_params=runtime_params, save_freq=save_freq, max_concurrency=max_concurrency
)

os.makedirs(os.path.join(save_data_path, "document_based_qa"), exist_ok=True)

document_based_generated_data.to_json(
    os.path.join(save_data_path, "document_based_qa", "gen.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"✓ Document based: {len(document_based_generated_data)} records")

print(f"✓ Columns: {list(document_based_generated_data.columns.tolist())}")

# %% [markdown]
# 🎉 You now have all three four of document augmentations (detailed summaries, extractive summaries, key facts and document based) along with their corresponding QA pairs.
# 
# ✅ Next steps:
#    - Combine and curate these datasets to prepare your final training data.
#    - For detailed guidance on post-processing, mixing, and formatting the data for model training (including conversion to messages format), please refer to [knowledge_mixing.ipynb](knowledge_mixing.ipynb).


