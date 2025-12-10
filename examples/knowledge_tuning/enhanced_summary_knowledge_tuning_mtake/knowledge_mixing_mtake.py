# %%
from datasets import load_dataset, Dataset, concatenate_datasets
import polars as pl
from transformers import AutoTokenizer
from tabulate import tabulate
from dotenv import load_dotenv
import os
from pathlib import Path
from knowledge_mixing_utils import (
    sample_doc_qa,
    generate_knowledge_qa_dataset,
    count_len_in_tokens,
    get_avg_summaries_per_raw_doc,
)

# %%
# Load environment variables from .env file
load_dotenv()

# Load configuration from environment variables
exp_folder = os.getenv("OUTPUT_DATA_FOLDER", "generated_output_data")
student_model = os.getenv("STUDENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
save_gpt_oss_format = os.getenv("SAVE_GPT_OSS_FORMAT", "false").lower() == "true"

# Parse cut sizes from environment variable
cut_sizes_str = os.getenv("CUT_SIZES", "10,20")
cuts = [int(x.strip()) for x in cut_sizes_str.split(",")]

# Get Q&A pairs per document
qa_per_doc = int(os.getenv("QA_PER_DOC", "3"))

# Define input and output paths relative to exp_folder
input_data_dir = os.path.join(exp_folder)
output_dir = os.path.join(exp_folder, "training_mix")

print(f"Experiment folder: {exp_folder}")
print(f"Student model: {student_model}")
print(f"GPT OSS format: {save_gpt_oss_format}")
print(f"Cut sizes: {cuts}")
print(f"Q&A pairs per document: {qa_per_doc}")
print(f"Input data directory: {input_data_dir}")
print(f"Output directory: {output_dir}")

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# %%
def load_tokenizer(student_model):
    """Initialize and return tokenizer."""
    print(f"Loading tokenizer: {student_model}")
    return AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)


def filter_gpt_oss_dataset(ds):
    """Apply GPT OSS format filtering to dataset."""
    original_size = len(ds)

    # Filter out problematic questions
    ds = ds.filter(
        lambda x: "..." not in x["question"]
        and "<question>" not in x["question"]
        and "<Insert question here>" not in x["question"]
    )

    # Clean response text
    ds = ds.map(
        lambda x: {
            "response": x["response"]
            .replace("[ANSWER]", "")
            .replace("[END]", "")
            .strip()
        }
    )

    filtered_size = len(ds)
    print(f"  Filtered {original_size - filtered_size} samples (kept {filtered_size})")
    return ds


def load_summary_dataset(summary_type):
    """Load a single summary dataset."""
    file_path = os.path.join(input_data_dir, f"{summary_type}")

    # Check if file exists
    if not Path(file_path).exists():
        print(f"⚠️  Warning: File not found: {file_path}")
        return None

    print(f"Loading {summary_type} from: {file_path}")
    ds = load_dataset("json", data_dir=file_path, split="train")
    if summary_type == "document_based_qa":
        ds = ds.rename_column("base_document", "raw_document")

    # Apply filtering if needed
    if save_gpt_oss_format:
        ds = filter_gpt_oss_dataset(ds)

    print(f"  Loaded {summary_type}: {len(ds)} samples")
    return ds.to_polars()


def load_all_summary_datasets():
    """Load all summary type datasets."""
    summary_types = [
        "extractive_summary",
        "detailed_summary",
        "key_facts_to_qa",
        "document_based_qa",
    ]

    summary_datasets = {}

    for summary_type in summary_types:
        dataset = load_summary_dataset(summary_type)
        if dataset is not None:
            summary_datasets[summary_type] = dataset

    if not summary_datasets:
        raise ValueError("No datasets were successfully loaded!")

    return summary_datasets


# Load tokenizer and datasets
try:
    tokenizer = load_tokenizer(student_model)
    summary_datasets = load_all_summary_datasets()
    # After loading each dataset

    for summary_type, dataset in summary_datasets.items():
        print(f" Columns: {list(dataset.columns)}")

        print(f" Sample record keys: {list(dataset.head(1).to_dicts()[0].keys())}")
    print(f"\n✅ Successfully loaded {len(summary_datasets)} summary datasets")
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    raise

# %%
def validate_cuts_for_datasets(summary_datasets, cuts):
    """Validate which cut sizes are feasible for each dataset."""
    feasible_cuts = set(cuts)

    print("🔍 Validating cut sizes against available data...")
    for summary_type, df in summary_datasets.items():
        if summary_type in ["key_facts_to_qa", "document_based_qa"]:
            print(f"\n📊 Skipping {summary_type}:")
            continue
        print(f"\n📊 Checking {summary_type}:")

        for cut in cuts:
            avg_summaries = get_avg_summaries_per_raw_doc(df)
            is_feasible = avg_summaries >= cut
            status = "✅ Feasible" if is_feasible else "❌ Too large"
            print(
                f"  Cut {cut}: {status} (avg summaries per raw doc: {avg_summaries:.1f})"
            )

            if not is_feasible:
                feasible_cuts.discard(cut)

    final_cuts = sorted(list(feasible_cuts))
    if len(final_cuts) < len(cuts):
        removed_cuts = set(cuts) - feasible_cuts
        print(f"\n⚠️  Removing infeasible cuts: {sorted(list(removed_cuts))}")

    print(f"\n✅ Final feasible cuts: {final_cuts}")
    return final_cuts


def process_single_summary_type(summary_type, df, cut, tokenizer, qa_per_doc):
    """Process a single summary type dataset."""
    try:
        print(f"  Processing {summary_type}...")
        if summary_type == "key_facts_to_qa":
            # Skip the sampling step for keys facts QA dataset as we discard the generated summary and only keep the qa pairs
            # Generate knowledge Q&A dataset
            generated_dataset = generate_knowledge_qa_dataset(
                df,
                keep_columns=[
                    "question",
                    "document_outline",
                    "raw_document",
                    "document",
                ],
                pre_training=True,
                keep_document_in_context=False,
            )
        else:
            if summary_type != "document_based_qa":
                # Sample documents and Q&A pairs (validation already done)
                df_cut = sample_doc_qa(df, n_docs_per_raw=cut, qa_per_doc=qa_per_doc)
            else:
                df_cut = df

            # Generate knowledge Q&A dataset
            generated_dataset = generate_knowledge_qa_dataset(
                df_cut,
                keep_columns=[
                    "question",
                    "document_outline",
                    "raw_document",
                    "document",
                ],
                pre_training=True,
                keep_document_in_context=True,
            )

        # Count tokens
        generated_dataset = count_len_in_tokens(generated_dataset, tokenizer)

        # Convert back to HuggingFace dataset
        generated_dataset = Dataset.from_polars(generated_dataset)

        # Calculate statistics
        unique_docs = len(set(generated_dataset["document"]))
        unique_raw_docs = len(set(generated_dataset["raw_document"]))
        generated_cut_size = unique_docs / unique_raw_docs if unique_raw_docs > 0 else 0

        stats = {
            "samples": len(generated_dataset),
            "unique_docs": unique_docs,
            "unique_raw_docs": unique_raw_docs,
            "avg_docs_per_raw": generated_cut_size,
            "total_tokens": sum(generated_dataset["token_length"]),
        }

        print(
            f"    ✅ Processed {len(generated_dataset)} samples ({generated_cut_size:.1f} summaries per raw doc)"
        )
        return generated_dataset, stats

    except Exception as e:
        print(f"    ❌ Error processing {summary_type}: {e}")
        return None, None


def combine_and_save_datasets(all_datasets, cut_stats, cut, output_dir):
    """Combine datasets and save to file."""
    if not all_datasets:
        print(f"  ❌ No datasets processed for cut size {cut}")
        return None

    try:
        # Combine all summary types for this cut
        combined_dataset = concatenate_datasets(all_datasets)
        total_tokens = sum(combined_dataset["token_length"])

        # Save combined dataset
        output_path = os.path.join(output_dir, f"combined_cut_{cut}x.jsonl")
        combined_dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)

        # Print results
        print(f"  💾 Saved to: {output_path}")
        print(f"  📈 Total samples: {len(combined_dataset)}")
        print(f"  🔢 Total tokens: {total_tokens:,}")

        # Print detailed statistics
        print(f"  📋 Summary statistics:")
        for summary_type, stats in cut_stats.items():
            print(
                f"    {summary_type}: {stats['samples']} samples, {stats['total_tokens']:,} tokens"
            )

        return (cut, total_tokens, len(combined_dataset))

    except Exception as e:
        print(f"  ❌ Error combining datasets for cut {cut}: {e}")
        return None


def process_single_cut(cut, summary_datasets, tokenizer, output_dir, qa_per_doc):
    """Process all summary types for a single cut size."""
    print(f"\n📊 Processing cut size: {cut}")
    all_datasets = []
    cut_stats = {}

    for summary_type, df in summary_datasets.items():
        dataset, stats = process_single_summary_type(
            summary_type, df, cut, tokenizer, qa_per_doc
        )

        if dataset is not None and stats is not None:
            all_datasets.append(dataset)
            cut_stats[summary_type] = stats

    return combine_and_save_datasets(all_datasets, cut_stats, cut, output_dir)


def process_and_mix_datasets(cuts, summary_datasets, tokenizer, output_dir, qa_per_doc):
    """Process and mix datasets with different cut sizes."""
    # First validate which cuts are feasible
    feasible_cuts = validate_cuts_for_datasets(summary_datasets, cuts)

    if not feasible_cuts:
        print("\n❌ No feasible cuts found! Check your data or reduce cut sizes.")
        return []

    token_count = []

    print(f"\nProcessing {len(feasible_cuts)} feasible cut sizes...")
    for cut in feasible_cuts:
        result = process_single_cut(
            cut, summary_datasets, tokenizer, output_dir, qa_per_doc
        )
        if result is not None:
            token_count.append(result)

    return token_count


def print_final_summary(token_count):
    """Print final summary table."""
    if token_count:
        print("\n" + "=" * 50)
        print("📊 FINAL SUMMARY")
        print("=" * 50)
        print(
            tabulate(
                token_count,
                headers=["Cut Size", "Total Tokens", "Total Samples"],
                tablefmt="github",
                numalign="right",
            )
        )
    else:
        print("\n❌ No datasets were successfully processed!")


# Process datasets
token_count = process_and_mix_datasets(
    cuts, summary_datasets, tokenizer, output_dir, qa_per_doc
)

# Print final summary
print_final_summary(token_count)

# %% [markdown]
# ### Instructlab and RAFT skills generation
# - You can follow below steps for generating [RAFT dataset](https://arxiv.org/abs/2403.10131)
# - Below cells will also go over mixing with existing skills, upsampling dataset etc. for instructlab skills phase training

# %%
dsets = []

# %%
from datasets import load_dataset
from raft_builder import RAFTConfig, build_raft_samples, build_messages

cut = cuts[-1]
fp = f"{output_dir}/combined_cut_{cut}x.jsonl"  # Replace this with path of the data generated in above cell
processed_knowledge_dataset = load_dataset("json", data_files=fp, split="train")

# Do this to remove the think field from the messages. Inline changing doesn't work.
processed_knowledge_dataset = processed_knowledge_dataset.map(
    lambda x: {
        "messages_without_think": [
            {"role": "user", "content": x["messages"][0]["content"]},
            {"role": "assistant", "content": x["messages"][1]["content"]},
        ]
    }
)
processed_knowledge_dataset = processed_knowledge_dataset.remove_columns(
    ["messages"]
).rename_column("messages_without_think", "messages")
upscale_data_size = 1

cfg = RAFTConfig(k_passages=5, max_tokens_per_chunk=400, p_include_oracle=0.9)
raft_samples = build_raft_samples(processed_knowledge_dataset, cfg)
raft_samples = raft_samples.map(build_messages).remove_columns(
    [
        "question",
        "context",
        "oracle_context",
        "cot_answer",
        "answer",
        "instruction",
        "type",
        "meta",
    ]
)
dsets += [raft_samples]

# %%
if False:
    fp = "<Instruction/Skills dataset>"  # TODO: Replace with huggingface dataset path once its uploaded
    skills = load_dataset("json", data_files=fp, split="train").remove_columns(
        ["metadata", "id"]
    )
    dsets += [skills]

    upscale_data_size = 5  # Number by which to upsample the generated data. Useful when generated data is magnitude smaller than the Skills data.

# %%
### Create InstructLab Skills dataset
skills_with_custom_knowledge = concatenate_datasets(
    dsets
    + [
        processed_knowledge_dataset.remove_columns(
            [
                "question",
                "document_outline",
                "raw_document",
                "document",
                "metadata",
                "unmask",
                "token_length",
            ]
        )
    ]
    * upscale_data_size
)

skills_with_custom_knowledge.to_json(
    f"{output_dir}/skills_with_custom_knowledge.jsonl", orient="records", lines=True, force_ascii=False
)


