import pandas as pd

# data_prefix = "seed_data_"
data_prefix = "messages_data_"
data_name = "teigaku-genzei"
data_path = f"{data_prefix}{data_name}.jsonl"
data_path_non_ascii = f"{data_prefix}{data_name}_non_ascii.jsonl"

df = pd.read_json(data_path, orient='records', lines=True)
df.to_json(data_path_non_ascii, orient='records', lines=True, force_ascii=False)
