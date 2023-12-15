import pandas as pd
import json

# List of JSON file paths
json_files = [
    './112-1_ADL_Final/poem_dataset/poem_0_2.json',
    './112-1_ADL_Final/poem_dataset/poem_2_3.json',
    './112-1_ADL_Final/poem_dataset/poem_4_6.json',
    './112-1_ADL_Final/poem_dataset/poem_6_7.json',
    './112-1_ADL_Final/poem_dataset/poem_7_8.json',
    './112-1_ADL_Final/poem_dataset/poem_8_9.json',
    './112-1_ADL_Final/poem_dataset/poem_9_10.json',
    './112-1_ADL_Final/poem_dataset/poem_10_11.json',
    './112-1_ADL_Final/poem_dataset/poem_12_13.json',
    './112-1_ADL_Final/poem_dataset/poem_13_14.json',
]

# Create an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Iterate over the JSON files
for json_file_path in json_files:
    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Create a DataFrame from the JSON data
    json_df = pd.DataFrame(json_data)

    # Add a new column "output" to the DataFrame with the paragraph from the JSON data
    json_df['output'] = json_df['poem']
    # Add a new column "instruction"
    json_df['instruction'] = json_df.apply(lambda row: f"Generate a poem for the following scenario:\n {row['translate']}\nOutput:", axis=1)

    # Concatenate the DataFrame to the merged DataFrame
    merged_df = pd.concat([merged_df, json_df], ignore_index=True)


# Convert the merged DataFrame to a dictionary
result_dict = merged_df.to_dict(orient='records')

# Write the result to a new JSON file
with open('./data/mergedPoemDataset.json', 'w', encoding='utf-8') as output_json_file:
    json.dump(result_dict, output_json_file, indent=4, ensure_ascii=False)


# # Read the first JSON file
# with open('./test.json', 'r') as json_file:
#     json_data_1 = json.load(json_file)

# # Create a DataFrame from the first JSON data
# json_df_1 = pd.DataFrame(json_data_1)


# # Add a new column "output" to the DataFrame with the paragraph from the JSON data
# # Add a new column "instruction"
# #json_df_1['instruction'] = json_df_1.apply(lambda row: f"Generate a poem for the following scenario:\n {row['output']}\nOutput:", axis=1)
# json_df_1['instruction'] = json_df_1.apply(lambda row: f"生成以下情境的詩:\n {row['output']}\nOutput:", axis=1)
# result_dict = json_df_1.to_dict(orient='records')

# with open('./data/test.json', 'w', encoding='utf-8') as output_json_file:
#     json.dump(result_dict, output_json_file, indent=4, ensure_ascii=False)

