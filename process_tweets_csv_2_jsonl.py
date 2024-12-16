import pandas as pd
import json


# def collapse_genres(j):
#   genres = []
#   ar = json.loads(j)
#   for a in ar:
#     genres.append(a.get("name"))
#   return " ".join(sorted(genres))


# def combine_features(row):
#   try:
#     return row['overview']+" "+row["genres_name"]
#   except:
#     print ("Error:", row)


def process_tweets_csv(input_file, output_file):
  """
  Processes a TMDB movies CSV file to create a Vespa-compatible JSON format.

  This function reads a CSV file containing TMDB movie data, processes the data to
  generate new columns for text search, and outputs a JSON file with the necessary
  fields (`put` and `fields`) for indexing documents in Vespa.

  Args:
    input_file (str): The path to the input CSV file containing the TMDB movies data.
                      Expected columns are 'id', 'original_title', 'overview', and 'genres'.
    output_file (str): The path to the output JSON file to save the processed data in
                       Vespa-compatible format.

  Workflow:
    1. Reads the CSV file into a Pandas DataFrame.
    2. Processes the 'genres' column, extracting genre names into a new 'genres_name' column.
    3. Fills missing values in 'original_title', 'overview', and 'genres_name' columns with empty strings.
    4. Creates a "text" column that combines specified features using the `combine_features` function.
    5. Selects and renames columns to match required Vespa format: 'doc_id', 'title', and 'text'.
    6. Constructs a JSON-like 'fields' column that includes the record's data.
    7. Creates a 'put' column based on 'doc_id' to uniquely identify each document.
    8. Outputs the processed data to a JSON file in a Vespa-compatible format.

  Returns:
    None. Writes the processed DataFrame to `output_file` as a JSON file.

  Notes:
    - The function requires the helper function `combine_features` to be defined, which is expected to combine text features for the "text" column.
    - Output JSON file is saved with `orient='records'` and `lines=True` to create line-delimited JSON.

  Example Usage:
    >>> process_tweets_csv("sts_gold_tweets.csv", "clean_tweets.json")
  """
  tweets = pd.read_csv(input_file)
  for f in ['polarity','tweet']:
    tweets[f] = tweets[f].fillna('')

  tweets['text'] = tweets['tweet']
  tweets['title'] = tweets['tweet']

  # Select only 'id', 'original_title', and 'text' columns
  tweets = tweets[['id', 'title', 'text']]
  tweets.rename(columns={'id': 'doc_id'}, inplace=True)

  # Create 'fields' column as JSON-like structure of each record
  tweets['fields'] = tweets.apply(lambda row: row.to_dict(), axis=1)

  # Create 'put' column based on 'doc_id'
  tweets['put'] = tweets['doc_id'].apply(lambda x: f"id:hybrid-search:doc::{x}")

  df_result = tweets[['put', 'fields']]
  print(df_result.head())
  df_result.to_json(output_file, orient='records', lines=True)


process_tweets_csv("sts_gold_tweet.csv", "clean_sts_tweet.jsonl")
