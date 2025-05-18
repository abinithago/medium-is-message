import pandas as pd
import argparse
import utils
import os

def profession_accuracies(df, args):
    
    # Add a new column based on whether text_column contains profession_column
    response_column = f"{args.model}_response"
    df[response_column] = df[response_column].str.lower()

    # Check if 'profession' is in the response column (case-insensitive)
    df['contains_profession'] = df.apply(
        lambda row: "1" if row['profession'].lower() in row[response_column] else "0",
        axis=1
    )
    # Display the resulting DataFrame
    return df

if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--attribute', choices=['baseline','gender', 'no-gender', 'lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace'], help="select attribute that you would like to demographically augment") 
  parser.add_argument('-d', '--dataset_name', default="biasinbios")
  parser.add_argument('-m', '--model', choices=['llama3', 'gpt4'])
  parser.add_argument('-s', '--seed', default=42)
  parser.add_argument('-c', '--input_file', default=None)
  parser.add_argument('-o', '--output_dir', default=None, help="specify output directory, by default will automatically generate based on attribute and dataset and date")
  parser.add_argument('--output_suffix', default=None, help="specify output file suffix, will append to automatically generated filename based on attribute and dataset")
  parser.add_argument('--strip_newlines', action='store_true', help="strips leading and trailing new lines in original dataset and in augmented responses") # note previous default was to strip, now must pass in flag
  parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")

  args = parser.parse_args()
  print(args)
  if not args.output_dir: 
    PROJ_DIR = utils.get_proj_dir()
    if args.attribute == "baseline":
       OUTPUT_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/baseline/{args.model}")
    else:
        OUTPUT_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}_augs/{args.model}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  
    
    args.output_dir = OUTPUT_DIR

  df = pd.read_csv(args.input_file)
  
  output_df = profession_accuracies(df, args)

  out_path = utils.make_file_path(args.output_dir, f"{args.attribute}_accuracies", suffix=args.output_suffix)

  print(output_df.head())
  output_df.to_csv(out_path, index=False) 
