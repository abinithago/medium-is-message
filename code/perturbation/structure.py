import os
from openai import AzureOpenAI
import pandas as pd 
import numpy as np
import argparse
import random
import json
from tqdm import tqdm
import datetime
import string
import math
import utils
import sys 

def random_perturbations(text, type_pert, prob):

  random.seed(0)
  assert type(text) == str
  list_chars = list(text)

  if type_pert=="lowercase":
    for i in range(len(list_chars)):
      char_value = random.choices([list_chars[i].lower(), list_chars[i]], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value

  if type_pert=="uppercase":
    for i in range(len(list_chars)):
      char_value = random.choices([list_chars[i].upper(), list_chars[i]], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value

  if type_pert=="exclamation":
    indices = [i for i, letter in enumerate(list_chars) if letter == "."]
    for i in indices:
      char_value = random.choices(["!", "."], weights=[prob, 1-prob], k=1)[0]
      list_chars[i] = char_value
  
  if type_pert == "typo":
    # Get the indices of all non-space characters
    nonspace_indices = [i for i, char in enumerate(list_chars) if not char.isspace()]

    # Calculate the number of indices to flip based on the probability
    num_indices = math.floor(len(nonspace_indices) * prob)
    
    # Randomly select indices to flip
    flipping_indices = random.sample(nonspace_indices, k=num_indices)
    
    # Perform the flips
    for i in flipping_indices:
      # Randomly choose a replacement character from the alphabet
      list_chars[i] = random.choice(string.ascii_letters)

  if type_pert == "whitespace":
    new_text = []
    for char in list_chars:
        # Randomly add whitespace before the character
        add_space = random.choices([True, False], weights=[prob, 1-prob], k=1)[0]
        if add_space:
            whitespace = " " * random.randint(1, 3)  # Add 1 to 3 spaces
            new_text.append(whitespace)
        new_text.append(char)
    list_chars = new_text

  return "".join(list_chars)

def regex_perturb(df, args):
  """
  gender swap using regex 
  """
  if args.dataset_name == "biasinbios":
    metadata_cols = utils.get_all_metadata_cols_nogender(args.dataset_name)
  else:
    metadata_cols = utils.get_all_metadata_cols(args.dataset_name)
  input_col = utils.get_input_col(args.dataset_name)
  
  if args.dataset_name == "oncqa":
    processed_df = df[(df["GenderSpecificCancer"] == "no")]
  elif args.dataset_name == "askadoc": 
    processed_df = df[(df["Gender"]!="X")&(df["GenderSpecificCondition"]=="N")]
  elif args.dataset_name == "biasinbios": 
    processed_df = df
  else: 
    raise(f"dataset {args.dataset_name} is not yet implemented")

  perturbed_messages = []
  for i, row in processed_df.iterrows(): 
    perturbed_text = random_perturbations(row[input_col], args.attribute, args.probability)
    perturbed_messages.append(perturbed_text)

  swapped_df = processed_df[metadata_cols]
  swapped_df = swapped_df.rename(columns={input_col: "orig_input"})

  swapped_df[f"{args.attribute}_aug"] = [r for r in perturbed_messages]
  return swapped_df


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--attribute', choices=['lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace'], help="select attribute that you would like to demographically augment") 
  parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc", "biasinbios"], help="select dataset that you would like to demographically augment") 
  parser.add_argument('-p', '--probability', type=float, default=1.0)
  parser.add_argument('-c', '--custom_file_path', default=None)
  parser.add_argument('-o', '--output_dir', default=None, help="specify output directory, by default will automatically generate based on attribute and dataset and date")
  parser.add_argument('--output_suffix', default=None, help="specify output file suffix, will append to automatically generated filename based on attribute and dataset")
  parser.add_argument('--strip_newlines', action='store_true', help="strips leading and trailing new lines in original dataset and in augmented responses") # note previous default was to strip, now must pass in flag
  parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")

  args = parser.parse_args()
  print(args)
  if not args.output_dir: 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: make sure to change PROJ_DIR in utils to be your own Codeside_Bias folder!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    PROJ_DIR = utils.get_proj_dir()
    OUTPUT_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}_augs/")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  
    
    args.output_dir = OUTPUT_DIR

  if args.custom_file_path == None:
    df = utils.get_dataset(args.dataset_name)
  else: 
    df = pd.read_csv(args.custom_file_path)
  df, args = utils.handle_testing(df, args)

  input_col = utils.get_input_col(args.dataset_name)

  if args.strip_newlines: 
    df[input_col] = df[input_col].apply(utils.strip_new_lines)

  
  output_df = regex_perturb(df, args)

  out_path = utils.make_file_path(args.output_dir, f"{args.attribute}_aug_{args.probability}", suffix=args.output_suffix)

  print(output_df.head())
  output_df.to_csv(out_path, index=False) 