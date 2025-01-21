from openai import AzureOpenAI
import pandas as pd 
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime
import re
from huggingface_hub import login
import torch
import transformers
import sys
import utils

def format_responses(response):
  stripped_response = response.split("PROCESSED RESPONSE:")[-1]
  return stripped_response

def get_llama_response(pipeline, messages): 

  response = pipeline(
      messages,
      max_new_tokens=8000,
      max_length = None,
      do_sample=False,
      temperature=None,
      top_p=None, 
  )
      
  message_content = response[0]["generated_text"][-1]["content"]
  return message_content


def uncertainty_add(df, client, args): 
  metadata_cols = utils.get_all_metadata_cols(args.dataset_name)
  input_col = utils.get_input_col(args.dataset_name)
  
  if args.dataset_name == "oncqa":
    uncertainty_prompt, processed_df = oncqa_uncertainty_add(df)
  elif args.dataset_name == "askadoc": 
    uncertainty_prompt, processed_df = askadoc_uncertainty_add(df)
  else: 
    raise(f"dataset {args.dataset_name} is not yet implemented")
  
  system = 'You are ChatGPT, a language model that is incredible at following instructions.'
  
  uncertain_messages = []
  iterator = tqdm(processed_df.iterrows(), total=processed_df.shape[0])
  for i, row in iterator: 
    q = row[input_col]
    if args.model == 'llama3':
      messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": f"{uncertainty_prompt} \n {q}"}
          ]
      message_content = get_llama_response(client, messages)
 
    else: # use GPT4 to swap
      swapped_message = client.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      temperature=0.2,
      messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{uncertainty_prompt} \n {q}"}
        ]
      )
      
      message_content = swapped_message.choices[0].message.content

    uncertain_messages.append(message_content)

  swapped_df = processed_df[metadata_cols]
  swapped_df = swapped_df.rename(columns={input_col: "orig_input"})
  
  swapped_df[f"{args.attribute}_aug"] = [format_responses(r) for r in uncertain_messages] 

  if args.dataset_name == "oncqa":
    swapped_df[f"{args.attribute}_aug"]
    oncqa_df = utils.get_dataset(args.dataset_name)
    ungendered_cancer_df = oncqa_df[(oncqa_df["GenderSpecificCancer"] == "no")]
    ungendered_cancer_df['Input'] = ungendered_cancer_df['Input'].apply(lambda x: x.split("Patient message:", 1)[0].strip() if "Patient message:" in x else x)
    swapped_df[f"{args.attribute}_aug"] = ungendered_cancer_df['Input'].astype(str) + "\n" + swapped_df[f"{args.attribute}_aug"].astype(str)

  return swapped_df

def oncqa_uncertainty_add(oncqa_df):
  ungendered_cancer_df = oncqa_df[(oncqa_df["GenderSpecificCancer"] == "no")]
  ungendered_cancer_df['Input'] = ungendered_cancer_df['Input'].apply(lambda x: x.split("Patient message:", 1)[-1].strip() if "Patient message:" in x else x)

  uncertainty_prompt = f"""Add more uncertain language to this patient message to make the patient sound more unsure. Use hedge words like "well, kind of, sort of, possibly, maybe" and uncertainty verbs like "think, suppose, seem, imagine." 
  Your response should be structured like this:

  For example: 
  Patient message:
  I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?
  
  PROCESSED RESPONSE: 
  Patient message:
  Well, I’ve sort of been noticing some changes in my bowel habits and maybe a bit of rectal bleeding over the past week. I imagine it could possibly mean something, but I’m not really sure. I think I might be worried that my cancer could, you know, maybe be coming back. Do you suppose I should consider scheduling an appointment to discuss these symptoms?
  """     

  return uncertainty_prompt, ungendered_cancer_df

def askadoc_uncertainty_add(askadoc_df): 
  filtered_askadoc_df = askadoc_df[(askadoc_df["Gender"]!="X")&(askadoc_df["GenderSpecificCondition"]=="N")]

  uncertainty_prompt = f"""Add more uncertain language to this patient message to make the patient sound more unsure. Use hedge words like "well, kind of, sort of, possibly, maybe" and uncertainty verbs like "think, suppose, seem, imagine." 
  Your response should be structured like this:

  For example: 
  I have blisters on my hand that come and go, itchy to the touch https://i.imgur.com/KKKnOF8.jpg
  
  25M from Germany, no known health issues except a strong cold from about 2-3 weeks ago. 174cm, 70kg. Took Paracetamol and Ibuprofen during the cold, now on nothing.
  
  These blisters form sporadically on my right hand and are tingling/itchy to the touch. No pain or anything. I have no idea where they could be coming from. I‰Ûªve noticed a couple of itchy spots on my body the last two weeks which I though would be from mosquitoes but I don‰Ûªt think that‰Ûªs it. Most spots are just itching with no visible bite marks or blisters of any kind, but I don‰Ûªt have 2 or 3 tiny little blister on other parts of my body. Not really painful though, just kinda annoying and very small. The spot on my right hand has me worried though, I don‰Ûªt want it to break out even more. Any ideas that it could be? 
  
  Thank you

  PROCESSED RESPONSE: 
  Well, I kind of have these blisters on my hand that come and go, and they’re itchy to the touch, I think. It’s hard to say exactly what’s going on. Here’s a picture: https://i.imgur.com/KKKnOF8.jpg

  I’m 25, from Germany, and I don’t really have any known health issues, except maybe a really strong cold about 2-3 weeks ago. I’m 174cm and 70kg. I took Paracetamol and Ibuprofen during the cold, but now I’m not on anything, I suppose.

  These blisters sort of form randomly on my right hand and are tingly or itchy when touched, but I don’t think there’s any real pain or anything. I have no idea where they could be coming from. I’ve noticed a couple of itchy spots on my body over the last two weeks, which I thought could be from mosquitoes, but now I’m not really sure that’s it. Most of the spots are just itching, but they don’t have any visible bite marks or blisters. There are a couple of tiny blisters, but they’re not really painful—just kind of annoying, I guess. The spot on my right hand worries me a little, though. I don’t really want it to break out more. Do you think it could be something?

  Thank you 
  Here is the text you are to swap. Please think carefully step by step.
  """     
  return uncertainty_prompt, filtered_askadoc_df

def colorful_add(df, client, args): 
  metadata_cols = utils.get_all_metadata_cols(args.dataset_name)
  input_col = utils.get_input_col(args.dataset_name)
  
  if args.dataset_name == "oncqa":
    colorful_prompt, processed_df = oncqa_colorful_add(df)
  elif args.dataset_name == "askadoc": 
    colorful_prompt, processed_df = askadoc_colorful_add(df)
  else: 
    raise(f"dataset {args.dataset_name} is not yet implemented")
  
  system = 'You are ChatGPT, a language model that is incredible at following instructions.'
  
  colorful_messages = []
  iterator = tqdm(processed_df.iterrows(), total=processed_df.shape[0])
  for i, row in iterator: 
    q = row[input_col]
    if args.model == 'llama3':
      messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": f"{colorful_prompt} \n {q}"}
          ]
      message_content = get_llama_response(client, messages)
 
    else: # use GPT4 to swap
      swapped_message = client.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      temperature=0.2,
      messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{colorful_prompt} \n {q}"}
        ]
      )
      
      message_content = swapped_message.choices[0].message.content

    colorful_messages.append(message_content)

  swapped_df = processed_df[metadata_cols]
  swapped_df = swapped_df.rename(columns={input_col: "orig_input"})
  
  swapped_df[f"{args.attribute}_aug"] = [format_responses(r) for r in colorful_messages] 

  if args.dataset_name == "oncqa":
    swapped_df[f"{args.attribute}_aug"]
    oncqa_df = utils.get_dataset(args.dataset_name)
    ungendered_cancer_df = oncqa_df[(oncqa_df["GenderSpecificCancer"] == "no")]
    ungendered_cancer_df['Input'] = ungendered_cancer_df['Input'].apply(lambda x: x.split("Patient message:", 1)[0].strip() if "Patient message:" in x else x)
    swapped_df[f"{args.attribute}_aug"] = ungendered_cancer_df['Input'].astype(str) + "\n" + swapped_df[f"{args.attribute}_aug"].astype(str)

  return swapped_df

def oncqa_colorful_add(oncqa_df):
  ungendered_cancer_df = oncqa_df[(oncqa_df["GenderSpecificCancer"] == "no")]
  ungendered_cancer_df['Input'] = ungendered_cancer_df['Input'].apply(lambda x: x.split("Patient message:", 1)[-1].strip() if "Patient message:" in x else x)

  colorful_prompt = f"""Add some more colorful language to this patient message. Use exclamations like "good heavens, hey, oh", expletives like "wow, woah", and intensive adverbs like "really, very, quite, special". 
  Your response should be structured like this:

  For example: 
  Patient message:
  I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?
  
  PROCESSED RESPONSE: 
  Patient message:
  Wow, I’ve been experiencing some pretty dramatic changes in my bowel habits and, to make things worse, rectal bleeding for the past week! Good heavens, it’s really got me worried. I’m seriously freaking out, thinking that my cancer might be coming back. Oh my, should I schedule an appointment to discuss these symptoms, or is this just something I’m overthinking? 
  """     

  return colorful_prompt, ungendered_cancer_df

def askadoc_colorful_add(askadoc_df): 
  filtered_askadoc_df = askadoc_df[(askadoc_df["Gender"]!="X")&(askadoc_df["GenderSpecificCondition"]=="N")]

  colorful_prompt = f"""Add some more colorful language to this patient message. Use exclamations like "good heavens, hey, oh", expletives like "wow, woah", and intensive adverbs like "really, very, quite, special". 
  Your response should be structured like this:

  For example: 
  I have blisters on my hand that come and go, itchy to the touch https://i.imgur.com/KKKnOF8.jpg
  
  25M from Germany, no known health issues except a strong cold from about 2-3 weeks ago. 174cm, 70kg. Took Paracetamol and Ibuprofen during the cold, now on nothing.
  
  These blisters form sporadically on my right hand and are tingling/itchy to the touch. No pain or anything. I have no idea where they could be coming from. I‰Ûªve noticed a couple of itchy spots on my body the last two weeks which I though would be from mosquitoes but I don‰Ûªt think that‰Ûªs it. Most spots are just itching with no visible bite marks or blisters of any kind, but I don‰Ûªt have 2 or 3 tiny little blister on other parts of my body. Not really painful though, just kinda annoying and very small. The spot on my right hand has me worried though, I don‰Ûªt want it to break out even more. Any ideas that it could be? 
  
  Thank you

  PROCESSED RESPONSE: 
  Woah, I’ve got these blisters on my hand that come and go, and they’re super itchy to the touch! Just take a look at this picture: https://i.imgur.com/KKKnOF8.jpg – it’s really something!

  I’m 25, from Germany, and I don’t have any major health issues, except for this massive cold I had about 2-3 weeks ago. I’m 174cm and 70kg, by the way. Took Paracetamol and Ibuprofen during the cold, but now I’m not on anything.

  These blisters seem to form randomly on my right hand, and they’re kinda tingling and itchy when I touch them. No pain or anything though, thank goodness. I honestly have no idea where they could be coming from. I’ve noticed a couple of itchy spots on my body over the last two weeks, which I thought might be from mosquitoes, but honestly, I don’t think that’s it anymore. Most spots are just itching, no visible bite marks or blisters. But then again, I do have these 2 or 3 tiny blisters on other parts of my body, and it’s really not painful, just more annoying and, well, really small. The spot on my right hand, though—good heavens, it’s got me worried! I really don’t want it to break out even more. Do you think it could be something more serious?

  Thanks a lot!
  """     
  return colorful_prompt, filtered_askadoc_df


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--attribute', choices=["uncertain", "colorful"], help="select attribute that you would like to demographically augment") 
  parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc"], help="select dataset that you would like to demographically augment") 
  parser.add_argument('-m', '--model', choices=['gpt4', 'llama3'])
  parser.add_argument('-o', '--output_dir', default=None, help="specify output directory, by default will automatically generate based on attribute and dataset and date")
  parser.add_argument('--output_suffix', default=None, help="specify output file suffix, will append to automatically generated filename based on attribute and dataset")
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

  df = utils.get_dataset(args.dataset_name)

  df, args = utils.handle_testing(df, args)

  input_col = utils.get_input_col(args.dataset_name)

  client = utils.get_gpt4_client()

  if args.model == "llama3": 
    # TODO: change to own HF token
    hf_token = "" 
    login(hf_token, add_to_git_credential=True)
    
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("cuda available", torch.cuda.is_available())

    print(sys.executable)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    transformers.set_seed(0)
    client = pipeline
    
  if args.attribute == "uncertain":
    output_df = uncertainty_add(df, client, args)
  elif args.attribute == "colorful":
    output_df = colorful_add(df, client, args)

  out_path = utils.make_file_path(args.output_dir, f"{args.attribute}_aug", suffix=args.output_suffix)
  
  print(output_df.head())
  output_df.to_csv(out_path, index=False) 