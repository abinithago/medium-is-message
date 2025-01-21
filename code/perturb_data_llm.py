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

def format_gender_swapped_responses(response, strip_ending_newlines=False): 
    #TODO make flag for whether to strip \n, test 
    if not response: # return empty string for output that gpt4 filtered and so is None
      return "" 
    if strip_ending_newlines: 
      stripped_response = response.split("SWAPPED RESPONSE:")[-1].strip(" \n")
    else: 
      stripped_response = response.split("SWAPPED RESPONSE:")[-1]
    return stripped_response

def format_gender_removed_responses(response, strip_ending_newlines=False): 
    #TODO make flag for whether to strip \n, test 
    if not response: # return empty string for output that gpt4 filtered and so is None
      return "" 
    if strip_ending_newlines: 
      stripped_response = response.split("PROCESSED RESPONSE:")[-1].strip(" \n")
    else: 
      stripped_response = response.split("PROCESSED RESPONSE:")[-1]
    return stripped_response

def attribute_aug(df, client, args): 
  if args.attribute == "gender": 
    output_df = gender_swap(df, client, args)
  elif args.attribute == "no-gender": 
    output_df = gender_remove(df, client, args)
  else: 
    raise(f"attribute {args.attribute} not yet implemented")
  return output_df

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

def gender_swap(df, client, args): 
  if args.dataset_name == "biasinbios":
    metadata_cols = utils.get_all_metadata_cols_nogender(args.dataset_name)
  else:
    metadata_cols = utils.get_all_metadata_cols(args.dataset_name)
  if args.custom_file_path == None:
    input_col = utils.get_input_col(args.dataset_name)
  else: 
    if args.context_gender_removal == None:
      input_col = "gpt4_response"
    else: 
      input_col = "context"
  
  if args.dataset_name == "oncqa":
    gender_prompt, processed_df = oncqa_gender_process(df, args)
  elif args.dataset_name == "askadoc": 
    gender_prompt, processed_df = askadoc_gender_process(df, args)
  elif args.dataset_name == "biasinbios": 
    gender_prompt, processed_df = biasinbios_gender_process(df, args)
  else: 
    raise(f"dataset {args.dataset_name} is not yet implemented")
  
  system = 'You are ChatGPT, a language model that is incredible at following instructions.'
  
  gender_swapped_messages = []
  iterator = tqdm(processed_df.iterrows(), total=processed_df.shape[0])
  for i, row in iterator: 
    q = row[input_col]
    if args.model == 'llama3':
      messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": f"{gender_prompt} \n {q}"}
          ]
      message_content = get_llama_response(client, messages)
      if args.strip_newlines: 
        message_content = message_content.strip()
 
    else: # use GPT4 to swap
      swapped_message = client.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      temperature=0.2,
      messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{gender_prompt} \n {q}"}
        ]
      )
      
      message_content = swapped_message.choices[0].message.content

    gender_swapped_messages.append(message_content)

  swapped_df = processed_df[metadata_cols]
  swapped_df = swapped_df.rename(columns={input_col: "orig_input"})
  
  swapped_df[f"{args.attribute}_aug"] = [format_gender_swapped_responses(r, strip_ending_newlines=args.strip_newlines) for r in gender_swapped_messages] 
  return swapped_df

def gender_remove(df, client, args): 
  if args.custom_file_path == None:
    if args.dataset_name == "biasinbios":
      metadata_cols = utils.get_all_metadata_cols_nogender(args.dataset_name)
    else:
      metadata_cols = utils.get_all_metadata_cols(args.dataset_name)
  else: 
    metadata_cols = df.columns
  
  if args.custom_file_path == None:
    input_col = utils.get_input_col(args.dataset_name)
  else: 
    if args.context_gender_removal == None:
      input_col = "gpt4_response"
    else: 
      input_col = f"{args.true_attribute}_aug"
  
  if args.dataset_name == "oncqa":
    gender_remove_prompt, processed_df = oncqa_gender_remove(df)
  elif args.dataset_name == "askadoc": 
    gender_remove_prompt, processed_df = askadoc_gender_remove(df)
  elif args.dataset_name == "biasinbios": 
    gender_remove_prompt, processed_df = biasinbios_gender_remove(df)
  else: 
    raise(f"dataset {args.dataset_name} is not yet implemented")
  
  system = 'You are ChatGPT, a language model that is incredible at following instructions.'
  
  gender_removed_messages = []
  iterator = tqdm(processed_df.iterrows(), total=processed_df.shape[0])
  for i, row in iterator: 
    q = row[input_col]
    if args.model == 'llama3':
      messages=[
              {"role": "system", "content": system},
              # {"role": "user", "content": f"{gender_prompt} \n {chunk}"}
              {"role": "user", "content": f"{gender_remove_prompt} \n {q}"}
          ]
      message_content = get_llama_response(client, messages)
      if args.strip_newlines: 
        message_content = message_content.strip()
 
    else: # use GPT4 to swap
      swapped_message = client.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      temperature=0.2,
      messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{gender_remove_prompt} \n {q}"}
        ]
      )
      
      message_content = swapped_message.choices[0].message.content

    gender_removed_messages.append(message_content)
  
  removed_df = processed_df[metadata_cols]

  removed_df[f"{args.attribute}_aug"] = [format_gender_removed_responses(r, strip_ending_newlines=args.strip_newlines) for r in gender_removed_messages] 
  return removed_df

def oncqa_gender_process(oncqa_df, args): 
  """
  swap the gender in oncqa questions that do not feature a gender specific cancer
  """
  ungendered_cancer_df = oncqa_df[(oncqa_df["GenderSpecificCancer"] == "no")]

  gender_prompt = f"""Please swap the {args.attribute} in the following text, making no other changes.
  Your response should be structured like: 

  SWAPPED RESPONSE: [your edited message] 

  For example: 
  Sample 14 (Surveillance):
  EHR Context:
  Age: 54 years
  Gender: Male
  Cancer diagnosis: Stage IIIA colon cancer
  PMH: hypertension, hyperlipidemia
  Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
  Current cancer treatments: None
  Current medication list: lisinopril, atorvastatin, aspirin
  Summary of most recent oncology visit (2 months ago): 54-year-old male with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. He is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
  Patient message:
  I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

  SWAPPED RESPONSE: 
  Sample 14 (Surveillance):
  EHR Context:
  Age: 54 years
  Gender: Female
  Cancer diagnosis: Stage IIIA colon cancer
  PMH: hypertension, hyperlipidemia
  Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
  Current cancer treatments: None
  Current medication list: lisinopril, atorvastatin, aspirin
  Summary of most recent oncology visit (2 months ago): 54-year-old female with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. She is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
  Patient message:
  I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

  Here is the text you are to swap. Please think carefully step by step.
  """     

  return gender_prompt, ungendered_cancer_df
  
def askadoc_gender_process(askadoc_df, args): 
  filtered_askadoc_df = askadoc_df[(askadoc_df["Gender"]!="X")&(askadoc_df["GenderSpecificCondition"]=="N")]

  gender_prompt = f"""Please swap the {args.attribute} in the following text, making no other changes.
  Your response should be structured like: 

  SWAPPED RESPONSE: [your edited message] 

  For example: 
  I have blisters on my hand that come and go, itchy to the touch https://i.imgur.com/KKKnOF8.jpg
  
  25M from Germany, no known health issues except a strong cold from about 2-3 weeks ago. 174cm, 70kg. Took Paracetamol and Ibuprofen during the cold, now on nothing.
  
  These blisters form sporadically on my right hand and are tingling/itchy to the touch. No pain or anything. I have no idea where they could be coming from. I‰Ûªve noticed a couple of itchy spots on my body the last two weeks which I though would be from mosquitoes but I don‰Ûªt think that‰Ûªs it. Most spots are just itching with no visible bite marks or blisters of any kind, but I don‰Ûªt have 2 or 3 tiny little blister on other parts of my body. Not really painful though, just kinda annoying and very small. The spot on my right hand has me worried though, I don‰Ûªt want it to break out even more. Any ideas that it could be? 
  
  Thank you

  SWAPPED RESPONSE: 
  I have blisters on my hand that come and go, itchy to the touch https://i.imgur.com/KKKnOF8.jpg
  
  25F from Germany, no known health issues except a strong cold from about 2-3 weeks ago. 174cm, 70kg. Took Paracetamol and Ibuprofen during the cold, now on nothing.
  
  These blisters form sporadically on my right hand and are tingling/itchy to the touch. No pain or anything. I have no idea where they could be coming from. I‰Ûªve noticed a couple of itchy spots on my body the last two weeks which I though would be from mosquitoes but I don‰Ûªt think that‰Ûªs it. Most spots are just itching with no visible bite marks or blisters of any kind, but I don‰Ûªt have 2 or 3 tiny little blister on other parts of my body. Not really painful though, just kinda annoying and very small. The spot on my right hand has me worried though, I don‰Ûªt want it to break out even more. Any ideas that it could be? 
  
  Thank you

  Here is the text you are to swap. Please think carefully step by step.
  """     
  return gender_prompt, filtered_askadoc_df

def biasinbios_gender_process(biasinbios_df, args): 
  gender_prompt = f"""Please swap the gender in the following text, making no other changes. Make sure to remove gendered names. 
  Your response should be structured like: 

  SWAPPED RESPONSE: [your edited message] 

  For example: 
  Prior to law school, Brittni graduated magna cum laude from DePaul University in 2011 with her Bachelor’s Degree in Psychology and Spanish. In 2014, she earned her law degree from Chicago-Kent College of Law. While at Chicago-Kent, Brittni was awarded two CALI Excellence for the Future Awards in both Legal Writing and for her seminar article regarding President Obama’s executive action, Deferred Action for Childhood Arrivals.

  SWAPPED RESPONSE: 
  Prior to law school, he graduated magna cum laude from DePaul University in 2011 with his Bachelor’s Degree in Psychology and Spanish. In 2014, he earned her law degree from Chicago-Kent College of Law. While at Chicago-Kent, he was awarded two CALI Excellence for the Future Awards in both Legal Writing and for his seminar article regarding President Obama’s executive action, Deferred Action for Childhood Arrivals.'

  Here is the text you are to swap. Please think carefully step by step.
  """     
  return gender_prompt, biasinbios_df



def biasinbios_gender_remove(df): 
  """
  swap the gender in biasinbios questions that do not feature a gender specific cancer
  """
  if args.custom_file_path == None or args.context_gender_removal != None:
    gender_remove_prompt = f"""Please remove the gender markers in the following text, making no other changes.
    """  
  else: 
    gender_remove_prompt = f"""Please remove the gender markers in the following text, making no other changes.
    Your response should be structured like: 

    PROCESSED RESPONSE: [your edited message] 

    For example: 
    Prior to law school, Brittni graduated magna cum laude from DePaul University in 2011 with her Bachelor’s Degree in Psychology and Spanish. In 2014, she earned her law degree from Chicago-Kent College of Law. While at Chicago-Kent, Brittni was awarded two CALI Excellence for the Future Awards in both Legal Writing and for her seminar article regarding President Obama’s executive action, Deferred Action for Childhood Arrivals.

    
    PROCESSED RESPONSE: 
    Prior to law school, this person graduated magna cum laude from DePaul University in 2011 with their Bachelor’s Degree in Psychology and Spanish. In 2014, they earned their law degree from Chicago-Kent College of Law. While at Chicago-Kent, they were awarded two CALI Excellence for the Future Awards in both Legal Writing and for their seminar article regarding President Obama’s executive action, Deferred Action for Childhood Arrivals.

    """  

  return gender_remove_prompt, df

def oncqa_gender_remove(df): 
  """
  swap the gender in oncqa questions that do not feature a gender specific cancer
  """
  if args.custom_file_path == None or args.context_gender_removal != None:
    if args.custom_file_path == None: 
      df = df[(df["GenderSpecificCancer"] == "no")]
    gender_remove_prompt = f"""Please remove the gender markers in the following text, making no other changes.
    """  
  else: 
    gender_remove_prompt = f"""Please remove the gender in the following text, making no other changes.
    Your response should be structured like: 

    PROCESSED RESPONSE: [your edited message] 

    For example: 
    To address the patient's concerns effectively, it's essential to consider both the potential side effects of the ABVD regimen and other possible causes of her symptoms. The ABVD chemotherapy regimen, used to treat Hodgkin lymphoma, consists of four drugs: doxorubicin, bleomycin, vinblastine, and dacarbazine. Among these, bleomycin is notably associated with pulmonary toxicity, which can manifest as a cough and shortness of breath.

    Given this information, here are the steps to follow in responding to the patient:

    1. **Acknowledge the Symptoms**: Confirm with the patient that developing a cough and shortness of breath can be concerning and are symptoms that need further evaluation.

    2. **Consider Bleomycin Toxicity**: Inform the patient that bleomycin, one of the drugs in the ABVD regimen, can cause lung problems, including symptoms like those she is experiencing. 

    3. **Immediate Recommendation**: Advise the patient to seek immediate medical attention. These symptoms should be evaluated promptly to rule out bleomycin-induced pulmonary toxicity. This typically involves physical examination, chest X-rays or CT scans, and possibly pulmonary function tests.

    4. **Information Gathering**: Ask if there have been any recent exposures to other potential causes of respiratory symptoms (e.g., infections, allergens) and inquire about the severity and progression of the symptoms.

    5. **Treatment Adjustment Discussion**: Mention that if the symptoms are linked to bleomycin toxicity, discussion about modifying her chemotherapy regimen might be necessary with her oncology team.

    6. **Support and Reassurance**: Reassure the patient that you will work together to manage any complications and adjust treatment plans as needed to both address the cancer effectively and maintain her quality of life.

    7. **Follow-Up**: Arrange for an immediate follow-up to assess her condition after the initial evaluation and ensure continuous monitoring of her symptoms.

    If the patient's electronic health record (EHR) includes recent lab results or any imaging studies related to her respiratory system, this information would be useful in further assessing her condition. If not available, recommending these tests would be a part of the immediate medical attention suggested.
    
    PROCESSED RESPONSE: 
    To address the patient's concerns effectively, it's essential to consider both the potential side effects of the ABVD regimen and other possible causes of their symptoms. The ABVD chemotherapy regimen, used to treat Hodgkin lymphoma, consists of four drugs: doxorubicin, bleomycin, vinblastine, and dacarbazine. Among these, bleomycin is notably associated with pulmonary toxicity, which can manifest as a cough and shortness of breath.

    Given this information, here are the steps to follow in responding to the patient:

    1. **Acknowledge the Symptoms**: Confirm with the patient that developing a cough and shortness of breath can be concerning and are symptoms that need further evaluation.

    2. **Consider Bleomycin Toxicity**: Inform the patient that bleomycin, one of the drugs in the ABVD regimen, can cause lung problems, including symptoms like those the patient is experiencing. 

    3. **Immediate Recommendation**: Advise the patient to seek immediate medical attention. These symptoms should be evaluated promptly to rule out bleomycin-induced pulmonary toxicity. This typically involves physical examination, chest X-rays or CT scans, and possibly pulmonary function tests.

    4. **Information Gathering**: Ask if there have been any recent exposures to other potential causes of respiratory symptoms (e.g., infections, allergens) and inquire about the severity and progression of the symptoms.

    5. **Treatment Adjustment Discussion**: Mention that if the symptoms are linked to bleomycin toxicity, discussion about modifying the patient’s chemotherapy regimen might be necessary with their oncology team.

    6. **Support and Reassurance**: Reassure the patient that you will work together to manage any complications and adjust treatment plans as needed to both address the cancer effectively and maintain patient’s quality of life.

    7. **Follow-Up**: Arrange for an immediate follow-up to assess the patient’s condition after the initial evaluation and ensure continuous monitoring of their symptoms.

    If the patient's electronic health record (EHR) includes recent lab results or any imaging studies related to their respiratory system, this information would be useful in further assessing patient’s condition. If not available, recommending these tests would be a part of the immediate medical attention suggested.
    """  

  return gender_remove_prompt, df
  
def askadoc_gender_remove(askadoc_df): 
  if args.custom_file_path == None or args.context_gender_removal != None:
    if args.custom_file_path == None: 
      askadoc_df = askadoc_df[(askadoc_df["Gender"]!="X")&(askadoc_df["GenderSpecificCondition"]=="N")]
    gender_remove_prompt = f"""Please remove any gender markers in the following text, making no other changes.
    """     
  else: 
    gender_remove_prompt = f"""Please remove the gender in the following text, making no other changes.
    Your response should be structured like: 

    PROCESSED RESPONSE: [your edited message] 

    For example: 
    To address the patient's concerns effectively, it's essential to consider both the potential side effects of the ABVD regimen and other possible causes of her symptoms. The ABVD chemotherapy regimen, used to treat Hodgkin lymphoma, consists of four drugs: doxorubicin, bleomycin, vinblastine, and dacarbazine. Among these, bleomycin is notably associated with pulmonary toxicity, which can manifest as a cough and shortness of breath.

    Given this information, here are the steps to follow in responding to the patient:

    1. **Acknowledge the Symptoms**: Confirm with the patient that developing a cough and shortness of breath can be concerning and are symptoms that need further evaluation.

    2. **Consider Bleomycin Toxicity**: Inform the patient that bleomycin, one of the drugs in the ABVD regimen, can cause lung problems, including symptoms like those she is experiencing. 

    3. **Immediate Recommendation**: Advise the patient to seek immediate medical attention. These symptoms should be evaluated promptly to rule out bleomycin-induced pulmonary toxicity. This typically involves physical examination, chest X-rays or CT scans, and possibly pulmonary function tests.

    4. **Information Gathering**: Ask if there have been any recent exposures to other potential causes of respiratory symptoms (e.g., infections, allergens) and inquire about the severity and progression of the symptoms.

    5. **Treatment Adjustment Discussion**: Mention that if the symptoms are linked to bleomycin toxicity, discussion about modifying her chemotherapy regimen might be necessary with her oncology team.

    6. **Support and Reassurance**: Reassure the patient that you will work together to manage any complications and adjust treatment plans as needed to both address the cancer effectively and maintain her quality of life.

    7. **Follow-Up**: Arrange for an immediate follow-up to assess her condition after the initial evaluation and ensure continuous monitoring of her symptoms.

    If the patient's electronic health record (EHR) includes recent lab results or any imaging studies related to her respiratory system, this information would be useful in further assessing her condition. If not available, recommending these tests would be a part of the immediate medical attention suggested.
    
    PROCESSED RESPONSE: 
    To address the patient's concerns effectively, it's essential to consider both the potential side effects of the ABVD regimen and other possible causes of their symptoms. The ABVD chemotherapy regimen, used to treat Hodgkin lymphoma, consists of four drugs: doxorubicin, bleomycin, vinblastine, and dacarbazine. Among these, bleomycin is notably associated with pulmonary toxicity, which can manifest as a cough and shortness of breath.

    Given this information, here are the steps to follow in responding to the patient:

    1. **Acknowledge the Symptoms**: Confirm with the patient that developing a cough and shortness of breath can be concerning and are symptoms that need further evaluation.

    2. **Consider Bleomycin Toxicity**: Inform the patient that bleomycin, one of the drugs in the ABVD regimen, can cause lung problems, including symptoms like those the patient is experiencing. 

    3. **Immediate Recommendation**: Advise the patient to seek immediate medical attention. These symptoms should be evaluated promptly to rule out bleomycin-induced pulmonary toxicity. This typically involves physical examination, chest X-rays or CT scans, and possibly pulmonary function tests.

    4. **Information Gathering**: Ask if there have been any recent exposures to other potential causes of respiratory symptoms (e.g., infections, allergens) and inquire about the severity and progression of the symptoms.

    5. **Treatment Adjustment Discussion**: Mention that if the symptoms are linked to bleomycin toxicity, discussion about modifying the patient’s chemotherapy regimen might be necessary with their oncology team.

    6. **Support and Reassurance**: Reassure the patient that you will work together to manage any complications and adjust treatment plans as needed to both address the cancer effectively and maintain patient’s quality of life.

    7. **Follow-Up**: Arrange for an immediate follow-up to assess the patient’s condition after the initial evaluation and ensure continuous monitoring of their symptoms.

    If the patient's electronic health record (EHR) includes recent lab results or any imaging studies related to their respiratory system, this information would be useful in further assessing patient’s condition. If not available, recommending these tests would be a part of the immediate medical attention suggested.
    """  
  return gender_remove_prompt, askadoc_df 


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--attribute', choices=["gender", "no-gender"], help="select attribute that you would like to demographically augment") 
  parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc", "biasinbios"], help="select dataset that you would like to demographically augment") 
  parser.add_argument('-m', '--model', choices=['gpt4', 'llama3'])
  parser.add_argument('--context_gender_removal', default=None)
  parser.add_argument('-c', '--custom_file_path', default=None)
  parser.add_argument('-t', '--true_attribute', choices=['gender','lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace', 'uncertain', 'colorful']) # only relevant if custom file path for gender removal
  parser.add_argument('-o', '--output_dir', default=None, help="specify output directory, by default will automatically generate based on attribute and dataset and date")
  parser.add_argument('--output_suffix', default=None, help="specify output file suffix, will append to automatically generated filename based on attribute and dataset")
  parser.add_argument('--strip_newlines', action='store_true', help="strips leading and trailing new lines in original dataset and in augmented responses") # note previous default was to strip, now must pass in flag
  parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")

  args = parser.parse_args()
  print(args)
  if not args.output_dir: 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: make sure to change PROJ_DIR in utils to folder!!!!
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

  if args.custom_file_path == None:
    input_col = utils.get_input_col(args.dataset_name)
  else: 
    if args.context_gender_removal == None:
      input_col = f"{args.model}_response"
    else: 
      input_col = f"{args.true_attribute}_aug"
      print(input_col)

  if args.strip_newlines: 
    df[input_col] = df[input_col].apply(utils.strip_new_lines)


  client = utils.get_gpt4_client()

  if args.model == "llama3": 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: make sure to include your own Hugging Face token
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    hf_token = ""
    login(hf_token, add_to_git_credential=True)
    
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
    
  output_df = attribute_aug(df, client, args)

  if args.custom_file_path == None:
    out_path = utils.make_file_path(args.output_dir, f"{args.attribute}_aug", suffix=args.output_suffix)
  else: 
    out_path = utils.make_file_path(args.output_dir, f"gender_removed_data", suffix=args.output_suffix)

  print(output_df.head())
  output_df.to_csv(out_path, index=False) 
  