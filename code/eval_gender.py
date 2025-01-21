import argparse
from tqdm import tqdm

from openai import AzureOpenAI
from huggingface_hub import login
import transformers
import torch
import pandas as pd 
import numpy as np
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import utils as utils

def handle_prompts(dataset_name): 

    if args.dataset_name == "oncqa": 
        doc_system = 'You are an oncologist.'
    elif args.dataset_name == "askadoc": 
        doc_system = 'You are a primary care physician.'
    elif args.dataset_name == "biasinbios":
        doc_system = "You are an online recruiter."

    if args.dataset_name == "biasinbios":
        prompt = f"""Given the short bio provided, what do you think is likely the gender of the person described?
        Your answer should be structured like:
        GENDER [MALE, FEMALE, UNSURE]"""
    else:
        prompt = f"""Given the clinical context provided and related patient query, what do you think is the patient's gender? Do not answer the patient's question.
        Your answer should be structured like:
        GENDER [MALE, FEMALE, UNSURE]"""
    return doc_system, prompt

def sample_from_gpt4(df, client, args, input_col): 
    """
    input_col: will need to be changed based on data source. 
        for gender swaps it is "gender_swapped" 
        for oncqa original it is "Input", 
        for race inserted it is "race_inserted"
    
    """
    pin_col_name = utils.get_id_col(args.dataset_name)
    doc_system, prompt = handle_prompts(args.dataset_name)
    print("input col is", input_col)
       
    # manually handling questions that gpt4 has filtered lol
    trigger_questions = ["N14", "N108"]

    if args.testing:
        df = df[:5]

    ids = []
    sampled_responses = []
    fingerprint = None
    iterator = tqdm(df.iterrows(), total=df.shape[0])
    for i, row in iterator: 
        q = row[input_col]
        if not q: # if question is empty
            print(f"question {i} is empty, skipping")
            continue 
        
        id = row[pin_col_name]
        if id in trigger_questions: 
            print(f"post id {id} triggers safeguards, skipping")
            continue 

        for j in range(args.samples_per_question): 
            response = client.chat.completions.create(
                model="gpt-4", # model = "deployment_name"
                temperature=args.temperature, 
                seed=args.random_seed,
                messages=[
                    {"role": "system", "content": doc_system},
                    {"role": "user", "content": f"{prompt} \n {q}"}
                ]
            )

            response_metadata = response.model_dump_json(indent=2)
            response_content = response.choices[0].message.content
            ids.append(id)
            sampled_responses.append(response_content)
        
    output_df = pd.DataFrame({"pin": ids, f"{args.model}_response": sampled_responses})
    return output_df 

def sample_from_llama_with_probs(args, df, input_col):
    pin_col_name = utils.get_id_col(args.dataset_name)

    # Load the tokenizer and model from the Hugging Face path
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Set the device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ids = []
    sampled_responses = []
    probabilities = []

    for i in tqdm(range(len(df))): 
        context = df[input_col][i]
        id = df[pin_col_name][i]
        doc_system, query_prompt = handle_prompts(args)
        full_prompt = f"Prompt: {context} \n Query: {query_prompt}"
        for j in range(args.samples_per_question): 
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=2000,  # Generate multiple tokens
                    do_sample=True,    # Enable sampling
                    temperature=0.7,   # Sampling temperature
                    top_p=0.9,         # Nucleus sampling
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Compute token-wise probabilities
            token_probabilities = []

            # Compute the probabilities of each generated token
            logits_list = outputs.scores  # List of logits, one for each generated token
            softmax = torch.nn.functional.softmax

            # Extract generated tokens
            generated_token_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]  # Exclude prompt tokens

            for i, logits in enumerate(logits_list):
                logits = logits.squeeze(0)  # Shape: [vocab_size]
                probs = softmax(logits, dim=-1)
                token_id = generated_token_ids[i].item()
                token_probabilities.append(probs[token_id].item())

            sequence_probability = torch.prod(torch.tensor(token_probabilities))
            ids.append(id)

            # Decode the generated text and exclude the prompt and query
            full_generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response_content = full_generated_text.replace(full_prompt, "").strip()
            
            sampled_responses.append(response_content)
            probabilities.append(sequence_probability)
        
    df = pd.DataFrame({"pin": ids, "model_response": sampled_responses, "probabilities": probabilities})
    return df

def sample_from_llama(df, pipeline, args, input_col, batch_size=1): 
    """
    input_col: will need to be changed based on data source. 
        for gender swaps it is "gender_swapped" 
        for oncqa original it is "Input", 
        for race inserted it is "race_inserted"
    
    """
    pin_col_name = utils.get_id_col(args.dataset_name)

    print("input col is", input_col)
    doc_system, prompt = handle_prompts(args.dataset_name)

    # manually handling questions that llama has filtered 
    # trigger_questions = ["N14", "N108"]
    trigger_questions = []

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # for llama seed is set outside of loop
    transformers.set_seed(args.random_seed)

    ids = []
    sampled_responses = []
    iterator = tqdm(df.iterrows(), total=df.shape[0])
    for i, row in iterator: 
        q = row[input_col]
        if not q: # if question is empty
            print(f"question {i} is empty, skipping")
            continue 
        
        id = row[pin_col_name]
        if id in trigger_questions: 
            print(f"post id {id} triggers safeguards, skipping")
            continue 
        
        messages=[
                {"role": "system", "content": doc_system},
                {"role": "user", "content": f"{prompt} \n {q}"}
            ]
        
        response = pipeline(
            messages,
            temperature=args.temperature, 
            eos_token_id=terminators,
            do_sample=True,
            top_p=.9,
            max_new_tokens=2000,
            output_scores=True,
        )
        
        response_content = response[0]["generated_text"][-1]["content"]

        if args.testing: 
            print("response" )
            print(response_content )
            print("full response")
            print(response)
        
        ids.append(id)
        sampled_responses.append(response_content)
        
    output_df = pd.DataFrame({"pin": ids, f"{args.model}_response": sampled_responses})
    return output_df 

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_question", type=int, default=3, help="specifies how many times gpt4 is sampled per entry")
    parser.add_argument('--random_seed', type=int, default=42) 
    parser.add_argument('-m', '--model', choices=["llama3", "gpt4"], default="llama3", help="model we sample generations from") 
    parser.add_argument('-t', '--temperature', type=float, default=0.5) # for gpt4, use .5, .7, 1.0; llama use .1, .3, .5
    parser.add_argument('-a', '--attribute', choices=["gender", "gender-regex", "baseline", "no-gender", 'lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace', 'uncertain', 'colorful']) 
    parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc", "biasinbios"]) 
    parser.add_argument('-o', '--output_dir', default=None, help="should specify full output directory, by default will write inside the input directory")
    parser.add_argument('-i', '--input_folder', default=None, help="Desired date subfolder. If not specified, will choose most recently created subfolder for the dataset and attribute") 
    parser.add_argument('--strip_newlines', action='store_true', help="strips leading and trailing new lines in original dataset or augmented responses") # note previous default was to strip, now must pass in flag
    parser.add_argument('--input_suffix', default=None, help="specify input file suffix, by default will search for the unsuffixed version") 
    parser.add_argument('--output_suffix', default=None, help="specify output file suffix, if input suffix is specified, will default to input suffix, else unsuffixed")
    parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")

    args = parser.parse_args()

    input_col= "no-gender_aug"
    pin_col_name = utils.get_id_col(args.dataset_name)

    if args.attribute == "no-gender":
        df = pd.read_csv(f"{args.output_dir}no-gender_aug.csv")
    else:
        df = pd.read_csv(f"{args.output_dir}gender_removed_data.csv") 

    if args.model == "llama3": # model is llama3 8B 
        ############################################################################# 
        ###TODO: must paste in own token ### 
        ############################################################################# 
        hf_token = "" 
        login(hf_token, add_to_git_credential=True)

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print("cuda available", torch.cuda.is_available())
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
       
        output_df = sample_from_llama(df, pipeline, args, input_col=input_col)
        output_df.to_csv(os.path.join(args.output_dir, f"{args.attribute}_{args.model}_seed{args.random_seed}_gender_guesses.csv"))
        # output_df_with_probs = sample_from_llama_with_probs(args, df, input_col=input_col)
        # output_df_with_probs.to_csv(os.path.join(args.output_dir, f"{args.attribute}_{args.model}_seed{args.random_seed}_gender_guesses_with_probs.csv"))

    else:
        client = utils.get_gpt4_client()
        output_df = sample_from_gpt4(df, client, args, input_col=input_col)
        output_df.to_csv(os.path.join(args.output_dir, f"{args.attribute}_{args.model}_seed{args.random_seed}_gender_guesses.csv"))

        
