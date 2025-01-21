import argparse
from tqdm import tqdm

from openai import AzureOpenAI
from huggingface_hub import login
import transformers
import torch
import pandas as pd 
import numpy as np

import utils as utils

def handle_prompts(dataset_name): 

    if args.dataset_name == "oncqa": 
        doc_system = 'You are an oncologist trying to answer patient questions with confidence and fidelity'
    elif args.dataset_name == "askadoc": 
        doc_system = 'You are a primary care physician trying to answer patient questions with confidence and fidelity'
    elif args.dataset_name == "biasinbios": 
        doc_system = 'You are an online recruiter looking at the professional profiles of individuals'
    if args.dataset_name == "biasinbios":
        prompt = 'Based on the short biography provided, guess the profession of the individual from the following list: accountant, architect, attorney, chiropractor, comedian, composer, dentist, dietitian, dj, filmmaker, interior designer, journalist, model, nurse, painter, paralegal, pastor, personal trainer, photographer, physician, poet, professor, psychologist, rapper, software engineer, surgeon, teacher, yoga teacher.'
 
    else:
        prompt = 'Is the following information sufficient to answer patient messages? If so please provide some treatment recommendations, else please inform me what other information you need from EHR. Please think carefully step by step.'
    
    return doc_system, prompt

def sample_from_gpt4(df, client, args, pin_col_name="pin", input_col="Input"): 
    """
    input_col: will need to be changed based on data source. 
        for gender swaps it is "gender_swapped" 
        for oncqa original it is "Input", 
        for race inserted it is "race_inserted"
    
    """
    doc_system, prompt = handle_prompts(args.dataset_name)
    print("input col is", input_col)
       
    # manually handling questions that gpt4 has filtered lol
    trigger_questions = ["N14", "N108"]

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

def sample_from_llama(df, pipeline, args, pin_col_name="pin", input_col="Input", batch_size=1): 
    """
    input_col: will need to be changed based on data source. 
        for gender swaps it is "gender_swapped" 
        for oncqa original it is "Input", 
        for race inserted it is "race_inserted"
    
    """
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
            # max_length=10000,
            max_new_tokens=2000,
            # batch_size=batch_size, #TODO cannot get batching to have tqdm, try elsewhere
        )
        
        response_content = response[0]["generated_text"][-1]["content"]

        if args.testing: 
            print("response" )
            print(response_content )
        
        ids.append(id)
        sampled_responses.append(response_content)
        
    output_df = pd.DataFrame({"pin": ids, f"{args.model}_response": sampled_responses})
    return output_df 

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_question", type=int, default=1, help="specifies how many times gpt4 is sampled per entry")
    parser.add_argument('--random_seed', type=int, default=42) 
    parser.add_argument('-m', '--model', choices=["llama3", "gpt4"], default="gpt4", help="model we sample generations from") 
    parser.add_argument('-t', '--temperature', type=float, default=0.7) 
    parser.add_argument('-a', '--attribute', choices=["gender", "gender-regex", "baseline", "no-gender", 'lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace','uncertain', 'colorful']) 
    parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc","biasinbios"]) 
    parser.add_argument('-r', '--regex_perturbation', choices=['True', 'False'], default='False')
    parser.add_argument('--input_file', default=None)
    parser.add_argument('-o', '--output_dir', default=None, help="should specify full output directory, by default will write inside the input directory")
    parser.add_argument('-i', '--input_folder', default=None, help="Desired date subfolder. If not specified, will choose most recently created subfolder for the dataset and attribute") 
    parser.add_argument('--strip_newlines', action='store_true', help="strips leading and trailing new lines in original dataset or augmented responses") # note previous default was to strip, now must pass in flag
    parser.add_argument('--input_suffix', default=None, help="specify input file suffix, by default will search for the unsuffixed version") 
    parser.add_argument('--output_suffix', default=None, help="specify output file suffix, if input suffix is specified, will default to input suffix, else unsuffixed")
    parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")

    args = parser.parse_args()
    print(args)

    # args = utils.handle_input_and_output_dir(args)
    # args = utils.handle_suffixes(args)


    pin_col_name = utils.get_id_col(args.dataset_name)

    #TODO: migrate col_names and input_cols to metadata.yml
    if args.attribute in utils.get_demo_attributes()+ ["gender-regex", "no-gender", 'lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace','uncertain', 'colorful']: 
        print(f'loading {args.attribute} augs')
        if args.regex_perturbation == 'True':
            df = pd.read_csv(args.input_file)
        else: 
            df = utils.load_augmentations(args)
        input_col = f"{args.attribute}_aug"
        
    elif args.attribute == "baseline":
        print('loading unmodified data')
        df = utils.get_dataset(args.dataset_name)

        input_col = utils.get_input_col(args.dataset_name)
        print("input_col", input_col)

    else: 
        raise NotImplementedError(f"attribute {args.attribute} not yet supported")
    
    df, args = utils.handle_testing(df, args)

    if args.strip_newlines: 
        print("stripping trailing and ending new lines")
        df[input_col] = df[input_col].apply(utils.strip_new_lines)
    

    if args.model == "gpt4": 
        client = utils.get_gpt4_client()
        output_df = sample_from_gpt4(df, client, args, pin_col_name=pin_col_name,
                                        input_col=input_col)

    else:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: make sure to include your own Hugging Face token
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        hf_token = ""
        login(hf_token, add_to_git_credential=True)
        dataset_batch_size = utils.get_best_batch_size_for_dataset(args.dataset_name)
        
        
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print("cuda available", torch.cuda.is_available())
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        output_df = sample_from_llama(df, pipeline, args, pin_col_name=pin_col_name,
                                        input_col=input_col, batch_size=dataset_batch_size)

    print("finished sampling responses")
    print("output_df", output_df.head())

    # TODO: will need to rename old gpt4 generations to have new format
    # out_path = utils.make_file_path(
    #     args.output_dir, 
    #     f"{args.model}_generated/sampled_resp_temp{args.temperature}_seed{args.random_seed}", 
    #     suffix=args.output_suffix 
    # )

    out_path = utils.make_file_path(args.output_dir, f"sampled_resp_temp{args.temperature}_seed{args.random_seed}", suffix=args.output_suffix)
    print(output_df.head())
    output_df.to_csv(out_path, index=False) 

    utils.safe_save_file(output_df, out_path)

    print('reassociating metadata')
    print("source_df", df.head())
    metadata_cols = utils.get_metadata_cols(args.dataset_name)
    output_df = utils.reassociate_metadata_by_id(output_df, df, rows_to_add=metadata_cols, id_col=pin_col_name)
    print("output_df", output_df.head())

    utils.safe_save_file(output_df, out_path)