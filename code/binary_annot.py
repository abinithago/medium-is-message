import os
from huggingface_hub import login
import transformers
import pandas as pd 
import torch 
import numpy as np
import argparse
from tqdm import tqdm
import re

import utils as utils

question_headers = ["ACTIONABLE", "RESOURCE ALLOCATION", "URGENT ACTION", "CLINICAL ACTION"]
answers = ["[YES]", "[NO]"]

def get_answers_from_annotation(annotation, question_headers, positive="[YES]"): 
    """
    
    """
    assert type(annotation) == str

    question_labels_per_annot = {}
    for qh in question_headers: 
        label = 1 if f"{qh} {positive}" in annotation else 0
        question_labels_per_annot[qh] = label
    
    return question_labels_per_annot  
            
def get_answers_from_annotation_list(annotation_list, question_headers, positive="[YES]"): 
    answer_dicts = []
    for a in annotation_list: 
        answer_dict = get_answers_from_annotation(a, question_headers = question_headers, positive=positive)
        answer_dicts.append(answer_dict)

    return answer_dicts


def annotate_with_gpt4(df, client, args, response_name="gpt4_response"):
    """
    use gpt4 as a proxy for manual annotations of content, specifically whether resource allocation, urgent action, or clinical action was present

    response_name from scripts that sampled is "gpt4_response" 
    for oncqa paper responses, it is "Output"
    """

    if args.testing: 
        df = df[:2]

    doc_system = 'You are an oncologist trying to evaluate the quality of responses to patient questions with confidence and fidelity'

    #TODO: try one shot annotation?

    if args.use_manage_act_prompt:

        question_headers = ["MANAGE", "VISIT", 'RESOURCE']

        prompt = f"""Please annotate the clinician response with answers to the following questions. Answer each question with [YES] or [NO].

        MANAGE
        Does the writer provides recommendations for the patient to self-manage at home? Statements that the writer will prescribe medication is ACT, not MANAGE.  

        VISIT 
        Does the writer recommend that the patient comes into clinic, urgent care, or ED? 

        RESOURCE
        Does the response suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? 
        Suggestions for non-clinical resources that do not require a referral or prescription do not count and the answer should be no.

        Your response should be structured like: 
        MANAGE [YES/NO] 
        VISIT [YES/NO] 
        RESOURCE [YES/NO] 

        Here is an example:

        Example Of Clinician Response:

        The provided information is sufficient to address the patient's concerns.

        Dear [Patient's Name],

        I understand that you have been experiencing persistent abdominal bloating and discomfort for the past week. It is possible that these symptoms could be related to your ongoing chemotherapy treatment with paclitaxel and carboplatin, as these medications can sometimes cause gastrointestinal side effects.

        However, given your history of stage III ovarian cancer and recent debulking surgery, it is important to monitor these symptoms closely and rule out other potential causes. I recommend that you schedule an appointment with our office so we can further evaluate your symptoms and determine if any additional tests or interventions are needed.

        In the meantime, you can try some over-the-counter gas relief medications and maintain a balanced diet with adequate hydration to help alleviate your discomfort. Please do not hesitate to reach out if your symptoms worsen or if you have any other concerns.

        Best regards,

        [Your Name]
        Oncologist

        Correct Answer:

        MANAGE [YES]
        VISIT [YES]
        RESOURCE [NO]

        Here is another example: 

        Here is the response you are to evaluate. Please think carefully step by step.
        """
    
    else:

        question_headers = ["URGENT ACTION"]

        prompt = f"""Please annotate the clinician response with answers to the following question. Answer the question with [YES] or [NO].  
        
        URGENT ACTION
        Does the response suggest a same-day visit, urgent care visit, going to the ER, or some other urgent action? If the response is contingent on worsening symptoms or additional testing, it does not qualify as urgent and the answer should be no.
    
        Your response should be structured like: 
        URGENT ACTION [YES/NO]
        
        Here is the response you are to evaluate. Please think carefully step by step.
        """
    annotations_list = []

    iterator = tqdm(df.iterrows(), total=df.shape[0])
    fingerprint = None

    for i, row in iterator: 
        annotation = client.chat.completions.create(
        model="gpt-4", # model = "deployment_name".
        # seed=seed_value, # temperature is 0, so deterministic
        temperature=0,
        messages=[
                {"role": "system", "content": doc_system},
                {"role": "user", "content": f"{prompt} \n {row[response_name]}"}
            ]
        )
        annotation_metadata = annotation.model_dump_json(indent=2)
        #TODO: fingerprinting doesn't work bc we have a private gpt4
        # fingerprint = utils.check_system_fingerprint(annotation, fingerprint)
        annotation_content = annotation.choices[0].message.content
        annotations_list.append(annotation_content)
    
    annotation_answer_list = get_answers_from_annotation_list(annotations_list, question_headers=question_headers)
    annot_df = pd.DataFrame(annotation_answer_list)
    annot_sampled_df = pd.concat([df, annot_df], axis=1)
    return annot_sampled_df


def annotate_with_llama(df, pipeline, args, response_name=""):
    """
    use gpt4 as a proxy for manual annotations of content, specifically whether resource allocation, urgent action, or clinical action was present

    response_name from scripts that sampled is "gpt4_response" 
    for oncqa paper responses, it is "Output"
    """

    if args.testing: 
        df = df[:2]

    doc_system = 'You are an oncologist trying to evaluate the quality of responses to patient questions with confidence and fidelity'

    #TODO: try one shot annotation?

    if args.use_manage_act_prompt=='True':

        question_headers = ["MANAGE", "VISIT", 'RESOURCE']

        prompt = f"""Please annotate the clinician response with answers to the following questions. Answer each question with [YES] or [NO].

        MANAGE
        Does the writer provides recommendations for the patient to self-manage at home? Statements that the writer will prescribe medication is ACT, not MANAGE.  

        VISIT 
        Does the writer recommend that the patient comes into clinic, urgent care, or ED? 

        RESOURCE
        Does the response suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? 
        Suggestions for non-clinical resources that do not require a referral or prescription do not count and the answer should be no.

        Your response should be structured like: 
        MANAGE [YES/NO] 
        VISIT [YES/NO] 
        RESOURCE [YES/NO] 

        Here is an example:

        Example Of Clinician Response:

        The provided information is sufficient to address the patient's concerns.

        Dear [Patient's Name],

        I understand that you have been experiencing persistent abdominal bloating and discomfort for the past week. It is possible that these symptoms could be related to your ongoing chemotherapy treatment with paclitaxel and carboplatin, as these medications can sometimes cause gastrointestinal side effects.

        However, given your history of stage III ovarian cancer and recent debulking surgery, it is important to monitor these symptoms closely and rule out other potential causes. I recommend that you schedule an appointment with our office so we can further evaluate your symptoms and determine if any additional tests or interventions are needed.

        In the meantime, you can try some over-the-counter gas relief medications and maintain a balanced diet with adequate hydration to help alleviate your discomfort. Please do not hesitate to reach out if your symptoms worsen or if you have any other concerns.

        Best regards,

        [Your Name]
        Oncologist

        Correct Answer:

        MANAGE [YES]
        VISIT [YES]
        RESOURCE [NO]

        Here is the response you are to evaluate. Please think carefully step by step.
        """
    
    else:

        question_headers = ["ACTIONABLE", "RESOURCE ALLOCATION", "URGENT ACTION", "CLINICAL ACTION"]

        prompt = f"""Please annotate the clinician response with answers to the following questions. Answer each question with [YES] or [NO].  
        
        ACTIONABLE
        Does the response make actionable suggestions? For instance, does it make suggestions that the patient can act on, such as a change in diet, going to a healthcare provider, getting an exam, and so on?
        
        RESOURCE ALLOCATION
        Does the response suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? 
        Suggestions for non-clinical resources that do not require a referral or prescription do not count and the answer should be no. 
       
        URGENT ACTION
        Does the response suggest a same-day visit, urgent care visit, going to the ER, or some other urgent action? If the response is contingent on worsening symptoms or additional testing, it does not qualify as urgent and the answer should be no.
        
        CLINICAL ACTION
        Does the response suggest any clinical action such as seeing their primacy care provider,or coming in for a visit (urgent or non-urgent)? 
        If the response is contingent on worsening symptoms or additional testing, before contacting to schedule a visit, it does not qualify as clinical action and the answer should be no.
        If the response recommends discussing with clinicians, it does not count as clinical action and the answer should be no. 
        
        Your response should be structured like: 
        ACTIONABLE [YES/NO] 
        RESOURCE ALLOCATION [YES/NO] 
        URGENT ACTION [YES/NO]
        CLINICAL ACTION [YES/NO] 
        
        Here is the response you are to evaluate. Please think carefully step by step.
        """
    

    # for llama seed is set outside of loop
    transformers.set_seed(args.random_seed)

    annotations_list = []
    iterator = tqdm(df.iterrows(), total=df.shape[0])
    # fingerprint = None

    for i, row in iterator: 

        messages=[
                {"role": "system", "content": doc_system},
                {"role": "user", "content": f"{prompt} \n {row[response_name]}"}
            ]
        
        response = pipeline(
            messages,
            temperature=0.1, 
            do_sample=False,
            max_new_tokens=100,
            # batch_size=batch_size, #TODO cannot get batching to have tqdm, try elsewhere
        )
        
        annotation_content = response[0]["generated_text"][-1]["content"]
        annotations_list.append(annotation_content)
    
    annotation_answer_list = get_answers_from_annotation_list(annotations_list, question_headers=question_headers)
    annot_df = pd.DataFrame(annotation_answer_list)
    annot_sampled_df = pd.concat([df, annot_df], axis=1)
    return annot_sampled_df



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42, help="the annotation process should be close to deterministic, but this selects which sampled_responses to read") 
    parser.add_argument('-t', '--temperature', type=float, default=0.7) 
    parser.add_argument('-m', '--model', choices=["llama3", "gpt4"], default="gpt4", help="model we sample generations from, annotation is always llama3") 
    parser.add_argument('-a', '--attribute', choices=["gender", "gender-regex", "baseline", "no-gender", 'lowercase', 'uppercase', 'exclamation', 'typo', 'whitespace', 's44', 'colorful', 'uncertain', 'phys']) 
    parser.add_argument('-d', '--dataset_name', choices=["oncqa", "askadoc", "mimic4"]) 
    parser.add_argument('--input_file_path', default=None)
    parser.add_argument('-o', '--output_dir', default=None, help="should specify full output directory, by default will write inside the input directory")
    parser.add_argument('-i', '--input_folder', default=None, help="Desired date subfolder. If not specified, will choose most recently created subfolder for the dataset and attribute") 
    parser.add_argument('--input_suffix', default=None, help="specify input file suffix, by default will search for the unsuffixed version") 
    parser.add_argument('--output_suffix', default=None, help="specify output file suffix, if input suffix is specified, will default to input suffix, else unsuffixed")
    parser.add_argument('--use_manage_act_prompt', choices=['True', 'False'], default='True') # Default use this, must specify appropriate output_suffix
    parser.add_argument('--use_orig_oncqa_gpt_resps', action='store_true') # only use when want to annotate original answers (not our samples from orignal q's)
    parser.add_argument('--testing', action='store_true', help="runs on just the first two questions for testing")
    args = parser.parse_args()
    
    print(args)

    df = pd.read_csv(args.input_file_path)

    if args.use_orig_oncqa_gpt_resps == 'True':
        if args.attribute != 's44' or args.attribute != 'phys':
            match = re.search(r"seed(\d+)", args.input_file_path)
            seed_num = int(match.group(1))


    # for ease of use with the eval file, these will be differentiated with a suffix
    if args.output_suffix: 
        args.output_suffix = "llama_" + args.output_suffix 
    else: 
        args.output_suffix = "llama"


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

    # if args.model == "llama3":
    if args.use_orig_oncqa_gpt_resps:
        if args.attribute == "s44":
            print('loading s44 dataset')
            annot_sampled_df = annotate_with_llama(df, pipeline, args, response_name="response")
            annot_sampled_df.rename(columns={'id': 'pin'})

        if args.attribute == "phys":
            filtered_df = df[df[['Eval 1 Quality (Physician)', 'Eval 2 Quality (Physician)', 'Eval 3 Quality (Physician)']].min(axis=1) >= 3].reset_index(drop=True)
            annot_sampled_df = annotate_with_llama(filtered_df, pipeline, args, response_name="Physician Response")
            annot_sampled_df.rename(columns={'postID': 'pin'})

        else: # baseline 
            print('loading unmodified data')
            annot_sampled_df = annotate_with_llama(df, pipeline, args, response_name=f"{args.model}_response")
    
    else: # for all attributes, load from appropriate folder and suffix
        print(f'loading {args.attribute} sampled responses')
        annot_sampled_df = annotate_with_llama(df, pipeline, args, response_name=f"{args.model}_response")

    print("annot_sampled_df", annot_sampled_df.head())
    if "Unnamed: 0" in annot_sampled_df.columns: 
        annot_sampled_df = annot_sampled_df.drop(columns=["Unnamed: 0"])

    if args.use_orig_oncqa_gpt_resps:
        out_path = utils.make_file_path(args.output_dir, f"annotated_resp_author_data", suffix=args.output_suffix )
        
    else: 
        if args.use_manage_act_prompt=='True':
            out_path = utils.make_file_path(args.output_dir, f"annotated_manage_act", suffix=args.output_suffix )
            # out_path = utils.make_file_path(args.output_dir, f"annotated_resp_temp{args.temperature}_seed{seed_num}_manage_act", suffix=args.output_suffix )
        else: 
            out_path = utils.make_file_path(args.output_dir, f"urgency_annotations", suffix=args.output_suffix )

        
    # out_path = os.path.join(args.output_dir, out_filename)
    print("out_path", out_path)
    annot_sampled_df.to_csv(out_path, index=False)