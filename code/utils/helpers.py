import os 
import pandas as pd
import datetime
import yaml
from openai import AzureOpenAI

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: make sure to replace with own project directory
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROJ_DIR = ""
METADATA_FILE = os.path.join(PROJ_DIR, "metadata.yml")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: make sure to replace with your API key information
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_gpt4_client(): 
    client = AzureOpenAI(
        api_key = "",  
        api_version = "",
        azure_endpoint = ""
    )
    return client 

def get_proj_dir(): 
    return PROJ_DIR

def get_genders():
    return ["M", "F"] 

def get_attr_vals(attribute): 
    if attribute in ["gender", "gender-regex"]: 
        return get_genders()
    else: 
        raise NotImplementedError(f"function will not work as expected, for {attribute}. please implement")

def load_metadata_yaml(dataset_name): 
    with open(METADATA_FILE, 'r') as file:
        metadata_dict = yaml.safe_load(file)
        return metadata_dict[dataset_name]
    
def get_gender_col(dataset_name): 
    return load_metadata_yaml(dataset_name)["gender_col"]

def get_attr_col(args): 
    return load_metadata_yaml(args.dataset_name)[args.attribute+"_col"]

def get_demo_attributes():
    return ["gender"]
    
def get_input_col(dataset_name): 
    # returns input col of dataset 
    return load_metadata_yaml(dataset_name)["input_col"]

def get_id_col(dataset_name): 
    # returns input col of dataset 
    return load_metadata_yaml(dataset_name)["id_col"]

def get_gender_specific_col(dataset_name): 
    # returns col name corresponding to manual annotations of whether 
    # condition is only possible for a particular gender
    return load_metadata_yaml(dataset_name)["gender_specific_col"]

def get_best_batch_size_for_dataset(dataset_name): 
    # returns manually tested best batchsize for sampling for each dataset
    return load_metadata_yaml(dataset_name)["best_batch_size"]

def get_dataset(dataset_name): 
    """
    gets original version of datasets
    """
    rel_path = load_metadata_yaml(dataset_name)["rel_file_path"]
    file_path = os.path.join(get_proj_dir(), rel_path)
    return pd.read_csv(file_path) 

def get_metadata_cols(dataset_name): 
    """
    returns metadata cols, without id and orig input_col
    """
    return load_metadata_yaml(dataset_name)["metadata_cols"]

def get_all_metadata_cols(dataset_name): 
    """
    returns metadata cols, including the id and original_input_col
    """
    metadata = load_metadata_yaml(dataset_name)
    metadata_cols = [metadata["id_col"]] + metadata["metadata_cols"] + [metadata["gender_specific_col"], metadata["input_col"]]
    return metadata_cols 

def get_all_metadata_cols_nogender(dataset_name): 
    """
    returns metadata cols, including the id and original_input_col
    """
    metadata = load_metadata_yaml(dataset_name)
    metadata_cols = [metadata["id_col"]] + metadata["metadata_cols"] + [metadata["input_col"]]
    return metadata_cols 

def last_created_subfolder(directory):
    """
    retrieves last date subfolder created as default folder of argparse
    """
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time)
    return os.path.join(directory, sorted_items[-1])

def handle_testing(df, args): 
    if args.testing: 
        df = df[:2]
        if args.output_suffix: 
            args.output_suffix = args.output_suffix + "_test"
        else: 
            args.output_suffix = "test"
    return df, args
    
def handle_input_dir(args): 
    #TODO: test input and output dirs are working after reorg
    """
    fills in input directory based on demographic selections
    """
    ATTRIBUTE_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}_augs/") 
    if args.attribute == "baseline": 
        print("baseline")
        ATTRIBUTE_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}/") 

    if args.input_folder: 
        full_filepath = os.path.join(ATTRIBUTE_DIR, args.input_folder)
        args.input_folder = full_filepath
        print(f"input directory {args.input_folder}")
    else: 
        # for demographic attributes, search for most recent augmentations 
        if args.attribute in get_demo_attributes():
            args.input_folder = last_created_subfolder(ATTRIBUTE_DIR)
            print(f"no input directory specified, defaulting to input directory {args.input_folder}")
        else: # for original datasets, logic is later 
            args.input_folder = None 
    
    return args

def handle_output_dir(args):
    #TODO: test input and output dirs are working after reorg
    """
    sets input directory as output directory, unless separate output directory specified
    """
    ATTRIBUTE_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}_augs/") 
    if args.attribute == "baseline": 
        print("baseline")
        ATTRIBUTE_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/{args.attribute}/") 

    if not args.output_dir:  # if no output_dir is specified, write into corresponding demo input dir
        if args.input_folder: 
            print(f"no output directory specified, defaulting to input directory {args.input_folder}")
            args.output_dir = args.input_folder
        else: # if no input_dir specified, create a new folder 
            output_dir = ATTRIBUTE_DIR
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)  
            args.output_dir = output_dir
    return args 

def handle_baseline_dir(args): 
    """
    gets the corresponding baseline directory or finds the most recently made baseline
    """
    BASELINE_DIR = os.path.join(PROJ_DIR, f"data/{args.dataset_name}/baseline/") 
    if args.baseline_input_folder: 
        full_filepath = os.path.join(BASELINE_DIR, args.baseline_input_folder)
        args.baseline_input_folder = full_filepath
        print(f"baseline input directory {args.baseline_input_folder}")
    else: 
        args.baseline_input_folder = last_created_subfolder(BASELINE_DIR)
    
    return args 

def handle_analysis_output_dir(args): 
    """
    finds input directory based on demographic selections
    sets input directory as output directory, unless separate output directory specified
    """
    if not args.output_dir:  # if no output_dir is specified, write into corresponding demo input dir
        output_dir = os.path.join(PROJ_DIR, f"analysis/{args.dataset_name}/{args.attribute}/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)  
        args.output_dir = output_dir
    
    return args 

def handle_input_and_output_dir(args, analysis=False): 
    """
    wrapper function to use both, must handle input dir before output dir for defaults
    """
    args = handle_input_dir(args)
    if analysis: 
        args = handle_baseline_dir(args)
        args = handle_analysis_output_dir(args)
    else: 
        args = handle_output_dir(args)
    return args 

def handle_suffixes(args, baseline=False):
    """
    if input_suffix is set but not output_suffix, conform output_suffix to input
    otherwise, return as is
    if neither input or output suffix are set, both stay None
    if output suffix is set but not input suffix, return as is

    if the baseline_suffix is toggled, will default input_suffix to the baseline_suffix (for eval)
    """
    if baseline:
        if args.baseline_suffix and not args.input_suffix: 
            args.input_suffix  = args.baseline_suffix
    
    if args.input_suffix and not args.output_suffix: 
        args.output_suffix = args.input_suffix 

    return args 

def reassociate_metadata_by_id(df, source_df, rows_to_add=["ActiveOrSurveill","Gender","Age", "GenderSpecificCancer"], id_col="pin"):
    """
    df: df you would like to reassociate metadata to 
    source_df: df that contains all ids of patients and metadata you would like to associate
    rows_to_add: rows from source_df to add to df 
    id_col: the identifier col in both df and source_df
    """ 
    # must set id_col to be index of metadata_df
    metadata_df = source_df.set_index(id_col)

    def reassoc_by_row(row): # Pandas treats rows as Series under the hood
        #must .loc select metadata_row in order to be a Series 
        # this is hardcoded because all dataset ids are being standardized as "pin"
        metadata_row = metadata_df.loc[row["pin"]] 
        metadata_info = metadata_row[rows_to_add]
        # concat must happen between Series to stay a row
        row = pd.concat([row, metadata_info])
        return row
    
    return df.apply(reassoc_by_row, axis=1)

def strip_new_lines(orig_input): 
    """
    removes leading and trailing spaces and new lines
    intended for OncQA but could be good preprocessing step for all datasets 
    """
    if not orig_input: # None, empty string, pd.nan etc 
        return orig_input
    stripped_response = orig_input.strip(" \n")

    #print("difference:", orig_input.replace(stripped_response, ''))

    return stripped_response

def make_file_path(folder, file_name, suffix=None, file_type="csv"): 
    """
    make full filepath based off file name header, suffixes, and extensions
    no _ needed for suffix, no . needed for extension
    """
    if suffix: 
        file_path = os.path.join(folder, file_name + f"_{suffix}.{file_type}")
    else: 
        file_path = os.path.join(folder, file_name + f".csv")
    return file_path 

def safe_save_file(df, file_path): 
    """makes any missing subdirectories in the filepath before saving file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def load_augmentations(args):

    file_path = make_file_path(args.input_folder, f"{args.attribute}_aug", suffix=args.output_suffix)
    #TODO: decide if na filter is good 
    return pd.read_csv(file_path, na_filter=False) 

def load_sampled_responses(args):
    """
    loads sampled responses for a model, particlar seed, and temperature
    """

    file_path = make_file_path(args.input_folder, f"{args.model}_generated/sampled_resp_temp{args.temperature}_seed{args.random_seed}", suffix=args.input_suffix)

    return pd.read_csv(file_path) 

def load_sampled_responses_not_from_args(input_folder, model, random_seed, temperature, input_suffix=None):
    """
    loads sampled responses for a particlar seed and temperature
    """
    file_path = make_file_path(input_folder, f"{model}_generated/sampled_resp_temp{temperature}_seed{random_seed}", suffix=input_suffix)

    return pd.read_csv(file_path) 

def load_all_sampled_responses(args, temperature, baseline=False): 
    df_list = [] 
    for seed in args.seeds: 
        if baseline: 
            annot_df = load_sampled_responses_not_from_args(args.baseline_input_folder, args.model, seed, temperature, input_suffix=args.baseline_suffix)
        else: 
            annot_df = load_sampled_responses_not_from_args(args.input_folder, args.model, seed, temperature, input_suffix=args.input_suffix)
        annot_df["seed"] = seed
        df_list.append(annot_df)
    
    return  pd.concat(df_list)

def load_annot_responses(input_folder, model, random_seed, temperature, input_suffix=None):
    """
    loads sampled responses for a seed and temperature
    """

    file_path = make_file_path(input_folder, f"{model}_generated/annotated_resp_temp{temperature}_seed{random_seed}", suffix=input_suffix)

    return pd.read_csv(file_path) 

def load_all_annot_responses(args, temperature, baseline=False): 
    """
    load sampled responses for all seeds, expects the extra arguments from eval.py
     - seeds  
     - baseline_input_folder
     - baseline_suffix
    make longform data for cleanness usage 
    """

    df_list = [] 
    for seed in args.seeds: 
        if baseline: 
            annot_df = load_annot_responses(args.baseline_input_folder, args.model, seed, temperature, input_suffix=args.baseline_suffix)
        else: 
            annot_df = load_annot_responses(args.input_folder, args.model, seed, temperature, input_suffix=args.input_suffix)
        annot_df["seed"] = seed
        df_list.append(annot_df)
    
    return  pd.concat(df_list)

def check_system_fingerprint(resp_obj, initial_fingerprint): 
    """
    monitors for changes in system fingerprint during runs, 
    if fingerprint changes and weirdness in results, may need to  rerun 
    """
    curr_fingerprint = resp_obj.system_fingerprint 
    if not initial_fingerprint: # initially fingerprint is None
        print("initial fingerprint", curr_fingerprint)
    elif initial_fingerprint != curr_fingerprint: 
        print(f"!!! fingerprint has changed from {initial_fingerprint} to {curr_fingerprint}!!!")
    else: 
        # All is normal, fingerprints align 
        pass

    return curr_fingerprint

    