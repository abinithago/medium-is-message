# MIM
The Medium is the Message: How Non-Clinical Information Shapes Clinical Decisions in LLMs

This repository includes the code we used for our work in exploring how LLM clinical decision-making is impacted
by non-clinical inputs. 

The general files work as follows:
| File | Description |
| --- | --- |
| perturb_data_llm.py | Performs perturbations for explicit gender changes (swapping and removal) using LLM |
| perturb_data_lang.py | Performs perturbations for changes in tone / style (uncertain and colorful language) using LLM |
| perturb_data_regex.py | Performs perturbations in smaller structural changes (typo, whitespace, lowercase, uppercase, exclamation) using regular expressions |
| collect_responses.py | Samples LLMs for responses |
| binary_annot.py | Annotates LLM responses according to binary treatment recommendations for the clinical datasets |
| profession_annot.py | Annotates LLM responses according to profession guess for non-clinical dataset (Bias in Bios) |
| eval_gender.py | Samples LLM for gender inference using gender-removed contexts |

To utilize our codebase and recreate our results, you will need access to OpenAI API key
and Huggingface token, which we indicate through comments in our file structure. 
