# The Medium is the Message: How Non-Clinical Information Shapes Clinical Decisions in LLMs

This repository contains the code for our research exploring how LLM clinical decision-making is impacted by non-clinical inputs.

## Project Overview

This project investigates how various non-clinical aspects of text input can influence the clinical decisions made by Large Language Models (LLMs). We explore different types of perturbations including gender changes, language style modifications, and structural alterations.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MIM-The-Medium-is-the-Message.git
cd MIM-The-Medium-is-the-Message
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_TOKEN=your_token_here
```

## Project Structure

```
.
├── code/
│   ├── perturb_data_llm.py      # Explicit gender changes (swapping and removal)
│   ├── perturb_data_lang.py     # Tone/style changes (uncertain and colorful language)
│   ├── perturb_data_regex.py    # Structural changes (typo, whitespace, case, etc.)
│   ├── collect_responses.py     # LLM response collection
│   ├── binary_annot.py         # Binary treatment recommendation annotation
│   ├── profession_annot.py     # Profession inference annotation
│   ├── eval_gender.py          # Gender inference from gender-removed contexts
│   └── utils.py               # Helper functions and file structure organization
├── baseline_data/             # Original dataset files
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation

```

## Usage

Each script in the `code/` directory serves a specific purpose in the research pipeline:

1. **Data Perturbation**:
   - `perturb_data_llm.py`: Gender-based modifications
   - `perturb_data_lang.py`: Language style alterations
   - `perturb_data_regex.py`: Basic text structure changes

2. **Analysis**:
   - `collect_responses.py`: Gather LLM responses
   - `binary_annot.py`: Analyze treatment recommendations
   - `profession_annot.py`: Analyze profession inferences
   - `eval_gender.py`: Evaluate gender detection

## Requirements

- Python 3.8+
- OpenAI API key
- Huggingface token
- See `requirements.txt` for full dependency list

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your-paper-citation,
  title={The Medium is the Message: How Non-Clinical Information Shapes Clinical Decisions in LLMs},
  author={Authors},
  journal={Journal},
  year={2023}
}
```

## License

[Add your license information here]

## Contact

[Add contact information here]
