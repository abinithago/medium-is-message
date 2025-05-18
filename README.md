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
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py           # Core type definitions
│   │   ├── perturbation.py    # Base perturbation class
│   │   ├── annotation.py      # Base annotation class
│   │   └── llm_sampling.py    # LLM interaction utilities
│   ├── perturbation/
│   │   ├── __init__.py
│   │   ├── gender.py          # Gender-based modifications
│   │   ├── language.py        # Language style alterations
│   │   └── structure.py       # Basic text structure changes
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── treatment.py       # Treatment recommendation annotation
│   │   └── profession.py      # Profession inference annotation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── gender_inference.py # Gender inference evaluation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── baseline_data/            # Original dataset files
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Module Overview

### Core Module
- `types.py`: Core data structures and type definitions
- `perturbation.py`: Abstract base class for text perturbations
- `annotation.py`: Abstract base class for response annotations
- `llm_sampling.py`: Utilities for interacting with LLMs

### Perturbation Module
- `gender.py`: Implementation of gender-based text modifications
- `language.py`: Implementation of language style alterations
- `structure.py`: Implementation of structural text changes

### Annotation Module
- `treatment.py`: Annotation of treatment recommendations
- `profession.py`: Annotation of profession inferences

### Evaluation Module
- `gender_inference.py`: Evaluation of gender inference from modified texts

### Utils Module
- `helpers.py`: Common utility functions used across modules

## Requirements

- Python 3.8+
- OpenAI API key
- Huggingface token
- See `requirements.txt` for full dependency list

