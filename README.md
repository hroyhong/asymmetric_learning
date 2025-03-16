# Asymmetric Learning

This repository contains code and resources for the asymmetric learning project, which investigates learning biases in AI models using a two-alternative forced choice (2AFC) slot machine paradigm.

## Project Overview

The project consists of three independent experiments:

1. **Task 1: Basic Partial Feedback** - Evaluates optimism bias by showing only the outcome of the chosen option.
2. **Task 2: Full Feedback** - Tests for confirmation bias by providing counterfactual information (outcomes of both chosen and unchosen options).
3. **Task 3: Agency Condition** - Examines the influence of agency on learning by comparing free-choice and forced-choice trials.

These experiments aim to determine whether models exhibit asymmetric learning patterns similar to humans, such as optimism bias and confirmation bias.

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)
- For DeepSeek integration: Volcengine API key (https://console.volcengine.com/)

### Installation
```bash
git clone https://github.com/hroyhong/asymmetric_learning.git
cd asymmetric_learning
pip install -r requirements.txt
```

### DeepSeek Integration Setup
This project supports integration with the DeepSeek-R1 model via Volcengine (火山引擎) for more realistic AI agent behavior in the experiments. To set up:

1. Run the setup script, which will guide you through the process:
```bash
python setup_deepseek.py
```

2. The script will:
   - Install required packages
   - Prompt for your Volcengine API key
   - Test the API connection
   - Optionally save your API key to your environment

If you already have your API key, you can provide it directly:
```bash
python setup_deepseek.py --api-key "your_api_key_here"
```

### Running Experiments
Without DeepSeek (using random choices):
```bash
python src/main.py --task=1  # For Task 1: Basic Partial Feedback
python src/main.py --task=2  # For Task 2: Full Feedback
python src/main.py --task=3  # For Task 3: Agency Condition
```

With DeepSeek (using AI decisions):
```bash
python src/main.py --task=1 --use-deepseek  # Task 1 with DeepSeek
python src/main.py --task=2 --use-deepseek  # Task 2 with DeepSeek
python src/main.py --task=3 --use-deepseek  # Task 3 with DeepSeek
```

You can also provide the API key directly:
```bash
python src/main.py --task=1 --use-deepseek --api-key "your_api_key_here"
```

## Project Structure
- `src/`: Source code for the experiments
  - `tasks/`: Implementation of the three experimental tasks
  - `models/`: Cognitive models (Rescorla-Wagner, RW±)
  - `analysis/`: Analysis scripts for experimental results
  - `api/`: API integrations for external models (DeepSeek, etc.)
- `data/`: Raw and processed experimental data
- `results/`: Analysis outputs and visualizations

## License

Details about the project license will be added here. 