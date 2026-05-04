# EngiAgent:  Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions

**EngiAgent** is a multi-agent Large Language Model (LLM) system designed for automated engineering problem solving. It combines specialized AI agents, mathematical optimization, and intelligent coordination to tackle complex engineering challenges through systematic problem analysis, mathematical modeling, solution verification, and quality assessment.

## 🎯 Key Features

- **Multi-Agent Architecture**: Five specialized agents working collaboratively
- **Intelligent Coordination**: Dynamic mode selection (fixed/free) for optimal problem solving
- **Comprehensive Evaluation**: Four-dimensional assessment framework
- **Mathematical Optimization**: Integrated Pyomo-based optimization modeling
- **Engineering Knowledge Integration**: HMML knowledge base retrieval
- **Automated Pipeline**: End-to-end processing with minimal human intervention

## 🏗️ System Architecture

### Core Components

1. **Enhanced Coordinator** (`enhanced_coordinator.py`): Multi-agent system orchestration
2. **Specialized Agents** (`specialized_agents.py`): Domain-specific problem solving agents
3. **Automated Pipeline** (`automated_engillm_pipeline.py`): Main execution framework
4. **Debug Framework** (`debug_framework.py`): Error handling and debugging system
5. **Evaluation System** (`EngiLLM_modeling_evaluator.py`): Quality assessment framework

### Specialized Agents

- **Problem Analyzer**: Extract key information and constraints
- **Modeling Agent**: Generate mathematical optimization models
- **Verification Agent**: Validate solution feasibility and correctness
- **Solution Agent**: Execute optimization solving
- **Simulation Agent**: Generate fallback solutions when needed

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager (recommended)
- Valid API keys for OpenAI GPT models and Google Gemini

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Mathematical Solvers

For macOS users:
```bash
bash install_solvers.sh
```

For manual installation:
```bash
# Create conda environment
conda create -n engillm python=3.10 -y
conda activate engillm

# Install solvers
conda install -c conda-forge glpk coin-or-cbc ipopt scip -y
pip install cylp
```

### Step 3: Configure API Keys

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
```

**Note**: API keys should be configured directly in the specialized_agents.py file or through environment variables. The system currently does not support .env file loading.

## 🚀 Quick Start

### Basic Usage

Run a single problem:
```bash
python automated_engillm_pipeline.py --problem P1 --model gemini --mode free
```

Run multiple problems:
```bash
python automated_engillm_pipeline.py --batch P1 P2 P3 --model gpt4o --mode free
```

Run all problems:
```bash
python automated_engillm_pipeline.py --all --model gemini --mode free
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--csv` | Input dataset file | `EngiLLM Dataset-Sheet1.xlsx` | `--csv "EngiLLM Dataset-Sheet1.csv"` |
| `--output` | Output directory | `automated_results` | `--output results` |
| `--mode` | Coordinator mode | `free` | `--mode fixed` |
| `--model` | LLM model | `gpt4o` | `--model gemini` |
| `--problem` | Single problem ID | - | `--problem P1` |
| `--index` | Problem by index | - | `--index 0` |
| `--batch` | Multiple problem IDs | - | `--batch P1 P5 P10` |
| `--all` | Run all problems | `False` | `--all` |
| `--list` | List available problems | `False` | `--list` |

### Supported Models

- `gemini`: Google Gemini (recommended for efficiency)
- `gpt4o`: OpenAI GPT-4o (recommended for quality)
- `gpt4`: OpenAI GPT-4
- `gpt5`: OpenAI GPT-5 (experimental)
- `gpt4.1nano`: OpenAI GPT-4.1 Nano (lightweight)
- `deepseek`: DeepSeek model

### Coordinator Modes

- **Free Mode** (`--mode free`): Dynamic agent selection based on problem characteristics
- **Fixed Mode** (`--mode fixed`): Predefined agent sequence for systematic processing

## 📊 Output Structure

```
engillm_outputs/
├── reports/           # Detailed problem solving reports
├── json_tracking/     # Process tracking and debugging info
├── evaluations/       # Quality assessment results
├── numerical_analysis/# Numerical solution analysis
├── visualizations/    # Quality curves and charts
└── summaries/         # Condensed result summaries

automated_results/     # Batch processing results
├── batch_results_*.json
└── individual problem folders
```

## 🧪 Example Usage Scenarios

### Research Experiment

```bash
# Large-scale evaluation with batch processing
nohup python automated_engillm_pipeline.py \
  --csv "EngiLLM Dataset-Sheet1.csv" \
  --model gemini \
  --mode free \
  --output automated_results \
  --batch P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 \
  > experiment_output.log 2>&1 &
```

### Development and Testing

```bash
# Test single problem with detailed output
python automated_engillm_pipeline.py --problem P1 --model gpt4o --mode free

# Compare modes on same problem
python automated_engillm_pipeline.py --problem P5 --model gemini --mode fixed
python automated_engillm_pipeline.py --problem P5 --model gemini --mode free
```

### Model Comparison

```bash
# Compare different models
python automated_engillm_pipeline.py --batch P1 P2 P3 --model gemini --mode free
python automated_engillm_pipeline.py --batch P1 P2 P3 --model gpt4o --mode free
python automated_engillm_pipeline.py --batch P1 P2 P3 --model deepseek --mode free
```

## 📈 Evaluation Framework

EngiLLM provides comprehensive evaluation across four dimensions:

1. **Information Extraction**: Ability to identify key problem elements
2. **Domain-Specific Reasoning**: Engineering knowledge application
3. **Multi-Objective Decision-Making**: Handling competing objectives
4. **Uncertainty Handling**: Managing ambiguous or incomplete information

Results include:
- Quantitative scores (0-10 scale)
- Detailed qualitative analysis
- Solution feasibility assessment
- Numerical accuracy validation

## 🛠️ Development

### Project Structure

```
EngiLLM/
├── automated_engillm_pipeline.py    # Main execution pipeline
├── enhanced_coordinator.py          # Multi-agent coordination
├── specialized_agents.py            # Domain-specific agents
├── base_agents.py                   # Agent framework
├── debug_framework.py               # Error handling system
├── json_tracking_manager.py         # Process tracking
├── EngiLLM_modeling_evaluator.py    # Evaluation framework
├── EngiLLM_numerical_analyzer.py    # Solution analysis
├── hmml_retriever.py                # Knowledge retrieval
├── install_solvers.sh               # Solver installation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── EngiLLM Dataset-Sheet1.csv       # Problem dataset
└── Engi-HMML-v2.md                 # Knowledge base
```

### Adding New Problems

1. Add problem to `EngiLLM Dataset-Sheet1.csv`
2. Include evaluation criteria in the dataset
3. Test with single problem run before batch processing

### Extending Agents

1. Inherit from `BaseAgent` class
2. Implement required methods
3. Register with coordinator
4. Add to specialized agents module

## 📋 Troubleshooting

### Common Issues

1. **Solver not found**: Ensure mathematical solvers are properly installed via conda
2. **API key errors**: Verify API keys are correctly set as environment variables
3. **Memory issues**: For large batch processing, consider splitting into smaller batches
4. **Timeout errors**: Increase timeout settings for complex problems

### Performance Tips

- Use `gemini` model for normal processing
- Use `gpt4o` model for higher quality results
- Install all packages for model optimization and fully prepare all kinds of solvers
- Monitor API usage to avoid rate limits

