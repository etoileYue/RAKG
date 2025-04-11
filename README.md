# RAKG

RAKG is a knowledge graph construction framework that leverages large language models for automated knowledge graph generation.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Environment Setup](#environment-setup)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)

## Quick Start

To get started with RAKG, follow the simple steps below:

1. Clone the repository
2. Set up the environment
3. Configure the necessary settings
4. Run the examples

## Configuration

### Ollama Configuration
Edit `src/config.py` to configure your Ollama settings:

- For local Ollama: Set `base_url` to `http://localhost:11434/v1/`
- For server-based Ollama: Set `base_url` to `http://your_server_ip`

Default model configurations:
- Main model: qwen2.5:72b, requires good instruction following
- Similarity check model: qwen2:7b, using smaller model for faster processing
- embedding model: bge

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/RAKG/RAKG.git
cd RAKG
```

2. Create and activate a conda environment:
```bash
conda create -n RAKG python=3.11
conda activate RAKG
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Text Input
To process text input:
```bash
cd examples
python RAKG_example.py --input "your input text" --output result/kg.json --topic "your_topic" --is-text
```

### Document Input
To process document input:
```bash
python KGC_MINE_RAKG_withRA_withdes.py --input data/MINE.json --output result/kg.json
```

### Reproducing Paper Results
To reproduce the results from the paper:
```bash
cd src/construct
python RAKG.py
```


## Evaluation

### LLM Evaluation
```bash
cd src/eval/llm_eval
```
For evaluation purposes, we recommend using the DeepEval platform. Please refer to the [DeepEval documentation](https://github.com/confident-ai/deepeval) for setup and usage instructions.

### MINE Evaluation
```bash
cd src/eval/MINE_eval
python evaluate_MINE_RAKG.py
```

### ideal_kg Evaluation
```bash
cd src/eval/ideal_kg_eval
python kg_eval.py
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

## Contact

For questions and support, please open an issue in the GitHub repository.