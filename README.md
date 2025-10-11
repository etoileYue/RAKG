# RAKG: Document-level Retrieval Augmented Knowledge Graph Construction

<h5 align="center"> If you find our project useful, please give us a star ⭐ on GitHub for the latest updates.</h5>

## 💡 Overview

<table class="center">
    <tr>
        <td width=100% style="border: none"><img src="image\RAKG_flow.jpg" style="width:100%"></td>
    </tr>
    <tr>
        <td width="100%" style="border: none; text-align: center; word-wrap: break-word">
       RAKG is a knowledge graph construction framework that leverages large language models for automated knowledge graph generation. The framework processes documents through sentence segmentation and vectorization, extracts preliminary entities, and performs entity disambiguation and vectorization. The processed entities undergo Corpus Retrospective Retrieval to obtain relevant texts and Graph Structure Retrieval to get related KG. Subsequently, LLM is employed to integrate the retrieved information for constructing relation networks, which are merged for each entity. Finally, the newly built knowledge graph is combined with the original one.
      </td>
    </tr>
</table>

## 🔧 Installation

### Prerequisites
- Python 3.11
- Conda (recommended)

### Environment Setup

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

## 🏃 Quick Start

### Configuration

#### Model Provider Configuration
Edit `src/config.py` to configure your model provider settings:

##### Ollama Configuration
- For local Ollama: Set `base_url` to `http://localhost:11434/v1/`
- For server-based Ollama: Set `base_url` to `http://your_server_ip`

Default Ollama model configurations:
- Main model: Qwen2.5-72B, requires good instruction following
- Similarity check model: Qwen2-7B, using smaller model for faster processing
- embedding model: BGE-M3

##### OpenAI Configuration
- Set your OpenAI API key in `OPENAI_API_KEY`
- Configure model selection:
  - Main model: Qwen2.5-72B-Instruct
  - Similarity check model: Qwen2.5-14B-Instruct
  - Embedding model: BGE-M3

To switch between providers, set `USE_OPENAI = True` for OpenAI or `False` for Ollama.

### Usage Examples

#### Text Input
To process text input:
```bash
cd examples
python RAKG_example.py --input "your input text" --output result/kg.json --topic "your_topic" --is-text
```

#### Document Input
To process document input:
```bash
python RAKG_example.py --input data/MINE.json --output result/kg.json
```

#### Reproducing Paper Results
To reproduce the results from the paper:
```bash
cd src/construct
python RAKG.py
```

## 📊 Evaluation

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

### Ideal KG Evaluation
```bash
cd src/eval/ideal_kg_eval
python kg_eval.py
```

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

## ❤️ Acknowledgement

This repo benefits from:
- [KGGen](https://github.com/stair-lab/kg-gen)
- [Graphrag](https://github.com/microsoft/graphrag)
- [Qwen](https://github.com/QwenLM/Qwen)
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding)
- [RARE](https://github.com/Open-DataFlow/RARE)

Thanks for these wonderful works.

## 📞 Contact

For any questions or feedback, please:
- Open an issue in the GitHub repository
- Reach out to us at [2212855@mail.nankai.edu.cn](2212855@mail.nankai.edu.cn)

## Citation
If you find our paper and codes useful, please kindly cite us via:

```bibtex
@misc{zhang2025rakgdocumentlevelretrievalaugmentedknowledge,
      title={RAKG:Document-level Retrieval Augmented Knowledge Graph Construction}, 
      author={Hairong Zhang and Jiaheng Si and Guohang Yan and Boyuan Qi and Pinlong Cai and Song Mao and Ding Wang and Botian Shi},
      year={2025},
      eprint={2504.09823},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.09823}, 
}
```
