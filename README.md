# PatternSmith — Natural Language to Regex with a Transformer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-purple" alt="Transformer">
  <img src="https://img.shields.io/badge/Tokenizer-GPT--2%20BPE-green" alt="GPT-2 BPE">
  <img src="https://img.shields.io/badge/Training-From%20Scratch-success" alt="From Scratch">
  <img src="https://img.shields.io/badge/Tests-133%20Passed-brightgreen" alt="Tests">
</p>

**PatternSmith** is a natural-language-to-regular-expression generation project built using a custom decoder-only GPT-style Transformer implemented from scratch in PyTorch.

It translates English descriptions of text patterns into regular expressions using supervised next-token prediction and autoregressive generation.

The objective of this project was to understand and implement the internal components of a GPT-style language model, including embeddings, causal self-attention, Transformer blocks, optimization, training, checkpointing, and inference.

---

## Project overview

**Input:**

```text
Match exactly five digits
```

**Expected output:**

```regex
^\d{5}$
```

The model learns to generate regex expressions from natural-language descriptions using paired training examples.

### Key highlights

- Custom decoder-only GPT-style Transformer implemented in PyTorch.
- 7.24 million trainable parameters.
- GPT-2 byte-level BPE tokenizer integration using `tiktoken`.
- Four Transformer blocks with multi-head causal self-attention.
- Custom embeddings, attention, MLP, LayerNorm, and residual connections.
- Complete supervised training pipeline.
- AdamW optimizer with weight decay.
- Linear warmup and cosine learning-rate scheduling.
- Gradient clipping and early stopping.
- Autoregressive inference.
- Evaluation using exact-match accuracy and semantic equivalence.
- 133 automated tests passed during development.

## Technology stack

| Category | Technologies |
|---|---|
| Programming language | Python 3.10 |
| Deep learning | PyTorch |
| Tokenization | tiktoken (GPT-2 BPE) |
| Numerical computing | NumPy |
| Data processing | Pandas |
| Training hardware | Google Colab NVIDIA Tesla T4 |
| Optimization | AdamW |
| Learning-rate scheduler | Linear warmup + cosine decay |
| Evaluation | regexbench |
| Testing | Pytest |
| Code formatting | Black |
| Version control | Git and GitHub |

## Model architecture

PatternSmith uses a decoder-only Transformer architecture inspired by GPT-2.

```text
       Natural Language Description
                    |
                    v
             GPT-2 BPE Tokenizer
                    |
                    v
             Token Embeddings
                    +
           Positional Embeddings
                    |
                    v
          Transformer Block 1
                    |
                    v
          Transformer Block 2
                    |
                    v
          Transformer Block 3
                    |
                    v
          Transformer Block 4
                    |
                    v
              Final LayerNorm
                    |
                    v
                LM Head
                    |
                    v
           Autoregressive Decoding
                    |
                    v
             Generated Regex
```

### Transformer configuration

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 128 |
| Embedding dimension | 128 |
| Number of Transformer blocks | 4 |
| Number of attention heads | 4 |
| Dimension per attention head | 32 |
| Feed-forward dimension | 512 |
| Activation function | GELU |
| Normalization | LayerNorm |
| Architecture | Pre-LayerNorm |
| Dropout | 0.1 |
| Output projection | Weight-tied LM head |
| Total trainable parameters | **7,242,624** |

## Detailed parameter-count breakdown

The model contains **7,242,624 trainable parameters**.

The following breakdown accounts for weight matrices, bias vectors, LayerNorm scale and bias parameters, and tied embedding weights.

### 1. Embedding parameters

PatternSmith uses token embeddings and learned positional embeddings.

**Token embedding**

\[
V \times D = 50,257 \times 128
\]

\[
= 6,432,896
\]

**Positional embedding**

\[
C \times D = 128 \times 128
\]

\[
= 16,384
\]

| Component | Calculation | Parameters |
|---|---|---:|
| Token embedding | 50,257 × 128 | 6,432,896 |
| Positional embedding | 128 × 128 | 16,384 |
| **Total** | | **6,449,280** |

### 2. Multi-head self-attention parameters

Each Transformer block contains four learned linear projections:

- Query (Q)
- Key (K)
- Value (V)
- Output projection

For each projection:

\[
W \in \mathbb{R}^{128 \times 128}
\]

\[
b \in \mathbb{R}^{128}
\]

Parameters per projection:

\[
(128 \times 128) + 128 = 16,512
\]

| Projection | Weight parameters | Bias parameters | Total |
|---|---:|---:|---:|
| Query | 16,384 | 128 | 16,512 |
| Key | 16,384 | 128 | 16,512 |
| Value | 16,384 | 128 | 16,512 |
| Output | 16,384 | 128 | 16,512 |
| **Total** | **65,536** | **512** | **66,048** |

The four attention heads operate on the projected 128-dimensional representations. They do not require four additional independent sets of projection matrices.

### 3. Feed-forward network parameters

Each Transformer block has a two-layer MLP:

```text
Input: 128
   |
   v
Linear(128, 512)
   |
   v
GELU
   |
   v
Linear(512, 128)
   |
   v
Output: 128
```

**First linear layer**

\[
(128 \times 512) + 512 = 66,048
\]

**Second linear layer**

\[
(512 \times 128) + 128 = 65,664
\]

| Layer | Weight parameters | Bias parameters | Total |
|---|---:|---:|---:|
| Linear 1 | 65,536 | 512 | 66,048 |
| Linear 2 | 65,536 | 128 | 65,664 |
| **Total** | **131,072** | **640** | **131,712** |

### 4. LayerNorm parameters

Each Transformer block contains two LayerNorm layers.

Each LayerNorm has:

- Learnable scale parameter (gamma): 128
- Learnable bias parameter (beta): 128

Therefore:

\[
2 \times (128 + 128) = 512
\]

| Component | Parameters |
|---|---:|
| LayerNorm 1 | 256 |
| LayerNorm 2 | 256 |
| **Total per block** | **512** |

### 5. Total parameters per Transformer block

| Subcomponent | Weights | Biases / additional parameters | Total |
|---|---:|---:|---:|
| Multi-head attention | 65,536 | 512 | 66,048 |
| Feed-forward network | 131,072 | 640 | 131,712 |
| Two LayerNorm layers | 256 | 256 | 512 |
| **One Transformer block** | **196,864** | **1,408** | **198,272** |

For four Transformer blocks:

\[
4 \times 198,272 = 793,088
\]

### 6. Final LayerNorm

The model also uses a final LayerNorm after the Transformer blocks.

\[
128 + 128 = 256
\]

### 7. Final model parameter summary

| Component | Parameters |
|---|---:|
| Token embeddings | 6,432,896 |
| Positional embeddings | 16,384 |
| Transformer block 1 | 198,272 |
| Transformer block 2 | 198,272 |
| Transformer block 3 | 198,272 |
| Transformer block 4 | 198,272 |
| Final LayerNorm | 256 |
| LM head | 0 additional |
| **Total** | **7,242,624** |

**Why does the LM head have zero additional parameters?**

PatternSmith uses weight tying between the token embedding matrix and the output language-model head.

Instead of maintaining a separate output projection matrix of shape \(50,257 \times 128\), the model reuses the token embedding weights.

This reduces the number of independent parameters and follows a commonly used language-model design.

## Dataset and preprocessing

PatternSmith uses the **NL-RX-Turk** dataset from the [deep-regex repository](https://github.com/nicholaslocascio/deep-regex).

The original dataset contains 10,000 natural-language descriptions paired with regular expressions.

### Dataset preprocessing

The preprocessing pipeline includes:

1. Dataset integrity and alignment checks.
2. Duplicate and conflicting-description analysis.
3. Removal of descriptions associated with multiple distinct regex targets.
4. Removal of exact duplicate pairs.
5. Deterministic train, validation, and test splitting.

After preprocessing:

| Split | Examples | Percentage |
|---|---:|---:|
| Training | 6,245 | 65% |
| Validation | 961 | 10% |
| Test | 2,402 | 25% |
| **Total** | **9,608** | **100%** |

Random seed: `42`.

### Tokenization

PatternSmith integrates the GPT-2 byte-level BPE tokenizer through `tiktoken`.

The tokenizer is reused, but the Transformer itself is initialized with random weights and trained specifically for this task.

Prompt format:

```text
Description: Match exactly five digits
Regex:
```

The target regex tokens are appended after the prompt, followed by the GPT-2 end-of-text token.

Context length was set to 128 based on the tokenization audit. This accommodates all examples in the cleaned dataset.

## Training methodology

PatternSmith uses supervised autoregressive next-token prediction with teacher forcing.

The model receives the prompt and target sequence during training, but the loss is calculated only over target tokens and the end-of-sequence token.

### Training configuration

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.01 |
| Learning-rate scheduler | Linear warmup + cosine decay |
| Warmup ratio | 5% |
| Minimum learning-rate ratio | 10% |
| Batch size | 16 |
| Maximum epochs | 40 |
| Gradient clipping | 1.0 |
| Early-stopping patience | 5 |
| Early-stopping minimum delta | 0.001 |
| Random seed | 42 |
| Training device | NVIDIA Tesla T4 |

### Training results

The best validation checkpoint was selected at epoch 16.

| Metric | Result |
|---|---:|
| Best epoch | 16 |
| Training loss | 0.1291 |
| Validation loss | **0.2044** |
| Validation perplexity | 1.2268 |

Training stopped through early stopping after validation performance stopped improving.

## Evaluation results

The model was evaluated on the complete held-out test set containing 2,402 examples.

### Exact-match evaluation

Exact match checks whether the predicted regex string is identical to the reference regex string.

| Metric | Result |
|---|---:|
| Test examples | 2,402 |
| Exact matches | 892 |
| Incorrect exact matches | 1,510 |
| Exact-match accuracy | **37.14%** |

### Semantic-equivalence evaluation

Two different regex strings can represent the same language.

For example, syntactically different expressions may match exactly the same set of strings. Exact-match evaluation alone cannot identify these cases.

PatternSmith therefore uses `regexbench` for semantic equivalence evaluation.

Evaluation configuration:

- Dialect: BRICS
- Semantics: FULLMATCH

| Verdict | Examples |
|---|---:|
| Equivalent | 1,334 |
| Different | 994 |
| Unsupported | 74 |
| Undecidable | 0 |
| **Total** | **2,402** |

\[
\text{Semantic Equivalence Rate}
=
\frac{1334}{2402}\times100
=
55.54\%
\]

**Final evaluation metrics**

| Metric | Result |
|---|---:|
| Exact-match accuracy | **37.14%** |
| Semantic-equivalence rate | **55.54%** |

The semantic-equivalence rate is higher because different regex expressions can represent equivalent languages.

Unsupported cases are not counted as equivalent. An unsupported result means the evaluator could not complete the comparison; it does not automatically establish that the prediction is incorrect.

### Preliminary syntax diagnostic

A Python `re.compile()` diagnostic succeeded for 2,329 of 2,402 predictions.

\[
\text{Python syntax pass rate} = 96.96\%
\]

This is a preliminary Python-regex syntax check, not a complete BRICS grammar validation. Python's regex dialect does not implement all BRICS operators and semantics.

## Pretrained Checkpoint

The trained PatternSmith model checkpoint is hosted on Hugging Face.

**[Download PatternSmith Checkpoint](https://huggingface.co/Ramu1904/PatternSmith)**

### Download the checkpoint

Install Hugging Face Hub:

```bash
python -m pip install -U huggingface_hub

hf download Ramu1904/PatternSmith best.pt --local-dir checkpoints
```

## Inference

PatternSmith generates regex expressions using greedy autoregressive decoding.

At each generation step:

1. The input description is converted into GPT-2 BPE tokens.
2. The model processes the current token sequence.
3. The logits at the final sequence position are extracted.
4. The highest-probability token is selected.
5. The selected token is appended to the sequence.
6. Generation stops when the end-of-text token is produced or the generation limit is reached.

Example:

```powershell
python -m patternsmith.inference "Match exactly five digits"
```

Choose a device:

```powershell
python -m patternsmith.inference "Match exactly five digits" --device cpu
```

View available options:

```powershell
python -m patternsmith.inference --help
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ramu1904/PatternSmith.git
cd PatternSmith
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Alternatively, install dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Training

Training is designed to run in a GPU-enabled environment such as Google Colab.

```bash
python scripts/train.py --epochs 40 --batch-size 16
```

View the available training arguments:

```bash
python scripts/train.py --help
```

## Project structure

```text
PatternSmith/
│
├── data/
│   ├── raw/NL-RX-Turk/
│   ├── processed/NL-RX-Turk/
│   └── splits/NL-RX-Turk/
│
├── notebooks/
│   └── PatternSmith_modeltrain.ipynb
│
├── outputs/
│   └── inference/
│       ├── semantic_evaluation.csv
│       └── test_predictions.csv
│
├── scripts/
│   ├── audit_dataset.py
│   ├── audit_tokenization.py
│   ├── clean_dataset.py
│   ├── create_splits.py
│   ├── train.py
│   ├── evaluate_inference.py
│   └── evaluate_semantics.py
│
├── src/
│   └── patternsmith/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── collate.py
│       │   └── dataset.py
│       │
│       ├── model/
│       │   ├── attention.py
│       │   ├── embeddings.py
│       │   ├── gpt_model.py
│       │   ├── mlp.py
│       │   ├── multi_head_attention.py
│       │   └── transformer_block.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── checkpoint.py
│       │   ├── early_stopping.py
│       │   ├── engine.py
│       │   ├── gradient_clipping.py
│       │   ├── optimizer.py
│       │   └── scheduler.py
│       │
│       ├── __init__.py
│       └── inference.py
│
├── tests/
│   ├── data/
│   ├── model/
│   └── training/
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Testing

The project includes automated tests covering:

- Dataset loading and tokenization.
- Dynamic padding and masking.
- Embedding layers.
- Causal attention.
- Multi-head attention.
- Feed-forward networks.
- Transformer blocks.
- GPT model forward pass.
- Optimizer and learning-rate scheduler.
- Gradient clipping.
- Training and validation.
- Checkpoint saving and loading.
- Inference.

The complete test suite passed **133 tests** during project development.

Run tests:

```bash
pytest -q
```

Check formatting:

```bash
black --check src tests scripts/audit_dataset.py scripts/audit_tokenization.py scripts/clean_dataset.py scripts/create_splits.py scripts/train.py scripts/evaluate_inference.py scripts/evaluate_semantics.py
```

## Limitations

- The model was trained on a relatively small dataset.
- It is not intended to generate reliable regex expressions for every possible real-world description.
- Syntactically valid predictions can still be semantically incorrect.
- Different regex strings may represent equivalent languages, making exact match an incomplete measure.
- The semantic evaluator can return unsupported results for expressions it cannot process.
- Evaluation results on NL-RX-Turk do not establish generalization to arbitrary regex-generation tasks.

PatternSmith is an educational and portfolio project demonstrating the implementation, training, inference, and evaluation of a small GPT-style Transformer. It is not a production-ready regex-generation system.

## Key learning outcomes

By building PatternSmith, I gained practical experience with:

- GPT-style decoder-only Transformer architecture.
- Token and positional embeddings.
- Scaled dot-product attention.
- Causal masking and multi-head self-attention.
- Residual connections and Pre-LayerNorm.
- Feed-forward networks and GELU activation.
- Weight tying.
- Teacher-forced supervised language-model training.
- Cross-entropy loss and masked labels.
- AdamW, weight decay, and cosine learning-rate scheduling.
- Gradient clipping and early stopping.
- Checkpoint management and autoregressive inference.
- Exact-match and semantic-equivalence evaluation.

## References

- [NL-RX-Turk / Deep Regex dataset](https://github.com/nicholaslocascio/deep-regex)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [tiktoken](https://github.com/openai/tiktoken)
- [regexbench](https://pypi.org/project/regexbench/)

---

**Built with Python and PyTorch — implementing a GPT-style Transformer from scratch.**