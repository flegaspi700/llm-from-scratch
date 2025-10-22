# LLM from Scratch

A GPT-style transformer language model implementation from scratch in PyTorch, trained on Shakespeare's works for character-level text generation.

## 🎯 Project Overview

This project builds a small-scale transformer model (~10M parameters) to understand LLM fundamentals by implementing every component from the ground up. The model uses word-level tokenization and is trained on Shakespeare's complete works to generate coherent Early Modern English text.

## 📊 Current Status

### ✅ Completed: Phase 1 - Data Preparation

- [x] Project structure setup
- [x] Virtual environment configuration
- [x] Word-level tokenizer with contraction handling
- [x] 90/10 train/validation split
- [x] Binary data format for efficient loading
- [x] Data integrity verification

### 🚧 Next: Phase 2 - Model Architecture

- [ ] Positional embeddings
- [ ] Multi-head self-attention
- [ ] Feed-forward networks
- [ ] Transformer blocks
- [ ] Complete GPT model

## 🏗️ Architecture

### Model Configuration
- **Vocabulary**: ~12,384 word-level tokens
- **Context Length**: 256 tokens
- **Layers**: 6 transformer blocks
- **Attention Heads**: 6
- **Embedding Dimension**: 384
- **Parameters**: ~10M (trainable on CPU)

### Data Pipeline
```
Shakespeare Text → Word Tokenizer → Train/Val Split → Binary Files
  (1.1M chars)      (~12K vocab)      (90/10 split)    (train.bin, val.bin)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/flegaspi700/llm-from-scratch.git
cd llm-from-scratch

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Prepare the Shakespeare dataset
python data/prepare_data.py

# Verify data integrity
python data/prepare_data.py --verify-only
```

**Expected Output:**
- `data/processed/vocab.pkl` - Tokenizer vocabulary (~152 KB)
- `data/processed/train.bin` - Training data (~887 KB, 226K tokens)
- `data/processed/val.bin` - Validation data (~98 KB, 25K tokens)

### Testing the Tokenizer

```bash
# Run tokenizer tests
python src/utils/tokenizer.py
```

## 📁 Project Structure

```
llm-from-scratch/
├── .github/
│   ├── copilot-instructions.md    # AI agent guidance
│   └── llm-project-spec.md        # Complete technical spec
├── config/
│   └── model_config.yaml          # Hyperparameters (coming soon)
├── data/
│   ├── raw/
│   │   └── shakespeare.txt        # Raw text data
│   ├── processed/
│   │   ├── train.bin              # Encoded training data
│   │   ├── val.bin                # Encoded validation data
│   │   └── vocab.pkl              # Tokenizer vocabulary
│   └── prepare_data.py            # Data preparation pipeline
├── src/
│   ├── model/                     # Model components (coming soon)
│   ├── training/                  # Training infrastructure (coming soon)
│   ├── utils/
│   │   └── tokenizer.py           # Word-level tokenizer
│   └── inference/                 # Text generation (coming soon)
├── scripts/                       # Training/eval scripts (coming soon)
├── tests/                         # Unit tests (coming soon)
├── notebooks/                     # Jupyter notebooks (coming soon)
├── checkpoints/                   # Saved models
├── logs/                          # TensorBoard logs
├── requirements.txt
└── README.md
```

## 🔧 Technical Highlights

### Word-Level Tokenization

Unlike the original spec's character-level approach, this implementation uses **word-level tokenization** with special handling for:

- **Contractions**: Keeps Shakespeare's contractions intact (`'tis`, `thou'rt`, `o'er`)
- **Punctuation**: Preserves as separate tokens for better linguistic structure
- **Unknown tokens**: Maps out-of-vocabulary words to `<UNK>`
- **Padding**: Supports `<PAD>` token for batch processing

**Vocabulary Statistics:**
- Total unique tokens: **12,384**
- Training corpus: **252,144 tokens**
- Special tokens: `<UNK>`, `<PAD>`

### Data Format

Binary files use `numpy.uint32` format for efficient loading:

```python
# Loading is fast (~10ms)
train_data = np.fromfile("data/processed/train.bin", dtype=np.uint32)
val_data = np.fromfile("data/processed/val.bin", dtype=np.uint32)
```

## 📚 Learning Resources

This project follows best practices from:

- **"Attention Is All You Need"** (Vaswani et al., 2017)
- **Andrej Karpathy's nanoGPT** - Architecture inspiration
- **"Let's build GPT"** - Training methodology

See [`.github/llm-project-spec.md`](.github/llm-project-spec.md) for the complete technical specification.

## 🎓 Development Philosophy

### Key Principles

1. **Modular Design**: Each component is independently testable
2. **Phase-Based Development**: Complete one milestone before moving to the next
3. **Educational Focus**: Code clarity over performance optimization
4. **Reproducibility**: Fixed random seeds and deterministic data splits

### Why Word-Level vs Character-Level?

While the original spec suggested character-level tokenization, word-level was chosen for:

- **Linguistic coherence**: Better preservation of Shakespeare's vocabulary
- **Reduced sequence length**: ~4x shorter sequences than character-level
- **Semantic units**: Words carry more meaning than individual characters
- **Authenticity**: Model learns Shakespeare's actual vocabulary (including `'tis`, `doth`, `thou'rt`)

## 📈 Performance Targets

### Training Goals
- **Training Loss**: < 1.5
- **Validation Loss**: < 1.8
- **Perplexity**: < 6.0
- **Training Time**: < 1 hour (GPU) / < 3 hours (CPU)

### Generation Quality
- Coherent Shakespeare-style text
- Proper use of Early Modern English contractions
- Maintains iambic pentameter patterns (stretch goal)

## 🛠️ Development Roadmap

### Phase 1: Data Preparation ✅ (Completed)
- Word-level tokenizer implementation
- Train/val data split (90/10)
- Binary format optimization

### Phase 2: Model Architecture 🚧 (Next)
- Positional embeddings (sinusoidal or learned)
- Multi-head self-attention with causal masking
- Feed-forward networks with GELU activation
- Transformer blocks with residual connections
- Complete GPT model assembly

### Phase 3: Training Infrastructure
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping
- TensorBoard logging
- Checkpointing system

### Phase 4: Training & Evaluation
- Main training loop
- Validation monitoring
- Perplexity calculation
- Loss curve analysis

### Phase 5: Text Generation
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling

### Phase 6: Experimentation
- Hyperparameter tuning
- Model size variations
- Different sampling strategies
- Performance analysis

## 🤝 Contributing

This is a personal learning project, but feedback and suggestions are welcome! Please see [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for development guidelines.

## 📝 License

This project is for educational purposes. Shakespeare's works are in the public domain.

## 🙏 Acknowledgments

- **Andrej Karpathy** - For nanoGPT and educational content
- **OpenAI** - For transformer architecture innovations
- **Shakespeare** - For the timeless training data

---

**Current Progress**: Phase 1 Complete | **Next Milestone**: Implementing positional embeddings and attention mechanisms

*Building transformers from scratch to understand how LLMs really work* 🚀