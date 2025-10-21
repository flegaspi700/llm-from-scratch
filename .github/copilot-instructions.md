# LLM from Scratch - AI Coding Agent Instructions

## Project Overview
This is a GPT-style transformer implementation from scratch in PyTorch, focused on character-level language modeling with Shakespeare text. The project follows a **modular, component-based architecture** with clear separation between model layers, training infrastructure, and inference.

## Architecture & Code Structure

### Core Model Components (`src/model/`)
- **`attention.py`**: Multi-head self-attention with causal masking for autoregressive generation
- **`embedding.py`**: Token + positional embeddings (sinusoidal or learned)  
- **`feedforward.py`**: Two-layer MLP with GELU activation and 4x expansion
- **`transformer_block.py`**: Combines attention + FFN with layer norm and residuals
- **`gpt.py`**: Complete model stack (N transformer blocks + output projection)

### Data Pipeline (`src/utils/`, `data/`)
- **Character-level tokenization**: Build vocab from Shakespeare text, save as binary files
- **`data_loader.py`**: Random batch sampling with configurable context windows
- **Data split**: 90% train, 10% validation, stored as `train.bin`/`val.bin`

### Training Infrastructure (`src/training/`)
- **AdamW optimizer** with weight decay (0.1) and different LR groups
- **Cosine LR scheduler** with 2000-step warmup and min LR
- **Gradient clipping** (max_norm=1.0) to prevent exploding gradients
- **Checkpointing**: Save best model + optimizer state every N iterations

## Key Implementation Patterns

### Causal Attention Masking
```python
mask = torch.tril(torch.ones(block_size, block_size))
attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
```

### Residual Connections Architecture
Follow: `x → LayerNorm → Attention → Add → LayerNorm → FFN → Add`

### Model Configuration (see `llm-project-spec.md`)
- **Small scale**: ~10M parameters, 256 context length, 6 layers/heads
- **Target metrics**: Train loss < 1.5, Val loss < 1.8, Perplexity < 6.0

## Development Workflow

### Phase-Based Implementation
1. **Phase 1**: Data preparation and character tokenizer
2. **Phase 2**: Model components (attention → FFN → blocks → GPT)
3. **Phase 3**: Training loop with monitoring
4. **Phase 4**: Text generation and sampling strategies

### Testing Strategy
- **Unit tests** for each component (`tests/test_*.py`)
- **Shape verification** at each layer
- **Gradient flow** validation
- **End-to-end integration** tests

### Key Scripts
- **`scripts/train.py`**: Main training entry point with config loading
- **`scripts/generate.py`**: CLI for text generation with sampling options
- **`scripts/evaluate.py`**: Model evaluation and perplexity calculation

## Project-Specific Conventions

### File Organization
- Keep model components as **separate, testable modules**
- Use **YAML config files** for hyperparameters (`config/model_config.yaml`)
- Store processed data as **binary files** for fast loading
- Separate **training utilities** from model architecture

### Error Handling & Debugging
- **Monitor training loss curves** - should decrease steadily
- **Check tensor shapes** at each forward pass step  
- **Validate tokenizer** encode/decode consistency
- **Save checkpoints frequently** during initial training phases

### Performance Considerations
- **Mixed precision training** available via `torch.cuda.amp`
- **Gradient accumulation** for larger effective batch sizes
- **CPU fallback** supported (3-hour training time vs 30min GPU)

## Getting Started Commands
```bash
# Setup environment
python -m venv venv && venv\Scripts\activate
pip install torch numpy matplotlib tensorboard tqdm

# Create project structure (follows spec exactly)
mkdir -p config data/{raw,processed} src/{model,training,utils,inference}
mkdir -p scripts tests notebooks checkpoints logs
```

Refer to `llm-project-spec.md` for complete implementation details, hyperparameters, and phase-by-phase development plan.