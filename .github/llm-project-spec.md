# Building an LLM from Scratch: Technical Specification

## Project Overview
**Goal**: Build a small-scale GPT-style transformer model from scratch to understand LLM fundamentals.

**Scope**: Character-level language model trained on Shakespeare text, capable of generating coherent text.

**Timeline**: 2-4 weeks (assuming 5-10 hours/week)

---

## Technical Stack

### Core Dependencies
```
python >= 3.9
torch >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
tensorboard >= 2.13.0
tqdm >= 4.65.0
```

### Development Tools
- VS Code with Python extension
- Jupyter notebooks (optional, for experimentation)
- Git for version control

---

## Project Structure

```
llm-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── model_config.yaml          # Hyperparameters
├── data/
│   ├── raw/
│   │   └── shakespeare.txt        # Raw text data
│   ├── processed/
│   │   ├── train.bin              # Processed training data
│   │   └── val.bin                # Processed validation data
│   └── prepare_data.py            # Data preparation script
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py           # Multi-head attention
│   │   ├── embedding.py           # Token + positional embeddings
│   │   ├── feedforward.py         # FFN layer
│   │   ├── transformer_block.py   # Single transformer block
│   │   └── gpt.py                 # Complete GPT model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── optimizer.py           # Optimizer setup
│   │   └── scheduler.py           # Learning rate scheduling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # Character-level tokenizer
│   │   ├── data_loader.py         # Batch creation
│   │   └── metrics.py             # Loss, perplexity calculation
│   └── inference/
│       ├── __init__.py
│       └── generator.py           # Text generation logic
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_generation_examples.ipynb
├── scripts/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation script
│   └── generate.py                # Text generation script
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   └── test_tokenizer.py
├── checkpoints/                   # Saved model weights
└── logs/                          # TensorBoard logs
```

---

## Implementation Plan

### Phase 1: Setup & Data Preparation (Week 1)

#### Milestone 1.1: Project Setup
- [ ] Create VS Code project structure
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Initialize git repository
- [ ] Create `.gitignore` (exclude checkpoints/, logs/, data/processed/)

#### Milestone 1.2: Data Pipeline
**Files to create**: `data/prepare_data.py`, `src/utils/tokenizer.py`

**Implementation steps**:
1. Download Shakespeare dataset
2. Implement character-level tokenizer
   - Build vocabulary (char → id mapping)
   - Encode/decode functions
   - Save vocabulary to file
3. Split data (90% train, 10% validation)
4. Save processed data as binary files

**Validation**:
- Verify tokenizer can encode and decode correctly
- Check train/val split sizes
- Print sample encoded sequences

---

### Phase 2: Model Architecture (Week 1-2)

#### Milestone 2.1: Building Blocks
**Files to create**: `src/model/attention.py`, `src/model/embedding.py`, `src/model/feedforward.py`

**Implementation steps**:

1. **Positional Encoding** (`embedding.py`)
   ```python
   class PositionalEncoding:
       - Sinusoidal or learned embeddings
       - Input: sequence length
       - Output: position embeddings
   ```

2. **Multi-Head Self-Attention** (`attention.py`)
   ```python
   class MultiHeadAttention:
       - Query, Key, Value projections
       - Scaled dot-product attention
       - Causal masking (for autoregressive generation)
       - Multiple attention heads
       - Output projection
   ```

3. **Feed-Forward Network** (`feedforward.py`)
   ```python
   class FeedForward:
       - Two linear layers with GELU activation
       - Expansion factor (typically 4x)
       - Dropout
   ```

**Testing**:
- Unit tests for each component
- Check tensor shapes at each step
- Verify gradient flow

#### Milestone 2.2: Transformer Block
**File to create**: `src/model/transformer_block.py`

**Implementation steps**:
1. Combine attention + FFN + layer norm + residual connections
2. Follow architecture: `x → LayerNorm → Attention → Add → LayerNorm → FFN → Add`

#### Milestone 2.3: Complete GPT Model
**File to create**: `src/model/gpt.py`

**Implementation steps**:
1. Token embeddings layer
2. Positional encoding
3. Stack N transformer blocks
4. Final layer norm
5. Output projection (to vocabulary size)
6. Implement forward pass

**Model hyperparameters** (configurable):
```yaml
vocab_size: ~65 (for Shakespeare character set)
block_size: 256        # Context length
n_layer: 6             # Number of transformer blocks
n_head: 6              # Number of attention heads
n_embd: 384            # Embedding dimension
dropout: 0.2           # Dropout rate
```

**Validation**:
- Forward pass with random input
- Count parameters (~10M expected)
- Verify output shape matches vocab_size

---

### Phase 3: Training Infrastructure (Week 2)

#### Milestone 3.1: Data Loading
**File to create**: `src/utils/data_loader.py`

**Implementation steps**:
1. Create function to load binary data
2. Implement batch generator
3. Random sampling with context window
4. Device handling (CPU/GPU)

#### Milestone 3.2: Training Loop
**Files to create**: `src/training/trainer.py`, `src/training/optimizer.py`, `src/training/scheduler.py`

**Implementation steps**:

1. **Optimizer Setup** (`optimizer.py`)
   - AdamW optimizer
   - Weight decay (0.1)
   - Different learning rates for different parameter groups

2. **Learning Rate Scheduler** (`scheduler.py`)
   - Warmup steps (2000 iterations)
   - Cosine decay schedule
   - Min learning rate

3. **Training Loop** (`trainer.py`)
   ```python
   class Trainer:
       - Forward pass
       - Loss calculation (cross-entropy)
       - Backward pass
       - Gradient clipping (max_norm=1.0)
       - Optimizer step
       - Learning rate scheduling
       - Logging (loss, learning rate)
       - Checkpointing
       - Validation evaluation
   ```

**Training configuration**:
```yaml
batch_size: 64
learning_rate: 3e-4
max_iters: 5000
eval_interval: 500
eval_iters: 200
```

#### Milestone 3.3: Monitoring & Logging
**File to create**: `src/utils/metrics.py`

**Implementation steps**:
1. Calculate and log training loss
2. Calculate validation loss & perplexity
3. TensorBoard integration
4. Progress bar with tqdm
5. Save best model checkpoint

---

### Phase 4: Training Execution (Week 2-3)

#### Milestone 4.1: Initial Training
**File to create**: `scripts/train.py`

**Steps**:
1. Load configuration
2. Initialize model, optimizer, scheduler
3. Load data
4. Run training loop
5. Save checkpoints

**Expected training time**:
- GPU: 20-30 minutes
- CPU: 2-3 hours

**Monitoring**:
- Training loss should decrease steadily
- Validation loss should track training loss
- Target: Training loss < 1.5, Validation loss < 1.8

#### Milestone 4.2: Debugging & Optimization
- Monitor for overfitting (val loss diverging from train loss)
- Adjust hyperparameters if needed
- Try gradient accumulation if memory issues
- Experiment with different batch sizes

---

### Phase 5: Inference & Generation (Week 3)

#### Milestone 5.1: Text Generation
**File to create**: `src/inference/generator.py`

**Implementation steps**:
1. Load trained model
2. Implement sampling strategies:
   - Greedy decoding
   - Temperature sampling
   - Top-k sampling
   - Top-p (nucleus) sampling
3. Generate text from prompts

**File to create**: `scripts/generate.py`
- CLI for text generation
- Adjustable temperature, top-k, top-p parameters

#### Milestone 5.2: Evaluation
**File to create**: `scripts/evaluate.py`

**Metrics to implement**:
1. Perplexity on test set
2. Sample generation quality (manual inspection)
3. Character distribution analysis

---

### Phase 6: Experimentation & Documentation (Week 4)

#### Milestone 6.1: Experiments
**Notebooks to create**:
1. `01_data_exploration.ipynb`: Analyze dataset statistics
2. `02_model_testing.ipynb`: Test individual components
3. `03_generation_examples.ipynb`: Generate and compare samples

**Experiments to try**:
- Different model sizes (smaller/larger)
- Different context lengths
- Different sampling temperatures
- Training on different datasets

#### Milestone 6.2: Documentation
**README.md sections**:
1. Project overview
2. Installation instructions
3. Usage examples (training, generation)
4. Architecture explanation
5. Results & sample outputs
6. Future improvements

---

## Key Implementation Details

### 1. Causal Self-Attention Mask
```python
# Prevent attending to future tokens
mask = torch.tril(torch.ones(block_size, block_size))
attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
```

### 2. Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Mixed Precision Training (Optional)
```python
# Use torch.cuda.amp for faster training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits, loss = model(x, y)
```

### 4. Checkpointing Strategy
- Save checkpoint every N iterations
- Save best model based on validation loss
- Include optimizer state for resuming training

---

## Testing Strategy

### Unit Tests
- `tests/test_tokenizer.py`: Encode/decode consistency
- `tests/test_attention.py`: Attention mask correctness, output shapes
- `tests/test_model.py`: Forward pass, parameter count, gradient flow

### Integration Tests
- End-to-end training for 10 iterations
- Generation from random prompt
- Checkpoint save/load

### Manual Validation
- Generated text quality
- Loss curves look reasonable
- Model learns to spell words correctly

---

## Performance Targets

### Model Size
- Parameters: ~10M (small, trainable on CPU)
- Context length: 256 characters
- Training time: < 1 hour on GPU, < 3 hours on CPU

### Quality Metrics
- Training loss: < 1.5
- Validation loss: < 1.8
- Perplexity: < 6.0
- Generated text: Mostly coherent Shakespeare-style language

---

## Potential Extensions (Post-MVP)

1. **Advanced Features**
   - Implement Flash Attention for efficiency
   - Add gradient accumulation
   - Multi-GPU training support

2. **Fine-tuning Stage**
   - Create instruction dataset
   - Implement supervised fine-tuning
   - Add LoRA for parameter-efficient tuning

3. **Better Tokenization**
   - Switch to BPE tokenizer
   - Train on larger vocabulary

4. **Deployment**
   - Export to ONNX
   - Quantization (INT8)
   - Simple web interface for generation

---

## Resources & References

### Essential Reading
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "Language Models are Few-Shot Learners" (GPT-3 paper)
3. Andrej Karpathy's "Let's build GPT" video
4. The Illustrated Transformer (Jay Alammar's blog)

### Code References
1. Andrej Karpathy's nanoGPT repository
2. HuggingFace Transformers library (for architecture inspiration)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Model doesn't train (loss stays high) | High | Start with proven hyperparameters, verify data pipeline, check gradients |
| Out of memory | Medium | Reduce batch size, use gradient accumulation, smaller model |
| Training too slow | Low | Use GPU, reduce model size, decrease iterations |
| Poor generation quality | Medium | Train longer, adjust temperature, check for bugs in attention mask |
| Overfitting | Low | Monitor val loss, add dropout, reduce model size |

---

## Success Criteria

**Minimum Viable Product (MVP)**:
- ✅ Model trains without errors
- ✅ Loss decreases consistently
- ✅ Can generate text that resembles Shakespeare
- ✅ Code is well-documented and modular

**Stretch Goals**:
- Generate multi-paragraph coherent text
- Implement multiple sampling strategies
- Add fine-tuning capability
- Create interactive demo

---

## Daily Development Checklist

**Every coding session**:
1. Pull latest changes (if team project)
2. Write tests first for new features
3. Implement feature
4. Run tests
5. Commit with descriptive message
6. Update documentation

**End of week**:
1. Review progress against milestones
2. Test full pipeline end-to-end
3. Generate sample outputs
4. Update README with learnings

---

## Getting Started Command Sequence

```bash
# 1. Create project
mkdir llm-from-scratch
cd llm-from-scratch

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Create requirements.txt and install
pip install torch numpy matplotlib tensorboard tqdm
pip freeze > requirements.txt

# 4. Initialize git
git init
echo "venv/\ncheckpoints/\nlogs/\n*.pyc\n__pycache__/\n.DS_Store" > .gitignore

# 5. Create directory structure
mkdir -p config data/{raw,processed} src/{model,training,utils,inference} scripts tests notebooks checkpoints logs

# 6. Start with data preparation
# Download Shakespeare data and implement tokenizer
```

---

This specification provides a complete roadmap. Start with Phase 1 and work sequentially through each milestone. Good luck building your LLM!