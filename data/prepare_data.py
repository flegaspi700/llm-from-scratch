"""
Data preparation script for Shakespeare dataset.
Downloads raw data, creates tokenizer, and generates train/val splits.
"""
import numpy as np
import requests
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.tokenizer import WordTokenizer


def download_shakespeare():
    """Download Shakespeare dataset from Karpathy's char-rnn repo."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    # Create raw data directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    print("Downloading Shakespeare dataset...")
    response = requests.get(url)
    
    with open("data/raw/shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print(f"✓ Shakespeare dataset downloaded")
    print(f"  Dataset size: {len(response.text):,} characters")
    
    return response.text


def prepare_data(train_split: float = 0.9):
    """
    Prepare training and validation data.
    
    Steps:
    1. Load raw Shakespeare text
    2. Build word-level tokenizer vocabulary
    3. Encode entire dataset to token IDs
    4. Split into train (90%) and validation (10%)
    5. Save as binary files for fast loading
    
    Args:
        train_split: Fraction of data to use for training (default: 0.9)
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Load raw text
    raw_path = Path("data/raw/shakespeare.txt")
    if not raw_path.exists():
        print("Shakespeare dataset not found. Downloading...")
        download_shakespeare()
    
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"\nStep 1: Loaded raw text")
    print(f"  Total characters: {len(text):,}")
    
    # Build tokenizer
    print(f"\nStep 2: Building vocabulary...")
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(text)
    
    # Save vocabulary
    vocab_path = Path("data/processed/vocab.pkl")
    tokenizer.save(str(vocab_path))
    
    # Encode entire dataset
    print(f"\nStep 3: Encoding dataset...")
    encoded_data = tokenizer.encode(text)
    
    # Convert to numpy array with appropriate dtype
    # Use uint32 for word-level tokenization (vocab can be > 65535)
    encoded_array = np.array(encoded_data, dtype=np.uint32)
    
    print(f"  Total tokens: {len(encoded_array):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Tokens per character: {len(encoded_array) / len(text):.2f}")
    
    # Split into train and validation (90/10 split)
    print(f"\nStep 4: Splitting data (train: {train_split*100}%, val: {(1-train_split)*100}%)")
    split_idx = int(len(encoded_array) * train_split)
    train_data = encoded_array[:split_idx]
    val_data = encoded_array[split_idx:]
    
    print(f"  Train: {len(train_data):,} tokens ({len(train_data)/len(encoded_array)*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} tokens ({len(val_data)/len(encoded_array)*100:.1f}%)")
    
    # Create processed data directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as binary files
    print(f"\nStep 5: Saving binary files...")
    train_path = processed_dir / "train.bin"
    val_path = processed_dir / "val.bin"
    
    train_data.tofile(str(train_path))
    val_data.tofile(str(val_path))
    
    print(f"  ✓ Train data: {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f"  ✓ Val data:   {val_path} ({val_path.stat().st_size / 1024:.1f} KB)")
    print(f"  ✓ Vocabulary: {vocab_path} ({vocab_path.stat().st_size / 1024:.1f} KB)")
    
    # Print sample from training data
    print(f"\n" + "=" * 60)
    print("SAMPLE FROM TRAINING DATA")
    print("=" * 60)
    sample_ids = train_data[:100].tolist()
    sample_text = tokenizer.decode(sample_ids)
    print(f"{sample_text}...")
    
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"✓ Raw data:      {raw_path}")
    print(f"✓ Vocabulary:    {vocab_path} ({tokenizer.vocab_size} tokens)")
    print(f"✓ Training set:  {train_path} ({len(train_data):,} tokens)")
    print(f"✓ Validation set: {val_path} ({len(val_data):,} tokens)")
    print(f"\nYou can now proceed to Phase 2: Model Architecture")
    
    return tokenizer, train_data, val_data


def verify_data_integrity():
    """
    Verify that all processed data files exist and can be loaded correctly.
    """
    print("\n" + "=" * 60)
    print("VERIFYING DATA INTEGRITY")
    print("=" * 60)
    
    required_files = {
        'vocab': Path("data/processed/vocab.pkl"),
        'train': Path("data/processed/train.bin"),
        'val': Path("data/processed/val.bin")
    }
    
    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"✓ {name:10s}: {path} ({size:.1f} KB)")
        else:
            print(f"✗ {name:10s}: {path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠ Some files are missing. Run prepare_data() first.")
        return False
    
    # Test loading
    print("\nTesting data loading...")
    
    # Load tokenizer
    tokenizer = WordTokenizer()
    tokenizer.load(str(required_files['vocab']))
    print(f"✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")
    
    # Load binary data
    train_data = np.fromfile(required_files['train'], dtype=np.uint32)
    val_data = np.fromfile(required_files['val'], dtype=np.uint32)
    print(f"✓ Train data loaded ({len(train_data):,} tokens)")
    print(f"✓ Val data loaded ({len(val_data):,} tokens)")
    
    # Verify tokens are valid
    max_train_token = train_data.max()
    max_val_token = val_data.max()
    
    if max_train_token >= tokenizer.vocab_size:
        print(f"✗ Invalid token in train data: {max_train_token} >= {tokenizer.vocab_size}")
        return False
    if max_val_token >= tokenizer.vocab_size:
        print(f"✗ Invalid token in val data: {max_val_token} >= {tokenizer.vocab_size}")
        return False
    
    print(f"✓ All tokens are valid (max train: {max_train_token}, max val: {max_val_token})")
    
    # Test decoding
    sample = train_data[:50]
    decoded = tokenizer.decode(sample.tolist())
    print(f"\nSample decoded text:\n  {decoded}...")
    
    print("\n" + "=" * 60)
    print("✓ ALL CHECKS PASSED - Data pipeline is ready!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Shakespeare dataset for training")
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data without re-processing'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Fraction of data for training (default: 0.9)'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_data_integrity()
    else:
        # Run full preparation
        prepare_data(train_split=args.train_split)
        
        # Verify everything worked
        print("\n")
        verify_data_integrity()