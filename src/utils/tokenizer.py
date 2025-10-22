"""
Word-level tokenizer for the Shakespeare dataset.
Uses regex to split on punctuation and whitespace while preserving tokens.
Handles contractions like "don't", "we'll", "it's" as single tokens.
"""
import json
import pickle
from pydoc import text
import re
from pathlib import Path
from typing import List, Dict


class WordTokenizer:
    """Word-level tokenizer with punctuation handling."""
    
    # Special tokens
    UNK_TOKEN = "<UNK>"  # Unknown token
    PAD_TOKEN = "<PAD>"  # Padding token (for batching)
    
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.unk_id: int = 0  # ID for unknown tokens
        self.pad_id: int = 1  # ID for padding tokens
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by splitting on punctuation and whitespace.
        Keeps contractions together (don't, we'll, it's).
        
        Args:
            text: Raw text string
            
        Returns:
            List of tokens (words and punctuation)
        """
        # First, protect contractions by handling them specially
        # Match word + apostrophe + letters (don't, we'll, it's, etc.)
        # Also match standalone punctuation and words
        pattern = r"(\w+'\w+|\w+|[,.:;?_!\"()\']|--)"
        tokens = re.findall(pattern, text)
        
        # Remove empty strings and strip whitespace
        tokens = [item.strip() for item in tokens if item.strip()]
        
        # Convert to lowercase
        tokens = [item.lower() for item in tokens]
        
        return tokens
    
    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text using word-level tokenization.
        
        Args:
            text: Raw text string to build vocabulary from
        """
        # Preprocess text to get tokens
        tokens = self._preprocess_text(text)
        
        print(f"Total tokens after preprocessing: {len(tokens):,}")
        print(f"First 20 tokens: {tokens[:20]}")
        
        # Get unique tokens and sort them for consistency
        unique_tokens = sorted(set(tokens))
        
        # Add special tokens at the beginning
        special_tokens = [self.UNK_TOKEN, self.PAD_TOKEN]
        all_tokens = special_tokens + unique_tokens
        
        self.vocab_size = len(all_tokens)
        
        # Create token to ID mapping
        self.token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        
        # Create ID to token mapping (for decoding)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Store special token IDs
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        
        print(f"\nVocabulary built with {self.vocab_size} unique tokens")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Regular tokens: {self.vocab_size - len(special_tokens)}")
        print(f"\nFirst 20 items in vocabulary:")
        for i, (token, idx) in enumerate(list(self.token_to_id.items())[:20]):
            print(f"  Token: '{token}' \t Token ID: {idx}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text string to list of token IDs.
        Unknown tokens are mapped to <UNK>.
        
        Args:
            text: String to encode
            
        Returns:
            List of integer token IDs
        """
        tokens = self._preprocess_text(text)
        # Use get() with default to unk_id instead of silently dropping
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode list of token IDs back to text string.
        
        Args:
            token_ids: List of integer token IDs
            skip_special_tokens: If True, skip <UNK> and <PAD> tokens
            
        Returns:
            Decoded text string
        """
        tokens = [self.id_to_token[idx] for idx in token_ids]
        
        # Optionally filter out special tokens
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.UNK_TOKEN, self.PAD_TOKEN]]
        
        # Reconstruct text with proper spacing
        result = []
        for i, token in enumerate(tokens):
            # Add space before token if:
            # - Not the first token
            # - Not punctuation (except apostrophe in contractions)
            # - Previous token wasn't an opening quote/paren
            if i > 0 and token not in [',', '.', ':', ';', '?', '!', ')', '"']:
                # Don't add space if token starts with apostrophe (contractions)
                if not (token.startswith("'") and len(token) > 1):
                    result.append(' ')
            result.append(token)
        
        return ''.join(result)

    def save(self, filepath: str) -> None:
        """
        Save vocabulary to disk.
        
        Args:
            filepath: Path to save vocabulary file
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'unk_id': self.unk_id,
            'pad_id': self.pad_id,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load vocabulary from disk.
        
        Args:
            filepath: Path to vocabulary file
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = vocab_data['id_to_token']
        self.vocab_size = vocab_data['vocab_size']
        self.unk_id = vocab_data['unk_id']
        self.pad_id = vocab_data['pad_id']
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocab size: {self.vocab_size}")


def test_tokenizer():
    """Test the WordTokenizer on the Shakespeare dataset."""
    # Load raw Shakespeare dataset
    with open("data/raw/shakespeare.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Dataset size: {len(raw_text):,} characters\n")
    print(f"First 100 characters:\n{raw_text[:100]}\n")
    print("=" * 60)

    # Initialize and build vocabulary
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(raw_text)

    print("\n" + "=" * 60)
    print("TESTING ENCODE/DECODE")
    print("=" * 60)

    # Test encode/decode with out-of-vocabulary words
    sample_text = "Hello, World! This is a test. Don't you love Shakespeare?"
    print(f"\nOriginal text: {sample_text}")
    
    # Show tokenization
    tokens = tokenizer._preprocess_text(sample_text)
    print(f"Tokens: {tokens}")
    
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    decoded_no_special = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded (skip special): {decoded_no_special}")

    # Test contractions specifically
    print("\n" + "=" * 60)
    print("TESTING CONTRACTIONS")
    print("=" * 60)
    contraction_tests = [
        "don't",
        "we'll", 
        "it's",
        "they're",
        "I'm",
        "wouldn't"
    ]
    
    for contraction in contraction_tests:
        tokens = tokenizer._preprocess_text(contraction)
        print(f"  '{contraction}' → {tokens}")

    # Test with Shakespeare excerpt (should have no unknowns)
    excerpt = raw_text[1000:1200]
    print(f"\n\nShakespeare excerpt (original):\n{excerpt}")
    
    encoded_excerpt = tokenizer.encode(excerpt)
    print(f"\nEncoded ({len(encoded_excerpt)} tokens): {encoded_excerpt[:20]}...")
    
    # Count unknown tokens
    unk_count = sum(1 for tok_id in encoded_excerpt if tok_id == tokenizer.unk_id)
    print(f"Unknown tokens in excerpt: {unk_count}")
    
    decoded_excerpt = tokenizer.decode(encoded_excerpt)
    print(f"\nDecoded:\n{decoded_excerpt}")
    
    # Verify encode/decode roundtrip - normalize whitespace for comparison
    # Since decode may have slightly different spacing, we check token-level consistency
    original_tokens = tokenizer._preprocess_text(excerpt)
    roundtrip_tokens = tokenizer._preprocess_text(decoded_excerpt)
    
    if original_tokens == roundtrip_tokens:
        print("✓ Shakespeare encode/decode test passed! (Token-level consistency)")
    else:
        print("⚠ Warning: Token-level mismatch detected")
        print(f"  Original tokens: {len(original_tokens)}")
        print(f"  Roundtrip tokens: {len(roundtrip_tokens)}")
        # Show first difference
        for i, (orig, rt) in enumerate(zip(original_tokens, roundtrip_tokens)):
            if orig != rt:
                print(f"  First diff at position {i}: '{orig}' != '{rt}'")
                break
    
    # Vocabulary statistics
    print("\n" + "=" * 60)
    print("VOCABULARY STATISTICS")
    print("=" * 60)
    print(f"Total unique tokens: {tokenizer.vocab_size}")
    print(f"Total tokens in dataset: {len(tokenizer.encode(raw_text)):,}")
    
    # Check for contractions in vocab
    # Fixed: Use tokenizer.UNK_TOKEN or WordTokenizer.UNK_TOKEN instead of self.UNK_TOKEN
    contractions_in_vocab = [t for t in tokenizer.token_to_id.keys() 
                            if "'" in t and t not in [WordTokenizer.UNK_TOKEN, "'"]]
    print(f"\nContractions found in Shakespeare vocab: {len(contractions_in_vocab)}")
    print(f"Sample contractions: {contractions_in_vocab[:10]}")
    
    # Sample some tokens
    print(f"\nSample tokens from vocabulary:")
    sample_tokens = list(tokenizer.token_to_id.items())[::len(tokenizer.token_to_id)//10]
    for token, idx in sample_tokens[:10]:
        print(f"  '{token}' → {idx}")
 
    # Save vocabulary
    print("\n" + "=" * 60)
    tokenizer.save("data/processed/vocab.pkl")
    
    # Test loading
    new_tokenizer = WordTokenizer()
    new_tokenizer.load("data/processed/vocab.pkl")
    assert new_tokenizer.vocab_size == tokenizer.vocab_size, "Vocab size mismatch after load!"
    print("✓ Save/load test passed!")
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = test_tokenizer()