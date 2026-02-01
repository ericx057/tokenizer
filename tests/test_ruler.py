
import pytest
from unittest.mock import MagicMock, patch
from tokenizer import Ruler, TokenizerMismatchError

@pytest.fixture
def mock_tokenizer():
    with patch("tokenizer.ruler.AutoTokenizer") as mock_auto:
        mock_instance = MagicMock()
        mock_instance.vocab_size = 128256
        mock_instance.encode.return_value = [1, 2, 3]
        mock_instance.decode.return_value = "hello"
        mock_instance.apply_chat_template.return_value = "formatted prompt"
        
        mock_auto.from_pretrained.return_value = mock_instance
        yield mock_instance

def test_ruler_initialization(mock_tokenizer):
    ruler = Ruler()
    assert ruler.vocab_size == 128256

def test_ruler_vocab_mismatch():
    with patch("tokenizer.ruler.AutoTokenizer") as mock_auto:
        mock_instance = MagicMock()
        mock_instance.vocab_size = 100 # Wrong size
        mock_auto.from_pretrained.return_value = mock_instance
        
        with pytest.raises(TokenizerMismatchError):
            Ruler(expected_vocab_size=128256)

def test_tokenize_caching(mock_tokenizer):
    ruler = Ruler()
    tokens1 = ruler.tokenize("test")
    tokens2 = ruler.tokenize("test")
    
    assert tokens1 == [1, 2, 3]
    # Verify cache usage by checking if encode was called only once if we were mocking properly, 
    # but here we just check output consistency.
    # To check cache hits we can inspect cache_info
    info = ruler.cache_info()
    assert info["tokenize"]["hits"] == 1
    assert info["tokenize"]["misses"] == 1

def test_count_prompt_tokens(mock_tokenizer):
    ruler = Ruler()
    # Mock return for count prompt tokens
    mock_tokenizer.apply_chat_template.return_value = [10, 20, 30, 40] 
    
    count = ruler.count_prompt_tokens("sys", "user")
    assert count == 4
