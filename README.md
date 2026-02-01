# Tokenizer

A standalone library for strict Llama 3.1 token counting and prompt building.

## Features
- **Strict Alignment**: Ensures alignment with Llama 3.1 tokenizer configuration.
- **Prompt Building**: Uses native chat templates for accurate prompt construction.
- **Caching**: Includes LRU caching to minimize redundant tokenization overhead.

## Usage

```python
from tokenizer import Ruler

# Initialize (defaults to Llama 3.1)
ruler = Ruler()

# Count tokens (ignoring special tokens)
count = ruler.count("Hello world")

# Count full prompt (including special tokens)
prompt_tokens = ruler.count_prompt_tokens(
    system_content="You are a helpful assistant.",
    user_content="Explain quantum gravity."
)

print(f"Prompt Tokens: {prompt_tokens}")
```
