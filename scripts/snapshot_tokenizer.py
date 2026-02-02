
import argparse
import os
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Download tokenizer assets for offline use.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID to download tokenizer from.")
    parser.add_argument("--output", type=str, default="./tokenizer_assets", help="Output directory.")
    args = parser.parse_args()

    print(f"Downloading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Saving to {args.output}...")
    tokenizer.save_pretrained(args.output)
    print("Done. You can now use this directory for offline initialization:")
    print(f"  ruler = Ruler(model_name='{os.path.abspath(args.output)}')")

if __name__ == "__main__":
    main()
