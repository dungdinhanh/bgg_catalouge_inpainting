import argparse
import json
import os

def convert_txt_to_jsonl(txt_path):
    base, _ = os.path.splitext(txt_path)
    jsonl_path = base + ".json"

    with open(txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    with open(jsonl_path, 'w') as f_out:
        for image_path in image_paths:
            record = {
                "text": "",
                "image": image_path,
                "mask": image_path  # same as image
            }
            f_out.write(json.dumps(record) + '\n')

    print(f"Saved JSON to {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a TXT file of image paths to JSON lines format.")
    parser.add_argument("--input", help="Path to the input .txt file")
    args = parser.parse_args()

    convert_txt_to_jsonl(args.input)
