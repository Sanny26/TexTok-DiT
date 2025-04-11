# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/webdataset/webdataset-imagenet/blob/main/convert-imagenet.py

import argparse
import os
import sys
import time

import webdataset as wds
from datasets import load_dataset
from huggingface_hub import login


def convert_imagenet21krecaption_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    """Convert Imagenet21K_Recaption dataset to WebDataset format.
    
    Args:
        output_dir: Directory to save the WebDataset files
        max_samples_per_shard: Maximum number of samples per shard
        data_dir: Optional local directory containing the dataset
    """
    assert not os.path.exists(os.path.join(output_dir, "imagenet21k-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, "imagenet21k-val-000000.tar"))

    # Load dataset
    print("Loading Imagenet21K_Recaption dataset...")
    opat = os.path.join(output_dir, "imagenet21k-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    
    # Load dataset with streaming
    
    dataset = load_dataset(
        "gmongaras/Imagenet21K_Recaption",
        streaming=True,
        split="train",
        token=True
    )
    
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_train_samples_per_shard == 0:
            print(f"Processing example {i}", file=sys.stderr)
        import pdb; pdb.set_trace()
        # Get image and captions
        img = example["image"]
        long_caption = example.get("recaption", [])  # Handle case where captions might not exist
        short_caption = example.get("recaption_short", [])  # Handle case where captions might not exist
        
        # Write to WebDataset format
        output.write({
            "__key__": "%08d" % i,
            "jpg": img.convert("RGB"),
            "txt_long": long_caption[0] if long_caption else "",  # Use first caption if available
            "txt_short": short_caption[0] if short_caption else "",  # Use first caption if available
            "cls": example.get("label", -1)  # Use -1 if label doesn't exist
        })
    
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} examples in {time_taken // 3600} hours.")


# def convert_imagenet1kcaption_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
#     """Convert Imagenet21K_Recaption dataset to WebDataset format.
    
#     Args:
#         output_dir: Directory to save the WebDataset files
#         max_samples_per_shard: Maximum number of samples per shard
#         data_dir: Optional local directory containing the dataset
#     """
#     # assert not os.path.exists(os.path.join(output_dir, "imagenet21k-train-000000.tar"))
#     # assert not os.path.exists(os.path.join(output_dir, "imagenet21k-val-000000.tar"))

#     # Load dataset
#     # print("Loading Imagenet1K_Recaption dataset...")
#     # opat = os.path.join(output_dir, "imagenet1k-val-%06d.tar")
#     # output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    
#     # # Load validation dataset with streaming
#     # dataset = load_dataset(
#     #     "visual-layer/imagenet-1k-vl-enriched",
#     #     streaming=True,
#     #     split="validation",
#     #     token=True
#     # )
#     # now = time.time()
#     # for i, example in enumerate(dataset):
#     #     if i % max_train_samples_per_shard == 0:
#     #         print(f"Processing example {i}", file=sys.stderr)
#     #     # import pdb; pdb.set_trace()
#     #     # Get image and captions
#     #     id = example["image_id"]
#     #     img = example["image"]
#     #     caption = example.get("caption_enriched", [])  # Handle case where captions might not exist
#     #     # print(img, caption, end='\r')
#     #     # Write to WebDataset format
#     #     output.write({
#     #         "__key__": "%08d" % i,
#     #         "jpg": img.convert("RGB"),
#     #         "txt": caption if caption else "",  # Use first caption if available
#     #         "cls": example.get("label", -1),  # Use -1 if label doesn't exist
#     #         "id": id
#     #     })
    
#     # output.close()
#     # time_taken = time.time() - now
#     # print(f"Wrote {i+1} val examples in {time_taken // 3600} hours.")


#     # Load training dataset with streaming
#     print("Loading training set...")
#     opat = os.path.join(output_dir, "imagenet1k-train-%06d.tar")
#     output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
#     dataset = load_dataset(
#         "visual-layer/imagenet-1k-vl-enriched",
#         streaming=True,
#         split="train",
#         token=True
#     )
#     now = time.time()
#     for i, example in enumerate(dataset):
#         if i % max_train_samples_per_shard == 0:
#             print(f"Processing example {i}", file=sys.stderr)
#         # import pdb; pdb.set_trace()
#         # Get image and captions
#         id = example["image_id"]
#         img = example["image"]
#         caption = example.get("caption_enriched", [])  # Handle case where captions might not exist
        
#         # Write to WebDataset format    
#         output.write({
#             "__key__": "%08d" % i,
#             "jpg": img.convert("RGB"),
#             "txt": caption[0] if caption else "",  # Use first caption if available
#             "cls": example.get("label", -1),  # Use -1 if label doesn't exist
#             "id": id
#         })  
#     output.close()
#     time_taken = time.time() - now
#     print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

def convert_imagenet1kcaption_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    """Convert Imagenet1K with captions dataset to WebDataset format.
    
    Args:
        output_dir: Directory to save the WebDataset files
        max_train_samples_per_shard: Maximum number of samples per training shard
        max_val_samples_per_shard: Maximum number of samples per validation shard
    """
    # Check for existing shards and find the last one
    def get_last_shard_number(pattern):
        import glob
        shards = glob.glob(pattern)
        if not shards:
            return -1
        # Extract numbers from filenames and find the max
        numbers = [int(os.path.basename(s).split('-')[-1].split('.')[0]) for s in shards]
        return max(numbers)

    # Training set
    train_pattern = os.path.join(output_dir, "imagenet1k-train-*.tar")
    last_train_shard = get_last_shard_number(train_pattern)
    start_train_idx = (last_train_shard + 1) * max_train_samples_per_shard if last_train_shard >= 0 else 0

    print(f"Starting training set from index {start_train_idx}")
    opat = os.path.join(output_dir, "imagenet1k-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    
    dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        streaming=True,
        split="train",
        token=True
    )
    
    # Get the iterator directly
    iterator = iter(dataset)
    
    # Fast forward to the starting index
    for _ in range(start_train_idx):
        next(iterator, None)
    
    now = time.time()
    for i, example in enumerate(iterator, start=start_train_idx):
        if i % max_train_samples_per_shard == 0:
            print(f"Processing training example {i}", file=sys.stderr)
        
        try:
            # Get image and captions
            id = example["image_id"]
            img = example["image"]
            caption = example.get("caption_enriched", [])
            
            # Skip if image is corrupt
            if img is None:
                print(f"Skipping corrupt image with id: {hash(id)}")
                continue
                
            # Write to WebDataset format    
            output.write({
                "__key__": "%08d" % i,
                "jpg": img.convert("RGB"),
                "txt": caption[0] if caption else "",  # Use first caption if available
                "cls": example.get("label", -1),  # Use -1 if label doesn't exist
                "id": id
            })
        except Exception as e:
            print(f"Error processing image with id {hash(id)}: {str(e)}")
            continue
    
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    # Validation set
    val_pattern = os.path.join(output_dir, "imagenet1k-val-*.tar")
    last_val_shard = get_last_shard_number(val_pattern)
    start_val_idx = (last_val_shard + 1) * max_val_samples_per_shard if last_val_shard >= 0 else 0

    print(f"Starting validation set from index {start_val_idx}")
    opat = os.path.join(output_dir, "imagenet1k-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    
    dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        streaming=True,
        split="validation",
        token=True
    )
    
    # Get the iterator directly
    iterator = iter(dataset)
    
    # Fast forward to the starting index
    for _ in range(start_val_idx):
        next(iterator, None)
    
    now = time.time()
    for i, example in enumerate(iterator, start=start_val_idx):
        if i % max_val_samples_per_shard == 0:
            print(f"Processing validation example {i}", file=sys.stderr)
        
        try:
            # Get image and captions
            id = example["image_id"]
            img = example["image"]
            caption = example.get("caption_enriched", [])
            
            # Skip if image is corrupt
            if img is None:
                print(f"Skipping corrupt image with id: {hash(id)}")
                continue
                
            # Write to WebDataset format    
            output.write({
                "__key__": "%08d" % i,
                "jpg": img.convert("RGB"),
                "txt": caption[0] if caption else "",  # Use first caption if available
                "cls": example.get("label", -1),  # Use -1 if label doesn't exist
                "id": id
            })
        except Exception as e:
            print(f"Error processing image with id {hash(id)}: {str(e)}")
            continue
    
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 3600} hours.")

def convert_imagenet_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    assert not os.path.exists(os.path.join(output_dir, "imagenet-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, "imagenet-val-000000.tar"))

    # Load training set
    print("Loading training set...")
    opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    # dataset = load_dataset("imagenet-1k", streaming=True, split="train", use_auth_token=True)
    # clear_cache("imagenet-1k")
    dataset = load_dataset("imagenet-1k", streaming=True, split="train", token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_train_samples_per_shard == 0:
            print(f"Processing training example {i}", file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    # Load validation set
    print("Loading validation set...")
    opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    dataset = load_dataset("imagenet-1k", streaming=True, split="validation", token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_val_samples_per_shard == 0:
            print(f"Processing validation example {i}", file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 60} min.")


def check_dataset_info():
    """Check and print information about the Imagenet21K_Recaption dataset."""
    print("Loading dataset info...")
    dataset = load_dataset(
        "gmongaras/Imagenet21K_Recaption",
        streaming=True,
        split="train",
        token=True
    )
    
    # Get dataset info
    print("\nDataset Features:")
    print(dataset.features)
    
    # Count total samples
    print("\nCounting total samples (this may take a while)...")
    total_samples = 0
    for _ in dataset:
        total_samples += 1
        if total_samples % 1000 == 0:
            print(f"Counted {total_samples} samples so far...")
    
    print(f"\nTotal samples in dataset: {total_samples}")
    
    # Get first example to show structure
    print("\nFirst example structure:")
    dataset = load_dataset(
        "gmongaras/Imagenet21K_Recaption",
        streaming=True,
        split="train",
        token=True
    )
    first_example = next(iter(dataset))
    print("\nKeys in example:")
    for key in first_example.keys():
        print(f"- {key}")
        if isinstance(first_example[key], list):
            print(f"  Length: {len(first_example[key])}")
        elif hasattr(first_example[key], 'size'):
            print(f"  Size: {first_example[key].size}")


if __name__ == "__main__":
    # create parase object
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--max_train_samples_per_shard", type=int, default=4000, help="Path to the output directory.")
    parser.add_argument("--max_val_samples_per_shard", type=int, default=1000, help="Path to the output directory.")
    parser.add_argument("--token", type=str, help="Hugging Face token for authentication")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "imagenet21k", "imagenet1k-caption"], default="imagenet21k",
                      help="Dataset to convert (default: imagenet21k)")
    parser.add_argument("--check_info", action="store_true", help="Check dataset information instead of converting")
    args = parser.parse_args()

    # Login to Hugging Face
    if args.token:
        login(token=args.token)
    else:
        # Try to use token from environment variable
        login()

    if args.check_info:
        check_dataset_info()
        sys.exit(0)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "cache"), exist_ok=True)

    if args.dataset == "imagenet21k":
        convert_imagenet21krecaption_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)
    elif args.dataset == "imagenet1k-caption":
        convert_imagenet1kcaption_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)
    else:
    convert_imagenet_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)