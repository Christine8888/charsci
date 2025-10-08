import asyncio
from random import shuffle
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm.asyncio import tqdm
from threading import Lock

from inspect_ai.model import get_model, ChatMessage, ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, GenerateConfig

from cai_prompt_datasets import load_ultrachat, load_ant_redteaming, load_evol_instruct, load_conversation_starters

# Import system prompt registry from code_data (optional)
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from code_data.prompts.system import system
except ImportError:
    system = None


# Global lock for file writing
write_lock = Lock()


async def sample_completion(
    messages: List[Dict[str, str]],
    model_id: str = "openai/gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: Optional[int] = 4096,
    system_prompt_id: Optional[str] = None,
) -> str:
    """
    Sample a completion using Inspect AI's inference API.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model_id: Model to use for completion (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt_id: System prompt ID from registry (None for no system prompt)

    Returns:
        Generated completion text

    Raises:
        ValueError: If system_prompt_id is provided but not found in registry
    """
    # Convert messages to ChatMessage objects
    chat_messages = []

    # Add system prompt if provided
    if system_prompt_id is not None:
        if system is None:
            raise ImportError("code_data.prompts.system module not available. Cannot use system_prompt_id.")
        if system_prompt_id not in system.list_ids():
            raise ValueError(f"System prompt ID '{system_prompt_id}' not found in registry. Available IDs: {system.list_ids()}")
        system_content = system.get(system_prompt_id)
        chat_messages.append(ChatMessageSystem(content=system_content))

    for msg in messages:
        if msg["role"] == "user":
            chat_messages.append(ChatMessageUser(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_messages.append(ChatMessageAssistant(content=msg["content"]))
        elif msg["role"] == "system":
            chat_messages.append(ChatMessageSystem(content=msg["content"]))

    # Get model and generate
    async with get_model(model_id) as model:
        response = await model.generate(
            input=chat_messages,
            config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        )

    return response.completion


async def process_single_sample(
    example: Dict,
    index: int,
    dataset_name: str,
    model_id: str,
    temperature: float,
    max_tokens: Optional[int],
    output_path: Path,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    system_prompt_id: Optional[str] = None,
) -> Optional[Dict]:
    """Process a single sample with semaphore control."""
    async with semaphore:
        messages = example["messages"]

        try:
            # Get completion
            completion = await sample_completion(
                messages=messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt_id=system_prompt_id,
            )

            # Create result with unique ID
            result = {
                "id": f"{dataset_name}_{index}",
                "messages": messages,
                "completion": completion,
                "model": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt_id": system_prompt_id,
            }

            # Write incrementally with thread lock
            with write_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

            pbar.update(1)
            return result

        except Exception as e:
            print(f"\nError processing example {index}: {e}")
            pbar.update(1)
            return None


async def sample_completions_for_dataset(
    dataset_name: str = "ultrachat",
    model_id: str = "openai/gpt-4o-mini",
    num_samples: int = 10,
    output_file: str = "sampled_completions.jsonl",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
    max_concurrent: int = 5,
    max_conv_length: int = 10,
    shuffle: bool = True,
    system_prompt_id: Optional[str] = None,
):
    """
    Sample completions for messages from a dataset using Inspect AI.

    Args:
        dataset_name: Name of the dataset to use
        model_id: Model to use for completions (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022")
        num_samples: Number of samples to generate
        output_file: Path to save completions
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_concurrent: Maximum number of concurrent API requests
        max_conv_length: Maximum conversation length (for applicable datasets)
        shuffle: Whether to shuffle the dataset
        system_prompt_id: System prompt ID from registry (None for no system prompt)
    """

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "ultrachat":
        dataset = load_ultrachat(
            size=num_samples, max_conv_length=max_conv_length, shuffle=shuffle
        )
    elif dataset_name == "ant_redteaming":
        dataset = load_ant_redteaming(
            size=num_samples, max_conv_length=max_conv_length, shuffle=shuffle
        )
    elif dataset_name == "evol_instruct":
        dataset = load_evol_instruct(size=num_samples)
    elif dataset_name == "conversation_starters":
        dataset = load_conversation_starters(
            size=num_samples, shuffle=shuffle
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert to list to allow indexing
    dataset_list = list(dataset)

    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if output_path.exists():
        output_path.unlink()

    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Sample completions
    print(
        f"Sampling {len(dataset_list)} completions using {model_id} with max {max_concurrent} concurrent requests"
    )

    # Create progress bar
    pbar = tqdm(total=len(dataset_list), desc="Processing samples")

    # Create tasks for all samples
    tasks = [
        process_single_sample(
            example=example,
            index=i,
            dataset_name=dataset_name,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            output_path=output_path,
            semaphore=semaphore,
            pbar=pbar,
            system_prompt_id=system_prompt_id,
        )
        for i, example in enumerate(dataset_list)
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Filter out None results
    successful_results = [r for r in results if r is not None]

    print(f"Completed! Saved {len(successful_results)} completions to {output_path}")
    return successful_results


async def main():
    """Main function to run completion sampling."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample completions from language models"
    )
    parser.add_argument(
        "--dataset", type=str, default="ultrachat", help="Dataset to use (ultrachat, ant_redteaming, evol_instruct, conversation_starters)"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-4o-mini", help="Model ID to use (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sampled_completions.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Maximum concurrent API requests"
    )
    parser.add_argument(
        "--max-conv-length", type=int, default=10, help="Maximum conversation length"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset",
    )
    parser.add_argument(
        "--system-prompt-id",
        type=str,
        default=None,
        help="System prompt ID from registry (None for no system prompt)",
    )
    args = parser.parse_args()

    await sample_completions_for_dataset(
        dataset_name=args.dataset,
        model_id=args.model,
        num_samples=args.num_samples,
        output_file=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        max_conv_length=args.max_conv_length,
        shuffle=args.shuffle,
        system_prompt_id=args.system_prompt_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
