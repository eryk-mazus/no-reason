import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fire
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

STEP_SEPARATOR = "\n\n"

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

system_prompt = f"""You are an expert AI assistant. Provide the reasoning how to approach and solve the given problem. Separate each reasoning step with `\n\n` separator.
Start from understanding the problem and analysing the given examples if they're provided.
Plan your work by making observations on possible approaches and the constraints.
Try out the approaches that you proposed. Test them internally using the reasoning.
If an approach is not promising or incorrect, describe your mistake and switch to a different approach.
Keep the individual reasoning steps short.
"""

default_user_prompt = r"""Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"""


@dataclass
class TreeNode:
    # for visualization, debugging:
    text: str
    token_ids: torch.Tensor

    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    depth: int = 0

    cot_confidence_score: float = 0.0
    log_prob: float = 0.0

    def add_child(self, child: "TreeNode"):
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "depth": self.depth,
            "cot_confidence_score": self.cot_confidence_score,
            # "log_prob": self.log_prob,
            "children": [child.to_dict() for child in self.children],
        }


def calculate_entropy(probs: torch.Tensor, top_k: int = None) -> float:
    """Calculate the entropy of a probability distribution, optionally using only top-k probabilities."""
    if top_k is not None:
        top_k_probs, _ = torch.topk(probs, k=top_k)
        # Renormalize the top-k probabilities
        top_k_probs = top_k_probs / torch.sum(top_k_probs)
        return -torch.sum(top_k_probs * torch.log2(top_k_probs))
    else:
        return -torch.sum(probs * torch.log2(probs))


def decide_num_branches(entropy: float, top_k: int, max_branches: int) -> int:
    """
    Decide the number of branches based on entropy of top-k tokens.
    Automatically adjusts thresholds based on the maximum possible entropy for the given top_k.
    """
    max_entropy = math.log2(top_k)

    # Define thresholds as fractions of max_entropy
    thresholds = [
        0.2 * max_entropy,  # Very low entropy
        0.4 * max_entropy,  # Low entropy
        0.6 * max_entropy,  # Medium entropy
        0.8 * max_entropy,  # High entropy
    ]

    if entropy < thresholds[0]:
        return 1
    elif entropy < thresholds[1]:
        return min(2, max_branches)
    elif entropy < thresholds[2]:
        return min(3, max_branches)
    elif entropy < thresholds[3]:
        return min(4, max_branches)
    else:
        return max_branches


def calculate_confidence_score(token_probs: List[torch.Tensor]) -> float:
    confidence_scores = []
    for probs in token_probs:
        top_two_probs, _ = torch.topk(probs, k=2)
        confidence_scores.append((top_two_probs[0] - top_two_probs[1]).item())
    return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0


def expand_branch(
    input_ids: torch.Tensor,
    first_token_id: int,
    first_token_log_prob: float,
    max_new_tokens: int,
) -> Tuple[str, List[float], float, torch.Tensor, bool]:
    current_input_ids = torch.cat(
        [input_ids, first_token_id.unsqueeze(0).unsqueeze(0)], dim=-1
    )
    log_probs = [first_token_log_prob]
    token_probs = []
    is_terminal = False

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(current_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            token_probs.append(next_token_probs[0])

            next_token_id = torch.argmax(next_token_probs, dim=-1)
            log_prob = torch.log(next_token_probs[0, next_token_id]).item()

            log_probs.append(log_prob)
            current_input_ids = torch.cat(
                [current_input_ids, next_token_id.unsqueeze(0)], dim=-1
            )

            if next_token_id.item() == tokenizer.eos_token_id:
                is_terminal = True
                break

            last_token = tokenizer.decode([next_token_id.item()])
            if STEP_SEPARATOR in last_token:
                break

    generated_text = tokenizer.decode(
        current_input_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    confidence_score = calculate_confidence_score(token_probs)
    return generated_text, log_probs, confidence_score, current_input_ids, is_terminal


def expand_tree(
    root: TreeNode, max_depth: int, top_k: int, max_branches: int, max_new_tokens: int
) -> TreeNode:
    current_node = root

    while current_node.depth < max_depth:
        with torch.no_grad():
            outputs = model(current_node.token_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        top_k_entropy = calculate_entropy(next_token_probs[0], top_k=top_k)
        num_branches = decide_num_branches(
            top_k_entropy, top_k, max_branches=max_branches
        )

        top_k_probs, top_k_indices = torch.topk(next_token_probs, num_branches)

        best_child = None
        best_score = float("-inf")

        logger.debug(f"Expanding node at depth {current_node.depth}")
        logger.debug(f"Number of branches: {num_branches}")

        for i in range(num_branches):
            first_token_id = top_k_indices[0, i]
            first_token_log_prob = torch.log(top_k_probs[0, i]).item()

            generated_text, log_probs, confidence_score, new_input_ids, is_terminal = (
                expand_branch(
                    current_node.token_ids,
                    first_token_id,
                    first_token_log_prob,
                    max_new_tokens,
                )
            )

            avg_log_probs = sum(log_probs) / len(log_probs)

            child_node = TreeNode(
                text=generated_text,
                token_ids=new_input_ids,
                depth=current_node.depth + 1,
                cot_confidence_score=confidence_score,
                log_prob=avg_log_probs,
            )
            current_node.add_child(child_node)

            logger.debug(f"Candidate {i + 1}:")
            logger.debug(f"Generated text: {generated_text}")
            logger.debug(f"Confidence score: {confidence_score:.4f}")
            logger.debug(f"Avg of log probabilities: {avg_log_probs:.4f}")
            logger.debug(f"Is terminal: {is_terminal}")
            logger.debug("---")

            if confidence_score > best_score:
                best_score = confidence_score
                best_child = child_node

        if best_child is None or is_terminal:
            break

        current_node = best_child
        logger.info(f"Expanded to depth {current_node.depth}")
        logger.info(f"Text: {current_node.text}")
        logger.info(f"Confidence Score: {current_node.cot_confidence_score:.4f}")
        logger.info("---")

    return root


def main(
    user_prompt: str = default_user_prompt,
    top_k: int = 5,  # number of top_k tokens to consider while calculating entropy
    max_branches: int = 5,  # max number of branches to spawn per step - should be equal or less than top_k
    max_depth: int = 20,  # max depth of the search tree
    max_new_tokens: int = 256,  # max number of tokens to generate per step
):
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    root = TreeNode(text=user_prompt, token_ids=inputs)

    expand_tree(
        root=root,
        max_depth=max_depth,
        top_k=top_k,
        max_branches=max_branches,
        max_new_tokens=max_new_tokens,
    )

    tree_dict = root.to_dict()
    with open("solution_tree.json", "w") as f:
        json.dump(tree_dict, f, indent=2)

    logger.info("Tree structure saved to solution_tree.json")


if __name__ == "__main__":
    fire.Fire(main)


# issues: sometimes picking between too similar tokens
