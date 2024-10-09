# no-reason

**Improving reasoning capabilities of LLMs without an additional fine-tunning.**

The project uses an LLM to expand a tree of possible reasoning paths. Starting from the user's problem, it generates multiple steps at each node, using entropy-based decision making to determine the breath of the branching. It calculates confidence scores for each branch, selecting the most promising ones to explore further. The process continues until reaching a solution or a predefined limit. The resulting reasoning paths are output as a JSON file for analysis.

**Note:** This repository is under active development. Contributions are welcome!

Example of solving a reasoning problem with `Llama-3.2-3B-Instruct`:

```bash
python ./no_reason/main.py
```

![screenshot](https://github.com/user-attachments/assets/e8aa5a85-776b-4abd-adc3-f9f4b5108711)

Every HF-based LLM should be supported out of the box.

## References:
* [Chain-of-Thought Reasoning without Prompting](https://arxiv.org/pdf/2402.10200)

## Contributing
Issues, new ideas, suggestions, and PRs are all welcome!
