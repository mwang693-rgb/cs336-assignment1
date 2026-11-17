# Steven's CS336 Assignment 1

Independent implementation based on Stanford CS336 Assignment 1.

[Open the training notebook in Colab](https://colab.research.google.com/github/mwang693-rgb/cs336-assignment1/blob/main/notebooks/colab_train.ipynb)

The notebook checks the GPU, prepares TinyStories, runs a 100-step smoke experiment,
and supports a full 327.68M-token baseline. Checkpoints, metrics, and generated text
are saved to Google Drive. Change MODE to full after the smoke run succeeds.

Default model configuration is 22,696,448 parameters, 4 layers, 16 heads,
d_model 512, d_ff 1344, context length 256, vocabulary size 10,000, RoPE,
Pre-RMSNorm, and SwiGLU. Input embeddings and the output head use separate weights.
Local training outcomes are not yet verified in this repository.

The supplied vocab.pkl and merges.pkl contain the existing 10K byte-level BPE tokenizer.
See the retained Stanford handout and original instructions below.

---

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

