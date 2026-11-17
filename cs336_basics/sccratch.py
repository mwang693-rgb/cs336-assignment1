import pickle
from cs336_basics.bpe_training import training_bpe
vocab, merges = training_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("merges.pkl", "wb") as f:
    pickle.dump(merges, f)



