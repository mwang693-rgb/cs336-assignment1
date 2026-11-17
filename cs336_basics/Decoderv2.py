import torch
from .Softmax import softmax

@torch.no_grad()
def Decoder(model, sentence, maxtimes, eos_id, temperature=1.0, p=0.8):
    # sentence: (1, prompt_length)，一次生成一条序列
    if sentence.ndim != 2 or sentence.shape[0] != 1 or sentence.shape[1] == 0:
        raise ValueError("sentence 必须是一条非空序列，形状为 (1, prompt_length)")
    if not temperature > 0 or not 0 < p <= 1:
        raise ValueError("temperature 必须大于 0，p 必须在 (0, 1] 内")

    prompt_length = sentence.shape[1]

    for _ in range(maxtimes):
        model_input = sentence[:, -model.context_length:]
        next_logits = model(model_input)[:, -1, :]  # (1, vocab_size)
        probs = softmax(next_logits / temperature, -1)

        sorted_probs, token_ids = probs.sort(dim=-1, descending=True)

        if p < 1:
            cumulative = sorted_probs.cumsum(dim=-1)
            keep_first = torch.ones_like(cumulative[:, :1], dtype=torch.bool)
            keep = torch.cat([keep_first, cumulative[:, :-1] < p], dim=-1)
            sorted_probs = sorted_probs.masked_fill(~keep, 0.0)

        candidate_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_pos = torch.multinomial(candidate_probs, num_samples=1)
        next_id = token_ids.gather(dim=-1, index=sampled_pos)

        if next_id.item() == eos_id:
            break
        sentence = torch.cat([sentence, next_id], dim=1)

    return sentence[:, prompt_length:]


