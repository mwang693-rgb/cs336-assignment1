import torch
from .Softmax import softmax

@torch.no_grad()
def Decoder(model, sentence, maxtimes, eos_id, k=10):
    prompt_length = sentence.shape[1]
    for _ in range(maxtimes):
        logits = model(sentence[:, -model.context_length:])
        next_logits = logits[:,-1,:]
        sorted_logits, token_ids = next_logits.sort(dim=-1, descending = True)
        candidate_logits = sorted_logits[:, :k]      # 前 k 个分数
        candidate_ids = token_ids[:, :k]              # 前 k 个分数对应的 token ID
        probs = softmax(candidate_logits, -1)
        sampled_pos = torch.multinomial(probs, num_samples=1)
        next_id = candidate_ids.gather(dim=1, index=sampled_pos)             
        if next_id.item() == eos_id:
            break
        sentence = torch.cat([sentence, next_id], dim=1)
    return sentence[:, prompt_length:]



