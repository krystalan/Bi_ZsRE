from transformers import AutoTokenizer
from ..util import HyperParams
from typing import List
import typing
import torch
import numpy as np

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: str,
    ground_truth: str,
    device,
    lang: str="en",
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 ground_truth,
                                                                 device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(" " + ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, inp_targets, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(" " + ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, inp_targets, device)
    elif 'baichuan' in model_name.lower():
        target_tok = tok(" " + ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [
            prompt + tok.decode(target_tok[:i])
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for i in range(len(target_tok))
        ]

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, inp_targets, device)


    ret = {
        f"{portability_key}_acc_{lang}": {
            "ans": textual_ans,
            "target": textual_target
        }
    }
    return ret

def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    
    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, truncation=True, max_length=hparams.max_length, return_tensors="pt").to(f"cuda:{device}")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = "".join(target)
        textual_target_ids = tok.encode(textual_target)
        textual_target = tok.decode(textual_target_ids, skip_special_tokens=True)

        # print(textual_ans) # 正常
        # print(textual_target) # 列表

        textual_ans = textual_ans.lower().strip()
        textual_target = textual_target.lower().strip()

        return textual_ans, textual_target

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target, device, locality=False):
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]