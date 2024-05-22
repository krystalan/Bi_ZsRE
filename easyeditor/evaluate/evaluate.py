"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality


def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False,
    source_lang: str = "en",
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new_en, target_new_zh, ground_truth = (
        record[x] for x in ["target_new_en", "target_new_zh", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase_en = record["rephrase_prompt_en"] if 'rephrase_prompt_en' in record.keys() else None
    rephrase_zh = record["rephrase_prompt_zh"] if 'rephrase_prompt_zh' in record.keys() else None
    # locality_prompt = record["locality_prompt"] if 'locality_prompt' in record.keys() else None
    # locality_ground_truth = record["locality_ground_truth"] if 'locality_ground_truth' in record.keys() else None

    # one_hop_prompt = record["one_hop_prompt"] if 'one_hop_prompt' in record.keys() else None
    # one_hop_ground_truth = record["one_hop_ground_truth"] if 'one_hop_ground_truth' in record.keys() else None
    # synonym_prompt = record["synonym_prompt"] if 'synonym_prompt' in record.keys() else None
    # synonym_ground_truth = record["synonym_ground_truth"] if 'synonym_ground_truth' in record.keys() else None
    # inverse_relation_prompt = record["inverse_relation_prompt"] if 'inverse_relation_prompt' in record.keys() else None
    # inverse_relation_ground_truth = record["inverse_relation_ground_truth"] if 'inverse_relation_ground_truth' in record.keys() else None

    if source_lang == "en":
        new_fact = f'New Fact: {prompt} {target_new_en}\nPrompt: {prompt}'
    elif source_lang == "zh":
        new_fact = f'New Fact: {prompt} {target_new_zh}\nPrompt: {prompt}'
    else:
        raise NotImplementedError()

    if pre_edit:
        if source_lang == "en":
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_en, prompt)
        else:
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_zh, prompt)
    else:
        if source_lang == "en":
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_en, new_fact)
        else:
            edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new_zh, new_fact)

    ret = {
        f"rewrite_acc": {
            "ans": edit_acc_ans,
            "target": edit_acc_target
        }
    }
    ret['locality_en'] = {}
    ret['locality_zh'] = {}
    ret['portability_en'] = {}
    ret['portability_zh'] = {}

    if rephrase_en is not None:
        rephrase_acc_en_ans, rephrase_acc_en_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new_en, f'New Fact: {prompt} {target_new_en}\nPrompt: {rephrase_en}')
        ret['rephrase_acc_en'] = {
            "ans": rephrase_acc_en_ans,
            "target": rephrase_acc_en_target
        }

    if rephrase_zh is not None:
        rephrase_acc_zh_ans, rephrase_acc_zh_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new_zh, f'New Fact: {prompt} {target_new_zh}\nPrompt: {rephrase_zh}')
        ret['rephrase_acc_zh'] = {
            "ans": rephrase_acc_zh_ans,
            "target": rephrase_acc_zh_target
        }

    if 'locality_en' in record.keys() and any(record['locality_en']):
        for locality_key in record['locality_en'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality_en'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new_en}\nPrompt: {record['locality_en'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality_en'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new_en}\nPrompt: {record['locality_en'][locality_key]['prompt']}", neighborhood=True)
            # if type(pre_neighbor) is not list:
            #     pre_neighbor = [pre_neighbor, ]
            # if type(post_neighbor) is not list:
            #     post_neighbor = [post_neighbor, ]
            # assert len(pre_neighbor) == len(post_neighbor)

            # ret['locality_en'][f'{locality_key}_acc_en'] = np.mean(np.equal(pre_neighbor, post_neighbor))

            ret['locality_en'][f'{locality_key}_acc_en'] = {
                "pre": pre_neighbor,
                "post": post_neighbor
            }

    if 'locality_zh' in record.keys() and any(record['locality_zh']):
        for locality_key in record['locality_zh'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality_zh'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['locality_zh'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality_zh'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['locality_zh'][locality_key]['prompt']}", neighborhood=True)
            # if type(pre_neighbor) is not list:
            #     pre_neighbor = [pre_neighbor, ]
            # if type(post_neighbor) is not list:
            #     post_neighbor = [post_neighbor, ]
            # assert len(pre_neighbor) == len(post_neighbor)

            # ret['locality_zh'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))

            ret['locality_zh'][f'{locality_key}_acc_zh'] = {
                "pre": pre_neighbor,
                "post": post_neighbor
            }

    
    # Form a list of lists of prefixes to test.
    if 'portability_en' in record.keys() and any(record['portability_en']):
        for portability_key in record['portability_en'].keys():
            if pre_edit:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_en'][portability_key]['ground_truth'],
                                              record['portability_en'][portability_key]['prompt'])
            else:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_en'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new_en}\nPrompt: {record['portability_en'][portability_key]['prompt']}")
            ret['portability_en'][f'{portability_key}_acc_en'] = {
                "ans": portability_acc_ans,
                "target": portability_acc_target
            }

    if 'portability_zh' in record.keys() and any(record['portability_zh']):
        for portability_key in record['portability_zh'].keys():
            if pre_edit:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_zh'][portability_key]['ground_truth'],
                                              record['portability_zh'][portability_key]['prompt'])
            else:
                portability_acc_ans, portability_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability_zh'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new_zh}\nPrompt: {record['portability_zh'][portability_key]['prompt']}")
            ret['portability_zh'][f'{portability_key}_acc_zh'] = {
                "ans": portability_acc_ans,
                "target": portability_acc_target
            }
    # if one_hop_prompt is not None:
    #     one_hop_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            one_hop_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {one_hop_prompt}')
    #     ret['one_hop_acc'] = one_hop_acc
    # if synonym_prompt is not None:
    #     synonym_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            synonym_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {synonym_prompt}')
    #     ret['synonym_acc'] = synonym_acc
    # if inverse_relation_prompt is not None:
    #     inverse_relation_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            inverse_relation_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {inverse_relation_prompt}')
    #     ret['inverse_relation_acc'] = inverse_relation_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower() or 'baichuan' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        
        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
        if not isinstance(ans_idss, list):
            ans_idss = [ans_idss]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True)

        if neighborhood:
            return textual_ans
            # return ans.squeeze().detach().cpu().numpy().tolist()

        

        return textual_ans, textual_target
        # return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()        
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

# TODO: Support GPT Evaluation(predict token one by one)
def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    lang: str = "en",
) -> typing.Dict:

    if 't5' in model_name.lower():
        stuff_probs = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                        prompt,
                                                        target_new,
                                                        device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # inp_targets = [
        #     tok.decode(target_tok[i])
        #     for i in range(len(target_tok))
        # ]
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] #erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'baichuan' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] #erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)


    # Structure the restuls as a dictionary.

    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    
    if not test_rephrase:
        ret = {
            f"{key}_acc": {
                "ans": textual_ans,
                "target": textual_target
            }
        }
    else:
        ret = {
            f"{key}_acc_{lang}": {
                "ans": textual_ans,
                "target": textual_target
            }
        }

    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
    lang: str = "en",
) -> typing.Dict:

    if 't5' in model_name.lower():
        locality_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 locality_ground_truth,
                                                                 device,
                                                                 locality=True)
    elif 'gpt' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'llama' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'baichuan' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    
    ret = {
        f"{locality_key}_output_{lang}": {
            "ans": textual_ans,
            "target": textual_target
        }
    }
    return ret


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    source_lang: str = "en",
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new_en, target_new_zh, ground_truth = (
        record[x] for x in ["target_new_en", "target_new_zh", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts_en = record["rephrase_prompt_en"] if 'rephrase_prompt_en' in record.keys() else None
    rephrase_prompts_zh = record["rephrase_prompt_zh"] if 'rephrase_prompt_zh' in record.keys() else None

    # locality_prompts = record["locality_prompt"] if 'locality_prompt' in record.keys() else None
    # locality_ground_truth = record["locality_ground_truth"] if 'locality_ground_truth' in record.keys() else None
    #
    # one_hop_prompt = record["one_hop_prompt"] if 'one_hop_prompt' in record.keys() else None
    # one_hop_ground_truth = record["one_hop_ground_truth"] if 'one_hop_ground_truth' in record.keys() else None
    # synonym_prompt = record["synonym_prompt"] if 'synonym_prompt' in record.keys() else None
    # synonym_ground_truth = record["synonym_ground_truth"] if 'synonym_ground_truth' in record.keys() else None
    # inverse_relation_prompt = record["inverse_relation_prompt"] if 'inverse_relation_prompt' in record.keys() else None
    # inverse_relation_ground_truth = record["inverse_relation_ground_truth"] if 'inverse_relation_ground_truth' in record.keys() else None

    if source_lang == "en":
        ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rewrite_prompts, target_new_en, device=device, lang="en")
    else:
        ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rewrite_prompts, target_new_zh, device=device, lang="zh")

    ret['locality_en'] = {}
    ret['locality_zh'] = {}
    ret['portability_en'] = {}
    ret['portability_zh'] = {}
    if rephrase_prompts_en is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts_en, target_new_en, device=device, test_rephrase=True, lang="en")
        )

    if rephrase_prompts_zh is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts_zh, target_new_zh, device=device, test_rephrase=True, lang="zh")
        )

    if 'locality_en' in record.keys() and any(record['locality_en']):
        for locality_key in record['locality_en'].keys():
            ret['locality_en'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality_en'][locality_key]['prompt'],
                                         record['locality_en'][locality_key]['ground_truth'], device=device, lang="en")
            )

    if 'locality_zh' in record.keys() and any(record['locality_zh']):
        for locality_key in record['locality_zh'].keys():
            ret['locality_zh'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality_zh'][locality_key]['prompt'],
                                         record['locality_zh'][locality_key]['ground_truth'], device=device, lang="zh")
            )


    if 'portability_en' in record.keys() and any(record['portability_en']):
        for portability_key in record['portability_en'].keys():
            ret['portability_en'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability_en'][portability_key]['prompt'],
                                            record['portability_en'][portability_key]['ground_truth'], device=device)
            )

    if 'portability_zh' in record.keys() and any(record['portability_zh']):
        for portability_key in record['portability_zh'].keys():
            ret['portability_zh'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability_zh'][portability_key]['prompt'],
                                            record['portability_zh'][portability_key]['ground_truth'], device=device)
            )
    # Form a list of lists of prefixes to test.

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

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        # if locality:
        #     return ans
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = tok.decode(target, skip_special_tokens=True)

        return textual_ans, textual_target
        # return np.mean(np.equal(ans, target)), textual_ans, textual_target

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

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


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
