import json
import os
from tqdm import tqdm


from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("/YOUR_MODEL_PATH/chinese-llama-plus-7b")

def obtain_f1_and_em(a,b):
    global tokenizer
    
    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em


def my_avg(a):
    return round(sum(a)*100 / float(len(a)), 2)
    
def calculate_metrics(file_root):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)

    reliablilty_f1_list = []
    reliablilty_em_list = []

    generalization_en_f1_list = []
    generalization_en_em_list = []
    generalization_zh_f1_list = []
    generalization_zh_em_list = []

    locality_en_f1_list = []
    locality_en_em_list = []
    locality_zh_f1_list = []
    locality_zh_em_list = []

    portablility_en_f1_list = []
    portablility_en_em_list = []
    portablility_zh_f1_list = []
    portablility_zh_em_list = []

    for item in tqdm(data):
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(item["post"]["rewrite_acc"]["ans"], item["post"]["rewrite_acc"]["target"])
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        generalization_zh_f1, generalization_zh_em = obtain_f1_and_em(item["post"]["rephrase_acc_zh"]["ans"], item["post"]["rephrase_acc_zh"]["target"])
        generalization_en_f1, generalization_en_em = obtain_f1_and_em(item["post"]["rephrase_acc_en"]["ans"], item["post"]["rephrase_acc_en"]["target"])
        generalization_en_f1_list.append(generalization_en_f1)
        generalization_en_em_list.append(generalization_en_em)
        generalization_zh_f1_list.append(generalization_zh_f1)
        generalization_zh_em_list.append(generalization_zh_em)

        locality_en_f1, locality_en_em = obtain_f1_and_em(item["post"]["locality_en"]["neighborhood_output_en"]["ans"], item["pre"]["locality_en"]["neighborhood_output_en"]["ans"])
        locality_zh_f1, locality_zh_em = obtain_f1_and_em(item["post"]["locality_zh"]["neighborhood_output_zh"]["ans"], item["pre"]["locality_zh"]["neighborhood_output_zh"]["ans"])
        locality_en_f1_list.append(locality_en_f1)
        locality_en_em_list.append(locality_en_em)
        locality_zh_f1_list.append(locality_zh_f1)
        locality_zh_em_list.append(locality_zh_em)

        portablility_en_f1, portablility_en_em = obtain_f1_and_em(item["post"]["portability_en"]["one_hop_acc_en"]["ans"], item["post"]["portability_en"]["one_hop_acc_en"]["target"])
        portablility_zh_f1, portablility_zh_em = obtain_f1_and_em(item["post"]["portability_zh"]["one_hop_acc_en"]["ans"], item["post"]["portability_zh"]["one_hop_acc_en"]["target"])
        portablility_en_f1_list.append(portablility_en_f1)
        portablility_en_em_list.append(portablility_en_em)
        portablility_zh_f1_list.append(portablility_zh_f1)
        portablility_zh_em_list.append(portablility_zh_em)

    print("="*20+file_root+"="*20)
    print("F1 score")
    print("reliablilty_f1: %f"%(my_avg(reliablilty_f1_list)))
    print("generalization_en_f1: %f"%my_avg(generalization_en_f1_list))
    print("generalization_zh_f1: %f"%my_avg(generalization_zh_f1_list))
    print("locality_en_f1: %f"%my_avg(locality_en_f1_list))
    print("locality_zh_f1: %f"%my_avg(locality_zh_f1_list))
    print("portablility_en_f1: %f"%my_avg(portablility_en_f1_list))
    print("portablility_zh_f1: %f"%my_avg(portablility_zh_f1_list))
    

    print("EM score")
    print("reliablilty_em: %f"%(my_avg(reliablilty_em_list)))
    print("generalization_en_em: %f"%my_avg(generalization_en_em_list))
    print("generalization_zh_em: %f"%my_avg(generalization_zh_em_list))
    print("locality_en_em: %f"%my_avg(locality_en_em_list))
    print("locality_zh_em: %f"%my_avg(locality_zh_em_list))
    print("portablility_en_em: %f"%my_avg(portablility_en_em_list))
    print("portablility_zh_em: %f"%my_avg(portablility_zh_em_list))


if __name__ == "__main__":
    

    calculate_metrics("results/chinese_llama7b_FT_en_zh_results.json")
    calculate_metrics("results/chinese_llama7b_FT_zh_en_results.json")

    calculate_metrics("results/chinese_llama2_7b_FT_en_zh_results.json")
    calculate_metrics("results/chinese_llama2_7b_FT_zh_en_results.json")

    calculate_metrics("results/baichuan7b_FT_en_zh_results.json")
    calculate_metrics("results/baichuan7b_FT_zh_en_results.json")

    # calculate_metrics("results/baichuan7b_SERAC_en_zh_results.json")
    # calculate_metrics("results/baichuan7b_SERAC_zh_en_results.json")

    # calculate_metrics("results/baichuan7b_MEND_en_zh_results.json")
    # calculate_metrics("results/baichuan7b_MEND_zh_en_results.json")

    # calculate_metrics("results/baichuan7b_KN_en_zh_results.json")
    # calculate_metrics("results/baichuan7b_KN_zh_en_results.json")

    # calculate_metrics("results/baichuan7b_ROME_en_zh_results.json")
    # calculate_metrics("results/baichuan7b_ROME_zh_en_results.json")

    # calculate_metrics("results/baichuan7b_MEMIT_en_zh_results.json")
    # calculate_metrics("results/baichuan7b_MEMIT_zh_en_results.json")
