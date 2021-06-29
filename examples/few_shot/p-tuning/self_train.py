import os
from collections import defaultdict, Counter
import json

def ensemble_eprstmt(multi_eaxmple, confidence=1.0):
    if len(multi_eaxmple) == 0:
        return None
    if len(multi_eaxmple) == 1:
        return multi_eaxmple[0]

    all_labels = []
    for example in multi_eaxmple:
        all_labels.append(example['label'])
    label2num = Counter(all_labels)
    ensemble_label, most_num = label2num.most_common(1)[0]
    #threshhold_num = len(multi_eaxmple) / 2

    threshhold_num = confidence * len(multi_eaxmple)

    if most_num >= threshhold_num:
        # gen ensemble_example
        ensemble_example = multi_eaxmple[0]
        ensemble_example['label'] = ensemble_label
        return ensemble_example
    else:
        # skip
        return None

def ensemble(all_unlabdled_examples, ensemble_fn, iter_num, confidence=1.0):
    print("start ensemble unlabedled.josn:{}".format(all_unlabdled_examples))
    all_id2examples = []

    for file in all_unlabdled_examples:
        id2examples = {}
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip().replace('\'', '\"')
                try:
                    example = json.loads(line)
                except:
                    continue
                id = example['id']
                id2examples[id] = example
        all_id2examples.append(id2examples)

    # get_all ids
    all_ids = set()
    for id2examples in all_id2examples:
        ids = id2examples.keys()
        all_ids = all_ids.union(set(ids))

    dirname = os.path.dirname(all_unlabdled_examples[0])
    output_file = os.path.join(dirname, "iter" + str(iter_num) + "_ensemble_unlabeled.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        for id in all_ids:
            multi_eaxmple = []
            for id2examples in all_id2examples:
                if id in id2examples:
                    multi_eaxmple.append(id2examples[id])
                else:
                    continue
            new_example = ensemble_fn(multi_eaxmple, confidence=confidence)
            if new_example:
                f.write(str(new_example) + "\n")
    print("[save ensemble unlabeled_json]:{}".format(output_file))
    return output_file

ensemble_dict = {
    'eprstmt': ensemble_eprstmt,
    'tnews': ensemble_eprstmt,
    'bustm': ensemble_eprstmt,
    'iflytek': ensemble_eprstmt,
    'csldcp': ensemble_eprstmt,
    'csl': ensemble_eprstmt,
    'chid': ensemble_eprstmt,
    'ocnli': ensemble_eprstmt,
    'cluewsc': ensemble_eprstmt
}

if __name__ == "__main__":
    all_unlabdled_examples = [
        "/home/tianxin04/global_data/paddlenlp//examples/few_shot/ptpet_macbert-large-chinese//output/bs8_lr2E-5_pnum8_maxlen512_ptid1/eprstmt/index0_1epoch_1iter_150step_unlabeled.json",
        "/home/tianxin04/global_data/paddlenlp//examples/few_shot/ptpet_macbert-large-chinese//output/bs8_lr2E-5_pnum8_maxlen512_ptid1/eprstmt/index0_1epoch_1iter_4900step_unlabeled.json",
        "/home/tianxin04/global_data/paddlenlp//examples/few_shot/ptpet_macbert-large-chinese//output/bs8_lr2E-5_pnum8_maxlen512_ptid1/eprstmt/index0_6epoch_0iter_40step_unlabeled.json"
    ]
    result = ensemble(all_unlabdled_examples, ensemble_dict["eprstmt"], 0)
    print(result)