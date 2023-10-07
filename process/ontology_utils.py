#coding=utf8
import numpy as np
import os, sys, json, random, argparse
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from itertools import combinations, chain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.configs import DATASETS, RELATIONS
from process.dataset_utils import read_data_recursively


def build_ontology_from_dataset(dataset: str, serialize: bool = True):
    """ Build ontology vocabulary for each domain from the list of all samples.
    @return:
        vocab~(dict): key is domain name, e.g., music, weather, music_cross_weather. value is dict of all available intents~(list) and slots~(list).
        {
            'music': {
                'domain': [music],
                'intent': [[music, intent1], [music, intent2], ...],
                'slot': [[music, slot1], [music, slot2], ...],
                'value': [[val1, val2, ...], [val8, val9, ...], ...], # only single domain store this key-value
                'domain_token_id': [[23]],
                'intent_token_id': [[11, 81], [19, 20, 56], ...],
                'slot_token_id': [[39, 110], [123, 215, 4568], [190], ...],
                'value_token_id': [[[638, 219], [349, 1234], [6810, 2954], ...], [[1120, 23154], [41341, 312, 1455], ...], [[954], ...], ...]
                'hierarchy': {'intent1': {'slot2': 21, 'slot8': 1000}, 'intent2': {...}, ...},
                'relation': int matrix of shape, (len(domain) + len(intent) + len(slot)*2) x (len(domain) + len(intent) + len(slot)*2)
            },
            ... other domains
        }
    """
    datas = read_data_recursively(DATASETS[dataset]['data'])
    ontology, hierarchy = dict(), dict()

    # accumulate info from all data
    for ex in datas:
        for struct in ex['semantics']:
            domain = struct['domain'].replace('_5_10', '')
            if domain not in ontology:
                ontology[domain] = {'domain': [domain], 'intent': set(), 'slot': set(), 'value': defaultdict(Counter), 'classifier': Counter()}
            if domain not in hierarchy: hierarchy[domain] = {}

            for sf in struct['intents']:
                intent = sf['intent']
                ontology[domain]['intent'].add(intent)
                if intent not in hierarchy[domain]: hierarchy[domain][intent] = {}

                if len(sf['slots']) == 0: # no slot-value, only domain-intent, add into classifier
                    ontology[domain]['classifier'].update([(domain, intent)])

                for slot in sf['slots']:
                    ontology[domain]['slot'].add(slot['name'])
                    if slot['name'] not in hierarchy[domain][intent]:
                        hierarchy[domain][intent][slot['name']] = 0
                    hierarchy[domain][intent][slot['name']] += 1
                    ontology[domain]['value'][slot['name']].update([slot['value']])

                    # domain-intent-slot-value, not appear in the question span
                    if slot['value'].lower().replace(' ', '') not in ex['input'].lower().replace(' ',  ''):
                        ontology[domain]['classifier'].update([(domain, intent, slot['name'], slot['value'])])

    # preprocess each single domain
    for domain in ontology:
        vocab = ontology[domain]
        vocab['intent'] = list(map(lambda x: [domain, x], sorted(vocab['intent'])))
        vocab['slot'] = list(map(lambda x: [domain, x], sorted(vocab['slot'])))
        vocab['value'] = [sorted(vocab['value'][s].keys(), key=lambda x: - vocab['value'][s][x]) for _, s in vocab['slot']]
        vocab['hierarchy'] = hierarchy[domain]
        vocab['classifier'] = sorted(filter(lambda x: vocab['classifier'][x] >= 5, vocab['classifier'].keys()), key=lambda x: - vocab['classifier'][x])

    # construct cross_domain
    all_domains = sorted(ontology.keys())
    for num in range(1, len(all_domains)):
        for domain_list in combinations(all_domains, num + 1):
            key = '_cross_'.join(domain_list)
            ontology[key] = {}
            ontology[key]['domain'] = list(domain_list)
            ontology[key]['intent'] = sum([ontology[domain]['intent'] for domain in domain_list], [])
            ontology[key]['slot'] = sum([ontology[domain]['slot'] for domain in domain_list], [])
            ontology[key]['classifier'] = sum([ontology[domain]['classifier'] for domain in domain_list], [])
    ontology[''] = {'domain': [], 'intent': [], 'slot': [], 'classifier': []} # null.json

    # construct labels for BIO labeling
    for key in ontology:
        label2id, id2label = {'O': 0}, ['O']
        vocab = ontology[key]
        for d in vocab['domain']:
            for _, intent in filter(lambda x: x[0] == d, vocab['intent']):
                for slot in sorted(hierarchy[d][intent].keys(), key=lambda s: vocab['slot'].index([d, s])):
                    label = '-'.join(['B', d, intent, slot])
                    label2id[label] = len(label2id)
                    id2label.append(label)
                    label = '-'.join(['I', d, intent, slot])
                    label2id[label] = len(label2id)
                    id2label.append(label)
        vocab['label2id'], vocab['id2label'] = label2id, id2label

    # construct relations for RGAT encoder
    dtype = np.int64
    for key in ontology:
        dnum, inum, snum = len(ontology[key]['domain']), len(ontology[key]['intent']), len(ontology[key]['slot'])
        num = dnum + inum + snum * 2 # multply 2 due to possible slot value nodes
        orel = np.full((num, num), RELATIONS.index('padding-relation'), dtype=dtype)

        # self-loop relations
        orel[range(dnum), range(dnum)] = RELATIONS.index('domain-identity-domain')
        orel[range(dnum, dnum + inum), range(dnum, dnum + inum)] = RELATIONS.index('intent-identity-intent')
        orel[range(dnum + inum, dnum + inum + snum), range(dnum + inum, dnum + inum + snum)] = RELATIONS.index('slot-identity-slot')
        orel[range(dnum + inum + snum, num), range(dnum + inum + snum, num)] = RELATIONS.index('value-identity-value')

        # domain-intent relations
        bias = dnum
        for ddx, dname in enumerate(ontology[key]['domain']):
            intent_idxs = [bias + idx for idx, (dn, _) in enumerate(ontology[key]['intent']) if dn == dname]
            orel[ddx, intent_idxs], orel[intent_idxs, ddx] = RELATIONS.index('domain-has-intent'), RELATIONS.index('intent-belongsto-domain')

        # domain-slot relations
        bias = dnum + inum
        for ddx, dname in enumerate(ontology[key]['domain']):
            slot_idxs = [bias + idx for idx, (dn, _) in enumerate(ontology[key]['slot']) if dn == dname]
            orel[ddx, slot_idxs], orel[slot_idxs, ddx] = RELATIONS.index('domain-has-slot'), RELATIONS.index('slot-belongsto-domain')
        
        # intent-slot relations
        intent_bias, slot_bias = dnum, dnum + inum
        for idx, (dname, iname) in enumerate(ontology[key]['intent']):
            slots = list(hierarchy[dname][iname].keys())
            has_slot_ids = [slot_bias + sdx for sdx, (dn, sn) in enumerate(ontology[key]['slot']) if dn == dname and sn in slots]
            orel[intent_bias + idx, has_slot_ids], orel[has_slot_ids, intent_bias + idx] = RELATIONS.index('intent-has-slot'), RELATIONS.index('slot-belongsto-intent')

        # slot-value relations
        slot_bias, value_bias = dnum + inum, dnum + inum + snum
        orel[range(slot_bias, slot_bias + snum), range(value_bias, value_bias + snum)] = RELATIONS.index('slot-has-value')
        orel[range(value_bias, value_bias + snum), range(slot_bias, slot_bias + snum)] = RELATIONS.index('value-belongsto-slot')

        # cross-domain relations
        # bias = dnum
        # intent_cnt = Counter(map(lambda x: x[1], ontology[key]['intent']))
        # for intent in intent_cnt:
        #     if intent_cnt[intent] > 1:
        #         intent_idxs = [bias + idx for idx, tup in enumerate(ontology[key]['intent']) if tup[1] == intent]
        #         dim1, dim2 = list(zip(*list(product(intent_idxs, intent_idxs))))
        #         orel[dim1, dim2] = RELATIONS.index('intent-exactmatch-intent')
        # bias = dnum + inum
        # slot_cnt = Counter(map(lambda x: x[1], ontology[key]['slot']))
        # for slot in slot_cnt:
        #     if slot_cnt[slot] > 1:
        #         slot_idxs = [bias + idx for idx, tup in enumerate(ontology[key]['slot']) if tup[1] == slot]
        #         dim1, dim2 = list(zip(*list(product(slot_idxs, slot_idxs))))
        #         orel[dim1, dim2] = RELATIONS.index('slot-exactmatch-slot')

        ontology[key]['relation'] = orel.tolist()

    if serialize:
        fp = DATASETS[dataset]["ontology"]
        print(f'Write ontology information to json file: {fp}')
        json.dump(ontology, open(fp, 'w'), ensure_ascii=False, indent=4)

    return ontology


def preprocess_ontology(ontology: dict, tokenizer: AutoTokenizer):
    """ Preprocess the ontology items for each single domain, tokenize each item and record the length.
    """
    for domain in ontology:
        vocab = ontology[domain]
        if 'cross' in domain or domain == '': continue

        vocab['domain_token_id'] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('domain') + tokenizer.tokenize(d)) for d in vocab['domain']]
        vocab['intent_token_id'] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('intent') + tokenizer.tokenize(i)) for _, i in vocab['intent']]
        vocab['slot_token_id'] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('slot') + tokenizer.tokenize(s)) for _, s in vocab['slot']]
        vocab['domain_token_len'] = [len(toks) for toks in vocab['domain_token_id']]
        vocab['intent_token_len'] = [len(toks) for toks in vocab['intent_token_id']]
        vocab['slot_token_len'] = [len(toks) for toks in vocab['slot_token_id']]
        # following the order of slots, [ [val1_ids, val2_ids, ...] , [] , []]
        vocab['value_token_id'] = [[tokenizer.convert_tokens_to_ids(tokenizer.tokenize('value') + tokenizer.tokenize(val)) for val in slot_vals] for slot_vals in vocab['value']]
    return ontology


def get_ontology_plm_position_ids(vocab, value_nums=[], value_token_lens=[], shuffle=True):
    # randomly shuffle the position ids of intents and slots to avoid over-fitting the orders
    start = vocab['domain_token_len'][0] + 1
    domain_position_id = list(range(1, start))

    intent_token_lens, intent_num = vocab['intent_token_len'], len(vocab['intent'])
    intent_position_id = [None] * intent_num # to be filled
    intent_idxs = list(range(intent_num))
    if shuffle:
        random.shuffle(intent_idxs)
    for idx in intent_idxs:
        intent_position_id[idx] = list(range(start, start + intent_token_lens[idx]))
        start += intent_token_lens[idx]

    slot_token_lens, slot_num = vocab['slot_token_len'], len(vocab['slot'])
    slot_position_id, value_position_id = [[] for _ in range(slot_num)], [[] for _ in range(slot_num)] # to be filled
    slot_idxs = list(range(slot_num))
    if shuffle:
        random.shuffle(slot_idxs)
    for idx in slot_idxs:
        slot_position_id[idx] = list(range(start, start + slot_token_lens[idx]))
        start += slot_token_lens[idx]
        if len(value_nums) > 0:
            tmp_value_position_id = []
            for vid in range(value_nums[idx]):
                tmp_value_position_id.extend(list(range(start, start + value_token_lens[idx][vid])))
                start += value_token_lens[idx][vid]
            value_position_id[idx] = tmp_value_position_id

    position_id = [0] + domain_position_id + list(chain.from_iterable(intent_position_id)) + list(chain.from_iterable(slot_position_id)) + \
        list(chain.from_iterable(value_position_id)) + [start]
    return position_id



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['aispeech', 'topv2'], default='aispeech', help='dataset name')
    args = parser.parse_args(sys.argv[1:])

    build_ontology_from_dataset(args.dataset, True)