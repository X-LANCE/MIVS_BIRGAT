#coding=utf8
import argparse
import os, sys, json
from prettytable import PrettyTable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process.dataset_utils import load_jsonl

DOMAIN_ORDERED = False
INTENT_ORDERED = True
SLOT_ORDERED = False

class Engine():

    # HARDNESS = ['easy', 'medium', 'hard', 'extra', 'all']
    HARDNESS = ['<=1', '2', '3', '4', '>=5', 'all']

    def eval_hardness(self, entry):
        count = sum([len(struct['intents']) for struct in entry['semantics']])
        if count <= 1: return '<=1'
        elif count == 2: return '2'
        elif count == 3: return '3'
        elif count == 4: return '4'
        else: return '>=5'
        # semantic = entry['semantics']
        # size = len(semantic)
        # for struct in semantic:
            # size += len(struct['intents'])
            # for sf in struct['intents']:
                # size += len(sf['slots'])
        # if size <= 4: return 'easy'
        # elif size <= 8: return 'medium'
        # elif size <= 12: return 'hard'
        # return 'extra'

    def flatten_semantic_frame(self, semantic):
        semantic = semantic['semantics'] if 'semantics' in semantic else semantic
        flattened = { 'domain': set(), 'intent': set(), 'slot': set(), 'all': set() }
        num_counter = { 'domain_num': len(semantic), 'intent_num': 0, 'slot_num': 0, 'all_num': (0, 0, 0) }
        for did, struct in enumerate(semantic):
            domain = struct['domain']
            if DOMAIN_ORDERED:
                domain = str(did) + domain
            flattened['domain'].add(domain)
            num_counter['intent_num'] += len(struct['intents'])
            for iid, sf in enumerate(struct['intents']):
                intent = sf['intent']
                if INTENT_ORDERED:
                    intent = str(iid) + intent
                flattened['intent'].add(tuple((domain, intent)))
                num_counter['slot_num'] += len(sf['slots'])
                for sid, slot in enumerate(sf['slots']):
                    slot_name = slot['name']
                    if SLOT_ORDERED:
                        slot_name = str(sid) + slot_name
                    flattened['slot'].add(tuple((domain, intent, slot_name)))
                    slot_value = slot['value'].lower().strip() # ignore case information
                    flattened['all'].add(tuple((domain, intent, slot_name, slot_value)))
        flattened = { k: sorted(flattened[k]) for k in flattened }
        num_counter['all_num'] = (num_counter['domain_num'], num_counter['intent_num'], num_counter['slot_num'])
        return flattened, num_counter

    def score(self, p, g):
        p_flt, p_cnt = self.flatten_semantic_frame(p)
        g_flt, g_cnt = self.flatten_semantic_frame(g)
        scores = { k: int(p_flt[k] == g_flt[k] and p_cnt[k + '_num'] == g_cnt[k + '_num']) for k in ['domain', 'intent', 'slot', 'all'] }
        return scores


def evaluate(pred, gold, verbose=True):
    if type(pred) == str:
        pred = load_jsonl(pred)
    if type(gold) == str:
        gold = load_jsonl(gold)
    if len(gold) != len(pred):
        print('[WARNING]: number of golden and predicted results do not equal !')

    engine = Engine()
    hardness = Engine.HARDNESS
    result = {
       h: { k: 0 for k in ['domain', 'intent', 'slot', 'all', 'count'] } for h in hardness
    }
    for p, g in zip(pred, gold):
        h = engine.eval_hardness(g)
        result[h]['count'] += 1
        result['all']['count'] += 1
        try:
            res = engine.score(p['semantics'], g['semantics'])
        except:
            res = { k: 0 for k in ['domain', 'intent', 'slot', 'all'] }

        if verbose: # print erroneous samples
            if res['all'] == 0:
                print('Error: %s' % (h))
                print('Input: %s' % (g['input']))
                print('Gold: %s' % (json.dumps(g['semantics'], ensure_ascii=False)))
                print('Pred: %s\n' % (json.dumps(p['semantics'], ensure_ascii=False)))
        for k in ['domain', 'intent', 'slot', 'all']:
            result[h][k] += res[k]
            result['all'][k] += res[k]

    for h in result:
        result[h]['ratio'] = round(float(result[h]['count']) / result['all']['count'], 4) if result['all']['count'] != 0 else 1.
        for k in ['domain', 'intent', 'slot', 'all']:
            result[h][k + '_acc'] = round(float(result[h][k]) / result[h]['count'], 4) if result[h]['count'] != 0 else 1.
    
    if verbose:
        table = PrettyTable(field_names=['Hardness'] + hardness)
        table.align = 'c'
        table.add_rows([
            ['count'] + [result[h]['count'] for h in hardness],
            ['ratio'] + [result[h]['ratio'] for h in hardness],
            ['domain_acc'] + [result[h]['domain_acc'] for h in hardness],
            ['intent_acc'] + [result[h]['intent_acc'] for h in hardness],
            ['slot_acc'] + [result[h]['slot_acc'] for h in hardness],
            ['all_acc'] + [result[h]['all_acc'] for h in hardness]
        ])
        print(f'\nDomain ordered: {DOMAIN_ORDERED} ; Intent ordered: {INTENT_ORDERED} ; Slot ordered: {SLOT_ORDERED}')
        print(table)
    return result['all']['all_acc']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='path to predicted file')
    parser.add_argument('--gold', type=str, help='path to golden file')
    args = parser.parse_args(sys.argv[1:])

    evaluate(args.pred, args.gold, verbose=True)
