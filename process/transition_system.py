# coding=utf-8
import os, sys, argparse, time, json
from tqdm import tqdm
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.configs import DATASETS
from process.postprocess import convert_tokens_to_value, flatten_semantic, reconstruct_semantic_from_quadruples


class TransitionSystem(object):

    def __init__(self, plm: str = None):
        super(TransitionSystem, self).__init__()
        plm_path = os.path.join(DATASETS['plm_dir'], plm)
        self.dataset = type(self).dataset
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path, add_prefix_space=True)
        self.ontology = json.load(open(DATASETS[self.dataset]['ontology'], 'r'))


    @classmethod
    def get_class_by_dataset(cls, dataset: str = 'aispeech'):
        if dataset == 'aispeech':
            return AISpeechTransitionSystem
        elif dataset == 'topv2':
            return TOPV2TransitionSystem
        else: raise ValueError(f'[ERROR]: Not recognized dataset name {dataset:s} !')


    def convert_semantic_to_sl(self, *args, **kwargs):
        raise NotImplementedError


    def convert_semantic_to_bart_lf(self, entry: dict, domain: str = None, ontology_copy: bool = True):
        if ontology_copy:
            return self.convert_semantic_to_lf(entry, domain)

        output_toks, semantic = [], entry['semantics']
        for struct in semantic:
            output_toks.extend(['{', struct['domain']])

            for sf in struct['intents']:
                output_toks.extend(['[', sf['intent']])

                for slot in sf['slots']:
                    output_toks.extend(['(', slot['name'], '=', slot['value'].strip(), ')'])

                output_toks.append(']')

            output_toks.append('}')

        output_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(output_toks))) + [self.tokenizer.sep_token_id]
        return output_ids


    def convert_bart_lf_to_semantic(self, output_ids: dict, entry: dict, domain: str = None, ontology_copy: bool = True):
        if ontology_copy:
            return self.convert_lf_to_semantic(output_ids, entry, domain)

        token_string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(output_ids, skip_special_tokens=True))
        semantic, flag = {'semantics': []}, True
        sep = '' if self.dataset == 'aispeech' else ' '
        try:
            token_buffer = []
            for tok in token_string.strip().split(' '):
                if tok == '{':
                    struct = {'domain': '', 'intents': []}
                    semantic['semantics'].append(struct)
                elif tok == '[':
                    sf = {'intent': '', 'slots': []}
                    struct['intents'].append(sf)
                    if len(token_buffer) > 0 and ' '.join(token_buffer).strip(): # the first intent after domain
                        struct['domain'] = sep.join(token_buffer).strip()
                        token_buffer = []
                elif tok == '}':
                    if len(token_buffer) > 0 and ' '.join(token_buffer).strip(): # domain without intent
                        struct['domain'] = sep.join(token_buffer).strip()
                        token_buffer = []
                elif tok == ']':
                    if len(token_buffer) > 0 and ' '.join(token_buffer).strip(): # intent without slot
                        sf['intent'] = sep.join(token_buffer).strip()
                        token_buffer = []
                elif tok == '(':
                    slot = {'name': '', 'value': ''}
                    sf['slots'].append(slot)
                    if len(token_buffer) > 0 and ' '.join(token_buffer).strip(): # the first slot after intent
                        sf['intent'] = sep.join(token_buffer).strip()
                        token_buffer = []
                elif tok == '=': # termination of slot name, starting for slot value
                    slot['name'] = sep.join(token_buffer).strip()
                    token_buffer = []
                elif tok == ')':
                    slot['value'] = convert_tokens_to_value(token_buffer, self.tokenizer, entry['input'], dataset='aispeech') \
                        if self.dataset == 'aispeech' else ' '.join(token_buffer).strip()
                    token_buffer = []
                else: token_buffer.append(tok)
        except:
            semantic = {'semantics': []}
            flag = False
        return semantic, flag


    def convert_sl_to_semantic(self, labels: list, entry: dict, domain: str = None, input_tokens: list = []):
        """ Convert sequence labeling outputs to semantic tree
        """
        domain = entry['domain'] if domain is None else domain
        id2label = self.ontology[domain]['id2label']
        labels = list(map(lambda idx: id2label[idx], labels))
        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(entry['input']) + [self.tokenizer.sep_token] \
            if not input_tokens else input_tokens
        quadruples, value_buffer, domain, intent, slot = [], [], '', '', ''
        assert len(tokens) == len(labels), "Length of output BIO labels and input question tokens should be exactly the same !"
        for w, l in zip(tokens, labels):
            if l == 'O':
                if len(value_buffer) != 0:
                    value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'])
                    quadruples.append((domain, intent, slot, value))
                value_buffer, domain, intent, slot = [], '', '', ''
            elif l.startswith('B-'):
                if len(value_buffer) != 0:
                    value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'])
                    quadruples.append((domain, intent, slot, value))
                    value_buffer = []
                domain, intent, slot = l.split('-')[1:]
                value_buffer.append(w)
            else: # l.startswith('I-')
                cur_domain, cur_intent, cur_slot = l.split('-')[1:]
                if cur_domain != domain or cur_intent != intent or cur_slot != slot:
                    if len(value_buffer) != 0:
                        value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'])
                        quadruples.append((domain, intent, slot, value))
                        value_buffer = []
                    domain, intent, slot = cur_domain, cur_intent, cur_slot
                value_buffer.append(w)
        if len(value_buffer) != 0:
            value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'])
            quadruples.append((domain, intent, slot, value))
        return reconstruct_semantic_from_quadruples(quadruples), True


    def convert_semantic_to_lf(self, entry: dict, domain: str = None):
        """ Compared to quadruples list, add boundaries for different schema items to recover the semantic tree, e.g.,
        [CLS] { domain [ intent ( slot value ) ( slot value ) ] [ intent ( slot value ) ] } { ... } [SEP]
        """
        semantic = entry['semantics']
        domain = entry['domain'] if domain is None else domain
        vocab, ontology_vocab = self.tokenizer.convert_tokens_to_ids, self.ontology[domain]
        dbias, ibias, sbias = self.tokenizer.vocab_size, self.tokenizer.vocab_size + len(ontology_vocab['domain']), self.tokenizer.vocab_size + len(ontology_vocab['domain']) + len(ontology_vocab['intent'])
        lcur, rcur, lbra, rbra, lpar, rpar = vocab('{'), vocab('}'), vocab('['), vocab(']'), vocab('('), vocab(')')
        output_ids = [self.tokenizer.cls_token_id]
        for struct in semantic:
            domain = struct['domain']
            output_ids.extend([lcur, dbias + ontology_vocab['domain'].index(domain)])

            for sf in struct['intents']:
                intent = sf['intent']
                output_ids.extend([lbra, ibias + ontology_vocab['intent'].index([domain, intent])])

                for slot in sf['slots']:
                    slot_name = slot['name']
                    output_ids.extend([lpar, sbias + ontology_vocab['slot'].index([domain, slot_name])])

                    tokens = self.tokenizer.tokenize(slot['value'].strip())
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    output_ids.extend(token_ids)

                    output_ids.append(rpar)

                output_ids.append(rbra)

            output_ids.append(rcur)

        # end of semantic structure
        output_ids.append(self.tokenizer.sep_token_id)
        return output_ids


    def convert_semantic_to_plf(self, entry: dict, domain: str = None):
        """ Domain parallel logical forms, e.g.,
        domain [ intent ( slot value ) ( slot value ) ] [ intent ( slot value ) ] [SEP]
        domain [ intent ( slot value ) ( slot value ) ] [SEP]
        @return:
            output_dict: domain -> output_ids
        """
        semantic = entry['semantics']
        domain = entry['domain'] if domain is None else domain
        vocab, ontology_vocab = self.tokenizer.convert_tokens_to_ids, self.ontology[domain]
        dbias, ibias, sbias = self.tokenizer.vocab_size, self.tokenizer.vocab_size + len(ontology_vocab['domain']), self.tokenizer.vocab_size + len(ontology_vocab['domain']) + len(ontology_vocab['intent'])
        lbra, rbra, lpar, rpar = vocab('['), vocab(']'), vocab('('), vocab(')')
        output_dict = {}
        for struct in semantic:
            domain = struct['domain']
            output_ids = [dbias + ontology_vocab['domain'].index(domain)]
            output_dict[domain] = output_ids

            for sf in struct['intents']:
                intent = sf['intent']
                output_ids.extend([lbra, ibias + ontology_vocab['intent'].index([domain, intent])])

                for slot in sf['slots']:
                    slot_name = slot['name']
                    output_ids.extend([lpar, sbias + ontology_vocab['slot'].index([domain, slot_name])])

                    tokens = self.tokenizer.tokenize(slot['value'])
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    output_ids.extend(token_ids)

                    output_ids.append(rpar)

                output_ids.append(rbra)

            output_ids.append(self.tokenizer.sep_token_id)

        return output_dict


    def convert_lf_to_semantic(self, output_ids: list, entry: dict, domain: str = None):
        cls_id, sep_id, pad_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
        output_ids = list(filter(lambda x: x not in [cls_id, sep_id, pad_id], output_ids))
        if len(output_ids) == 0: return {'semantics': []}, True

        domain = entry['domain'] if domain is None else domain
        ontology_vocab = self.ontology[domain]
        vocab_size, id2word = self.tokenizer.vocab_size, self.tokenizer.convert_ids_to_tokens
        dbias, ibias, sbias = vocab_size, vocab_size + len(ontology_vocab['domain']), vocab_size + len(ontology_vocab['domain']) + len(ontology_vocab['intent'])

        def id_to_token(idx):
            if idx < dbias: return id2word(idx)
            elif idx < ibias: return ontology_vocab['domain'][idx - dbias]
            elif idx < sbias: return ontology_vocab['intent'][idx - ibias][1]
            else: return ontology_vocab['slot'][idx - sbias][1]

        tokens = list(map(id_to_token, output_ids))
        flag, semantic = True, {'semantics': []}
        try:
            idx, value_buffer = 0, []
            while idx < len(tokens):
                if tokens[idx] == '{': # next should be domain
                    idx += 1
                    domain = tokens[idx]
                    struct = {'domain': domain, 'intents': []}
                    semantic['semantics'].append(struct)
                elif tokens[idx] == '[': # next should be intent
                    idx += 1
                    intent = tokens[idx]
                    sf = {'intent': intent, 'slots': []}
                    struct['intents'].append(sf)
                elif tokens[idx] in ['}', ']']: pass
                elif tokens[idx] == '(':
                    idx += 1
                    slot_name = tokens[idx]
                    slot = {'name': slot_name, 'value': ''}
                    sf['slots'].append(slot)
                elif tokens[idx] == ')': # add slot value
                    if len(value_buffer) > 0:
                        value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'], dataset=self.dataset)
                        slot['value'] = value.strip()
                        value_buffer = []
                else: value_buffer.append(tokens[idx])
                idx += 1
        except: flag = False
        return semantic, flag


    def convert_plf_to_semantic(self, output_dict: dict, entry: dict, domain: str = None, checker = None):
        """ Pass an extra parameter checker, class PostChecker, to check the validity of each domain semantics.
        If not specified, directly retrieve the top of beam for each domain. Structure of output_dict:
        {
            domain1: {
                'ids': [[11, 23, 12], [34, 45, 67], ...], list of all hyps, or a single id list [11, 23, 12]
            },
            domain2: ...
        }
        @return:
            semantics: dict
        """
        domain = entry['domain'] if domain is None else domain
        ontology_vocab = self.ontology[domain]
        vocab_size, id2word = self.tokenizer.vocab_size, self.tokenizer.convert_ids_to_tokens
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        dbias, ibias, sbias = vocab_size, vocab_size + len(ontology_vocab['domain']), vocab_size + len(ontology_vocab['domain']) + len(ontology_vocab['intent'])

        def id_to_token(idx):
            if idx < dbias: return id2word(idx)
            elif idx < ibias: return ontology_vocab['domain'][idx - dbias]
            elif idx < sbias: return ontology_vocab['intent'][idx - ibias][1]
            else: return ontology_vocab['slot'][idx - sbias][1]

        def parse_id_list(output_ids, domain):
            output_ids = list(filter(lambda x: x not in [cls_id, sep_id], output_ids))
            if len(output_ids) == 0: return {'domain': domain, 'intents': []}, False
            tokens = list(map(id_to_token, output_ids))
            flag, struct = True, {'domain': domain, 'intents': []}
            try:
                idx, value_buffer = 0, []
                while idx < len(tokens):
                    if tokens[idx] == '[': # next should be intent
                        idx += 1
                        intent = tokens[idx]
                        sf, value_buffer = {'intent': intent, 'slots': []}, []
                        struct['intents'].append(sf)
                    elif tokens[idx] == ']': pass
                    elif tokens[idx] == '(':
                        idx += 1
                        slot_name = tokens[idx]
                        slot, value_buffer = {'name': slot_name, 'value': ''}, []
                        sf['slots'].append(slot)
                    elif tokens[idx] == ')': # add slot value
                        if len(value_buffer) > 0:
                            value = convert_tokens_to_value(value_buffer, self.tokenizer, entry['input'], dataset=self.dataset)
                            slot['value'] = value
                            value_buffer = []
                    else: value_buffer.append(tokens[idx])
                    idx += 1
            except: flag = False
            return struct, flag

        semantics, global_flag = {'semantics': []}, True
        for domain in output_dict:
            output_ids = output_dict[domain]
            if type(output_ids[0]) != int: # list of hyps
                tmp_structs = []
                for hid, hyp in enumerate(output_ids):
                    struct, flag = parse_id_list(hyp.action, domain) # each is SLUHypothesis object
                    if hid == 0 and len(struct['intents']) == 0: # no info about this domain
                        break

                    if checker is None: # directly retrieve the top of beam structure for each domain
                        semantics['semantics'].append(struct)
                        global_flag &= flag
                        break
                    else:
                        if flag and checker([struct]):
                            semantics['semantics'].append(struct)
                            break
                        else: tmp_structs.append(struct)
                else:
                    global_flag = False
                    semantics['semantics'].append(tmp_structs[0])
            else:
                struct, flag = parse_id_list(output_ids, domain)
                semantics['semantics'].append(struct)
                global_flag &= flag
        return semantics, global_flag


    def convert_semantic_to_sl_plus_clf(self, entry: dict, domain: str = None, input_tokens: list = []):
        sl_labels = self.convert_semantic_to_sl(entry, domain, input_tokens)
        used_tuples = flatten_semantic(entry['semantics'])
        domain = entry['domain'] if domain is None else domain
        clf_labels = [1 if tp in used_tuples else 0 for tp in self.ontology[domain]['classifier']]
        return sl_labels, clf_labels


    def convert_sl_plus_clf_to_semantic(self, sl_labels: list, clf_labels: list, entry: dict, domain: str = None, input_tokens: list = []):
        semantics, flag = self.convert_sl_to_semantic(sl_labels, entry, domain, input_tokens)
        domain = entry['domain'] if domain is None else domain
        labels = self.ontology[domain]['classifier']
        quadruples = [l for flag, l in zip(clf_labels, labels) if flag > 0.5]
        old_quadruples = flatten_semantic(semantics['semantics'])
        quadruples = list(filter(lambda x: x not in old_quadruples, quadruples))
        return reconstruct_semantic_from_quadruples(quadruples, semantics), flag


class AISpeechTransitionSystem(TransitionSystem):

    dataset = 'aispeech'

    def convert_semantic_to_sl(self, entry: dict, domain: str = None, input_tokens: list = []):
        """ Convert the structured semantic dict to sequence labeling labels,
        labels are constructed by B/I-domain-intent-slot and O, in total 2 * slot_num + 1 labels. Index 0 is reserved for label O.
        """
        semantic = entry['semantics']
        domain = entry['domain'] if domain is None else domain
        input_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(entry['input']) + [self.tokenizer.sep_token] \
            if not input_tokens else input_tokens
        new_input_tokens = []
        for idx, w in enumerate(input_tokens):
            if w == self.tokenizer.cls_token: new_input_tokens.append('|') # | is PLACEHOLDER
            elif w == self.tokenizer.sep_token: new_input_tokens.append('|')
            elif w == self.tokenizer.unk_token: # unknown chinese char, assume at most one
                prefix = self.tokenizer.convert_tokens_to_string(new_input_tokens[1:idx]).replace(' ', '')
                new_input_tokens.append(entry['input'].replace(' ', '')[len(prefix)])
            elif w.startswith('##'): new_input_tokens.append(w.lstrip('##'))
            else: new_input_tokens.append(w)
        inp = ''.join(new_input_tokens)
        char2word = [idx for idx, w in enumerate(new_input_tokens) for _ in range(len(w))]

        def flatten_semantic(s):
            quadruples = []
            for struct in s:
                domain = struct['domain']
                for sf in struct['intents']:
                    intent = sf['intent']
                    for slot in sf['slots']:
                        slot_name, slot_value = slot['name'], slot['value'].replace(' ', '').lower()
                        start = slot['pos'][0] if 'pos' in slot and slot_value in inp[slot['pos'][0]:] else 0
                        if slot_value in inp[start:]: # only deal with slot values which appear as a continuous span in the input question
                            start_id = inp.index(slot_value, start)
                            start, end = char2word[start_id], char2word[start_id + len(slot_value) - 1] + 1
                            quadruples.append((domain, intent, slot_name, (start, end)))
            return sorted(set(quadruples), key=lambda tp: tp[3])

        quadruples = flatten_semantic(semantic)
        label2id = self.ontology[domain]['label2id']
        o_label = label2id['O']
        labels = [o_label] * len(input_tokens)
        for d, i, s, pos in quadruples:
            start, end = pos
            if labels[start: end].count(o_label) != end - start: continue # already filled with other labels
            b_label, i_label = label2id['-'.join(['B', d, i, s])], label2id['-'.join(['I', d, i, s])]
            labels[start: end] = [b_label] + [i_label] * (end - start - 1)
        return labels


class TOPV2TransitionSystem(TransitionSystem):

    dataset = 'topv2'

    def convert_semantic_to_sl(self, entry: dict, domain: str = None, input_tokens=None):
        """ Convert the structured semantic dict to sequence labeling labels,
        labels are constructed by B/I-domain-intent-slot and O, in total 2 * slot_num + 1 labels. Index 0 is reserved for label O.
        """
        semantic = entry['semantics']
        domain = entry['domain'] if domain is None else domain
        if type(input_tokens) != dict:
            tokens = self.tokenizer(entry['input'], return_offsets_mapping=True)
        tokens, mappings = tokens['input_ids'], tokens['offset_mapping']
        if input_tokens: assert len(input_tokens) == len(tokens)
        char2word = {s + i: idx for idx, (s, e) in enumerate(mappings) for i in range(e - s)}

        def flatten_semantic(s):
            quadruples = []
            for struct in s:
                domain = struct['domain']
                for sf in struct['intents']:
                    intent = sf['intent']
                    for slot in sf['slots']:
                        start_id, end_id = slot['pos']
                        if start_id not in char2word or (end_id - 1) not in char2word: continue
                        start, end = char2word[start_id], char2word[end_id - 1] + 1
                        quadruples.append((domain, intent, slot['name'], (start, end)))
            return sorted(set(quadruples), key=lambda tp: tp[3])

        quadruples = flatten_semantic(semantic)
        label2id = self.ontology[domain]['label2id']
        o_label = label2id['O']
        labels = [o_label] * len(tokens)
        for d, i, s, pos in quadruples:
            start, end = pos
            if labels[start: end].count(o_label) != end - start: continue # already filled with other labels
            b_label, i_label = label2id['-'.join(['B', d, i, s])], label2id['-'.join(['I', d, i, s])]
            labels[start: end] = [b_label] + [i_label] * (end - start - 1)
        return labels


if __name__ == '__main__':

    from eval.evaluation import evaluate
    from process.dataset_utils import read_data_recursively

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['aispeech', 'topv2'], default='aispeech', help='dataset name')
    parser.add_argument('--plm', default='chinese-bert-wwm-ext', help='pre-trained language model name for tokenization')
    parser.add_argument('--decode_method', choices=['lf', 'sl', 'plf', 'sl+clf', 'gplm', 'gplm+copy'], default='lf', help='decode method used to perform transduction')
    args = parser.parse_args(sys.argv[1:])

    tranx = TransitionSystem.get_class_by_dataset(args.dataset)(args.plm)
    dataset = read_data_recursively(DATASETS[args.dataset]['data'])

    def semantic_to_lf_to_semantic(dataset):
        recovered, domain = [], None
        for ex in tqdm(dataset, desc='Logical Form Transduction:', total=len(dataset)):
            output_ids = tranx.convert_semantic_to_lf(ex, domain=domain)
            recovered_label, flag = tranx.convert_lf_to_semantic(output_ids, ex, domain=domain)
            assert flag
            recovered.append(recovered_label)
        return recovered


    def semantic_to_sl_to_semantic(dataset):
        recovered, domain = [], None
        for ex in tqdm(dataset, desc='Sequence Labeling Transduction:', total=len(dataset)):
            labels = tranx.convert_semantic_to_sl(ex, domain=domain)
            recovered_label, flag = tranx.convert_sl_to_semantic(labels, ex, domain=domain)
            assert flag
            recovered.append(recovered_label)
        return recovered


    def semantic_to_plf_to_semantic(dataset):
        recovered, domain = [], None
        from eval.evaluator import PostChecker
        checker = PostChecker(tranx.ontology)
        for ex in tqdm(dataset, desc='Domain Parallel Logical Form Transduction:', total=len(dataset)):
            labels = tranx.convert_semantic_to_plf(ex, domain=domain)
            recovered_label, flag = tranx.convert_plf_to_semantic(labels, ex, domain=domain, checker=checker)
            assert flag
            recovered.append(recovered_label)
        return recovered


    def semantic_to_sl_plus_clf_to_semantic(dataset):
        recovered, domain = [], None
        for ex in tqdm(dataset, desc='Sequence Labeling + Classifier Transduction:', total=len(dataset)):
            sl_labels, clf_labels = tranx.convert_semantic_to_sl_plus_clf(ex, domain=domain)
            recovered_label, flag = tranx.convert_sl_plus_clf_to_semantic(sl_labels, clf_labels, ex, domain=domain)
            assert flag
            recovered.append(recovered_label)
        return recovered


    def semantic_to_bart_lf_to_semantic(dataset, ontology_copy=True):
        recovered, domain = [], None
        assert 'bart' in args.plm
        for ex in tqdm(dataset, desc='BART Logical Form Transduction:', total=len(dataset)):
            output_ids = tranx.convert_semantic_to_bart_lf(ex, domain=domain, ontology_copy=ontology_copy)
            recovered_label, flag = tranx.convert_bart_lf_to_semantic(output_ids, ex, domain=domain, ontology_copy=ontology_copy)
            assert flag
            recovered.append(recovered_label)
        return recovered


    start_time = time.time()
    if args.decode_method == 'lf':
        rcv_dataset = semantic_to_lf_to_semantic(dataset)
    elif args.decode_method == 'sl':
        rcv_dataset = semantic_to_sl_to_semantic(dataset)
    elif args.decode_method == 'plf':
        rcv_dataset = semantic_to_plf_to_semantic(dataset)
    elif args.decode_method == 'sl+clf':
        rcv_dataset = semantic_to_sl_plus_clf_to_semantic(dataset)
    elif args.decode_method in ['gplm', 'gplm+copy']:
        rcv_dataset = semantic_to_bart_lf_to_semantic(dataset, 'copy' in args.decode_method)
    else:
        raise ValueError(f'[ERROR]: Not recognized decode method {args.decode_method} !')


    # def remove_slot_value_not_in_input(ex): # compute the ratio of examples which cannot be successfully recovered from quadruples
    #     new_semantics, semantics, input = [], ex['semantics'], ex['input']
    #     for struct in semantics:
    #         new_sfs = []
    #         for sf in struct['intents']:
    #             new_slots = []
    #             for slot in sf['slots']:
    #                 if slot['value'].lower() in input.lower():
    #                     new_slots.append(slot)
    #             if len(new_slots) > 0:
    #                 sf['slots'] = new_slots
    #                 new_sfs.append(sf)
    #         if len(new_sfs) > 0:
    #             struct['intents'] = new_sfs
    #             new_semantics.append(struct)
    #     ex['semantics'] = new_semantics
    #     return ex

    # dataset = [remove_slot_value_not_in_input(ex) for ex in dataset]

    print(f'Dataset recovered acc: {evaluate(rcv_dataset, dataset, verbose=True):.4f}')
    print(f'Transduction checking costs {time.time() - start_time:.2f}s')
