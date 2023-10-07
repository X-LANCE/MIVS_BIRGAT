#coding=utf8
import torch, random
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from functools import partial
from utils.example import Example
from utils.configs import RELATIONS
from model.model_utils import lens2mask, cached_property
from process.ontology_utils import get_ontology_plm_position_ids


def from_example_list_encoder(ex_list, device='cpu', train=True, **kwargs):
    batch = Batch(ex_list, device)
    # batch.ids = [ex.id for ex in ex_list] # example ids
    encode_method, init_method = Example.encode_method, Example.init_method
    use_value, ontology_encoding = Example.use_value, Example.ontology_encoding
    pad_idx = Example.tokenizer.pad_token_id

    question_lens = [ex.question_len for ex in ex_list]
    batch.question_lens = torch.tensor(question_lens, dtype=torch.long, device=device)
    max_len = batch.question_mask.size(1)
    questions = [ex.question_id + [pad_idx] * (max_len - len(ex.question_id)) for ex in ex_list]
    questions = torch.tensor(questions, dtype=torch.long, device=device)
    batch.copy_ids = questions # for copy-generator network in the decoder

    ontology = Example.ontology
    batch.select_domains = [ex.domain for ex in ex_list]
    domains = sorted(set(batch.select_domains)) # all domains needed in the current batch, including cross-domains
    single_domains = sorted(set(sum([d.split('_cross_') for d in domains], [])))

    if init_method in ['plm', 'gplm', 'gplm+copy']: # plm or swv
        batch.question_inputs = {}
        batch.question_inputs["input_ids"] = questions
        batch.question_inputs["attention_mask"] = batch.question_mask.float()
        batch.question_inputs["position_ids"] = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(questions)
        if ontology_encoding: # need to encode ontology items with text descriptions
            batch.ontology_inputs, tokenizer = {}, Example.tokenizer
            ontology_inputs, ontology_token_lens, value_nums, value_token_lens, domain2bias, bias = [], [], defaultdict(list), defaultdict(list), {}, 0
            for d in single_domains:
                ontology_toks, domain2bias[d] = [], bias
                ontology_toks.extend(ontology[d]['domain_token_id'] + ontology[d]['intent_token_id'] + ontology[d]['slot_token_id'])
                if use_value:
                    for idx in range(len(ontology[d]['slot'])):
                        sample_size = min([2, len(ontology[d]['value_token_id'][idx])])
                        value_tokens = random.sample(ontology[d]['value_token_id'][idx], k=sample_size) if train else ontology[d]['value_token_id'][idx][:sample_size]
                        ontology_toks.extend(value_tokens)
                        value_nums[d].append(sample_size)
                        value_token_lens[d].append([len(toks) for toks in value_tokens])
                bias += len(ontology[d]['domain']) + len(ontology[d]['intent']) + len(ontology[d]['slot']) + sum(value_nums[d])
                # tokens list -> flattened input sequence
                # [CLS] [domain] music [intent] play music [intent] ... [slot] singer [slot] song name ... [value] jay chou [value] ... [SEP]
                ontology_inputs.append([tokenizer.cls_token_id] + sum(ontology_toks, []) + [tokenizer.sep_token_id])
                ontology_token_lens.extend([len(toks) for toks in ontology_toks])
            batch.ontology_token_lens = torch.tensor(ontology_token_lens, dtype=torch.long, device=device)
            max_ont_len = max([len(ont_tok) for ont_tok in ontology_inputs])
            batch.ontology_inputs["input_ids"] = torch.tensor([ont_tok + [pad_idx] * (max_ont_len - len(ont_tok)) for ont_tok in ontology_inputs], dtype=torch.long, device=device)
            batch.ontology_inputs["attention_mask"] = (batch.ontology_inputs["input_ids"] != pad_idx).float()
            batch.ontology_inputs["position_ids"] = torch.tensor([get_ontology_plm_position_ids(ontology[d], value_nums[d], value_token_lens[d], shuffle=train) + [0] * (max_ont_len - len(ontology_inputs[idx])) for idx, d in enumerate(single_domains)], dtype=torch.long, device=device)
            batch.ontology_plm_mask = torch.tensor([[0] + [1] * (len(ont_tok) - 2) + [0] * (max_ont_len + 1 - len(ont_tok)) for ont_tok in ontology_inputs], dtype=torch.bool, device=device)
    else:
        batch.question_inputs = questions
        if ontology_encoding:
            ontology_inputs, value_nums, domain2bias, bias = [], defaultdict(list), {}, 0
            for d in single_domains:
                domain2bias[d] = bias
                ontology_inputs.extend(ontology[d]['domain_token_id'] + ontology[d]['intent_token_id'] + ontology[d]['slot_token_id'])
                if use_value:
                    for idx in range(len(ontology[d]['slot'])):
                        sample_size = min([(5 if train else 5), len(ontology[d]['value_token_id'][idx])])
                        value_tokens = random.sample(ontology[d]['value_token_id'][idx], k=sample_size) if train else ontology[d]['value_token_id'][idx][:sample_size]
                        ontology_inputs.extend(value_tokens)
                        value_nums[d].append(sample_size)
                bias += len(ontology[d]['domain']) + len(ontology[d]['intent']) + len(ontology[d]['slot']) + sum(value_nums[d])
            batch.ontology_token_lens = torch.tensor([len(toks) for toks in ontology_inputs], dtype=torch.long, device=device)
            max_tok_len = batch.ontology_token_mask.size(1)
            batch.ontology_inputs = torch.tensor([ont_tok + [pad_idx] * (max_tok_len - len(ont_tok)) for ont_tok in ontology_inputs], dtype=torch.long, device=device)

    batch.ontology_lens = torch.tensor([len(ontology[d]['domain']) + len(ontology[d]['intent']) + len(ontology[d]['slot']) for d in batch.select_domains], dtype=torch.long, device=device)
    batch.ontology_full_lens = torch.tensor([len(ontology[d]['domain']) + len(ontology[d]['intent']) + (1 + int(use_value)) * len(ontology[d]['slot']) for d in batch.select_domains], dtype=torch.long, device=device)
    # re-organize ontology items for each cross_domain, all domains first, followed by all intents, then all slots and all values
    # -> domain1 domain2 ... intent1 intent2 ... slot1 slot2 ... val1 val2 val3 ...
    if use_value: # firstly, aggregate multiple slot values into one single node
        value_indexes = []
        for d in single_domains:
            bias = domain2bias[d] + len(ontology[d]['domain']) + len(ontology[d]['intent']) + len(ontology[d]['slot'])
            value_indexes.extend(list(range(bias, bias + sum(value_nums[d]))))
        batch.value_indexes = torch.tensor(value_indexes, dtype=torch.long, device=device) # used to select and aggregate slot value nodes
        batch.value_lens = torch.tensor(sum([value_nums[d] for d in single_domains], []), dtype=torch.long, device=device)

    batch.select_ontology_indexes, batch.select_value_indexes = OrderedDict(), OrderedDict()
    for cross_domain in domains:
        d_list = sorted(cross_domain.split('_cross_'))
        select_indexes = {'domain': [], 'intent': [], 'slot': [], 'value': []}
        for d in d_list:
            bias = domain2bias[d] if ontology_encoding else sum([len(ontology[single_domains[idx]]['domain']) + len(ontology[single_domains[idx]]['intent']) + len(ontology[single_domains[idx]]['slot']) for idx in range(single_domains.index(d))])
            select_indexes['domain'].append(bias)
            bias += len(ontology[d]['domain'])
            select_indexes['intent'].extend(list(range(bias, bias + len(ontology[d]['intent']))))
            bias += len(ontology[d]['intent'])
            select_indexes['slot'].extend(list(range(bias, bias + len(ontology[d]['slot']))))
            if use_value:
                bias = sum([len(ontology[single_domains[idx]]['slot']) for idx in range(single_domains.index(d))])
                select_indexes['value'].extend(list(range(bias, bias + len(ontology[d]['slot']))))
        batch.select_ontology_indexes[cross_domain] = torch.tensor(select_indexes['domain'] + select_indexes['intent'] + select_indexes['slot'], dtype=torch.long, device=device)
        if use_value: # notice that select_value_indexes are not from the same tensor as select_ontology_indexes
            batch.select_value_indexes[cross_domain] = torch.tensor(select_indexes['value'], dtype=torch.long, device=device)

    # relations construction
    if encode_method in ['rgat', 'gat']:
        pad_rel_idx, ontology_relations = RELATIONS.index('padding-relation'), Example.relations
        max_len = batch.question_mask.size(1)
        question_relations = [F.pad(ex.question_relation, (0, max_len - ex.question_relation.size(0), 0, max_len - ex.question_relation.size(0)), value=pad_rel_idx) for ex in ex_list]
        max_len = batch.ontology_full_mask.size(1)
        lens = batch.ontology_full_lens.tolist()
        ontology_relations = [F.pad(ontology_relations[d][:lens[idx], :lens[idx]], (0, max_len - lens[idx], 0, max_len - lens[idx]), value=pad_rel_idx) for idx, d in enumerate(batch.select_domains)]
        batch.question_relations = torch.stack(question_relations, dim=0)
        batch.ontology_relations = torch.stack(ontology_relations, dim=0)
        if encode_method == 'rgat':
            batch.question_relations, batch.ontology_relations = batch.question_relations.to(device), batch.ontology_relations.to(device)
            batch.question_relations_mask = (batch.question_relations == pad_rel_idx)
            batch.ontology_relations_mask = (batch.ontology_relations == pad_rel_idx)
        else:
            batch.question_relations_mask = (batch.question_relations == pad_rel_idx).to(device)
            batch.ontology_relations_mask = (batch.ontology_relations == pad_rel_idx).to(device)
    return batch


def from_example_list_decoder(ex_list, batch, device='cpu', train=True, **kwargs):
    decode_method, ontology = Example.decode_method, Example.ontology
    domains = set(batch.select_domains)

    if decode_method == 'plf': # parallel logical form decoding
        domain_lens = torch.tensor([len(d.split('_cross_')) for d in batch.select_domains], dtype=torch.long)
        batch.domain_mask = lens2mask(domain_lens, max_len=batch.ontology_mask.size(1)).to(device)
        batch.use_domain_classifier = True

    if decode_method.startswith('sl'): # sequence labeling
        label_dict = {}
        for d in domains:
            dis_tuple = [label.split('-')[1:] for label in ontology[d]['id2label'][1:]] # ignore the first index for O label
            domain_ids = torch.tensor([ontology[d]['domain'].index(d) for d, _, _ in dis_tuple], dtype=torch.long)
            intent_ids = torch.tensor([len(ontology[d]['domain']) + ontology[d]['intent'].index([d, i]) for d, i, _ in dis_tuple], dtype=torch.long)
            slot_ids = torch.tensor([len(ontology[d]['domain']) + len(ontology[d]['intent']) + ontology[d]['slot'].index([d, s]) for d, _, s in dis_tuple], dtype=torch.long)
            label_dict[d] = torch.stack([domain_ids, intent_ids, slot_ids], dim=1).contiguous().view(-1).to(device) # label_num * 3
        batch.label_dict = label_dict
        batch.label_lens = torch.tensor([len(ontology[d]['id2label']) for d in batch.select_domains], dtype=torch.long, device=device)

    if train:
        pad_id = Example.tokenizer.pad_token_id if decode_method in ['lf', 'plf'] else -100
        if decode_method == 'lf':
            batch.tgt_lens = torch.tensor([len(ex.lf_action) - 1 for ex in ex_list], dtype=torch.long, device=device)
            max_len = batch.tgt_mask.size(1) + 1
            batch.lf_tgt_actions = torch.tensor([ex.lf_action + [pad_id] * (max_len - len(ex.lf_action)) for ex in ex_list], dtype=torch.long, device=device)
        elif decode_method == 'plf':
            plf_tgt_actions = [ex.lf_action[d] for ex in ex_list for d in ontology[ex.domain]['domain'] if d in ex.lf_action]
            batch.tgt_lens = torch.tensor([len(actions) - 1 for actions in plf_tgt_actions], dtype=torch.long, device=device)
            max_len = batch.tgt_mask.size(1) + 1
            batch.lf_tgt_actions = torch.tensor([actions + [pad_id] * (max_len - len(actions)) for actions in plf_tgt_actions], dtype=torch.long, device=device)
            batch.repeat_indexes = torch.tensor([eid for eid, ex in enumerate(ex_list) for _ in range(len(ex.lf_action))], dtype=torch.long, device=device)
            batch.domain_label = torch.tensor([.85 if d in ex.lf_action else .15 for ex in ex_list for d in ontology[ex.domain]['domain']], dtype=torch.float, device=device)
        elif decode_method.startswith('gplm'):
            batch.tgt_lens = torch.tensor([len(ex.lf_action) for ex in ex_list], dtype=torch.long, device=device)
            max_len = batch.tgt_mask.size(1)
            batch.lf_tgt_actions = torch.tensor([ex.lf_action + [pad_id] * (max_len - len(ex.lf_action)) for ex in ex_list], dtype=torch.long, device=device)
        elif decode_method == 'sl':
            max_len = batch.question_mask.size(1)
            batch.sl_tgt_actions = torch.tensor([ex.sl_action + [pad_id] * (max_len - len(ex.sl_action)) for ex in ex_list], dtype=torch.long, device=device)
        elif decode_method == 'sl+clf':
            max_len = batch.question_mask.size(1)
            batch.sl_tgt_actions = torch.tensor([ex.sl_action + [pad_id] * (max_len - len(ex.sl_action)) for ex in ex_list], dtype=torch.long, device=device)
            batch.clf_tgt_actions = torch.tensor([ex.clf_action for ex in ex_list], dtype=torch.float, device=device)
        else: raise NotImplementedError
    else:
        if decode_method in ['lf', 'plf']:
            batch.indexes = {d: (len(ontology[d]['domain']), len(ontology[d]['intent']), len(ontology[d]['slot'])) for d in domains}
        batch.max_action_num = 200
    return batch


class Batch():

    def __init__(self, examples, device='cpu'):
        super(Batch, self).__init__()
        self.examples = examples
        self.device = device


    @classmethod
    def from_example_list(cls, ex_list, device='cpu', train=True, **kwargs):
        batch = from_example_list_encoder(ex_list, device, train, **kwargs)
        batch = from_example_list_decoder(ex_list, batch, device, train, **kwargs)
        return batch


    @classmethod
    def get_collate_fn(cls, **kwargs):
        return partial(cls.from_example_list, **kwargs)


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        return self.examples[idx]


    @cached_property
    def question_mask(self):
        return lens2mask(self.question_lens)


    @cached_property
    def ontology_token_mask(self):
        return lens2mask(self.ontology_token_lens)


    @cached_property
    def ontology_mask(self):
        return lens2mask(self.ontology_lens)


    @cached_property
    def ontology_full_mask(self):
        return lens2mask(self.ontology_full_lens)


    @cached_property
    def value_mask(self):
        return lens2mask(self.value_lens)


    @cached_property
    def label_mask(self):
        return lens2mask(self.label_lens)


    @cached_property
    def tgt_mask(self):
        return lens2mask(self.tgt_lens)
