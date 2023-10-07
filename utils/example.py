#coding=utf8
import torch
from eval.evaluator import Evaluator
from process.transition_system import TransitionSystem
from process.ontology_utils import preprocess_ontology
from process.dataset_utils import read_data_recursively, get_lambda_func_from_files
from utils.configs import DATASETS, RELATIONS, MAX_RELATIVE_DISTANCE
from torch.utils.data import Dataset


class SLUDataset(Dataset):

    def __init__(self, examples) -> None:
        super(SLUDataset, self).__init__()
        self.examples = examples


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int):
        return self.examples[index]


class Example:

    @classmethod
    def configuration(cls, dataset='aispeech', plm=None, ontology_encoding=True, use_value=True,
            init_method='plm', encode_method='rgat', decode_method='lf'):
        cls.dataset, cls.plm = dataset, plm
        cls.tranx = TransitionSystem.get_class_by_dataset(cls.dataset)(plm)
        cls.ontology, cls.tokenizer = cls.tranx.ontology, cls.tranx.tokenizer
        cls.ontology = preprocess_ontology(cls.ontology, cls.tokenizer)
        cls.relations = {key: torch.tensor(cls.ontology[key]['relation'], dtype=torch.long) for key in cls.ontology}
        cls.evaluator = Evaluator(cls.tranx)
        cls.init_method, cls.ontology_encoding, cls.use_value = init_method, ontology_encoding, use_value
        cls.encode_method, cls.decode_method = encode_method, decode_method


    @classmethod
    def load_dataset(cls, data_split='train', files=None, domains=None, DEBUG=False, fine_tuning=False, few_shot=50):
        assert data_split in ['train', 'valid', 'test']
        if DEBUG: data_split = 'train'

        old_data_split = data_split
        if fine_tuning and data_split == 'train': data_split = 'valid'

        read_func, skip_func = get_lambda_func_from_files(files, data_split)
        data = read_data_recursively(DATASETS[cls.dataset]['data'], read_func=read_func, skip_func=skip_func)
        examples, domain = [], '_cross_'.join(sorted(set(domains))) if domains else None # use required domains or default known domain

        for idx, ex in enumerate(data):
            ex = cls(ex, domain, labeled=(data_split != 'test'), id=data_split + str(idx))
            if data_split == 'train' and len(ex.lf_action) <= 120:
                examples.append(ex)
            elif data_split != 'train':
                examples.append(ex)
            if DEBUG and len(examples) >= 1000: break

        if fine_tuning and data_split == 'valid':
            # examples = examples if old_data_split == 'valid' else examples
            examples = examples[few_shot:] if old_data_split == 'valid' else examples[:few_shot]
        return SLUDataset(examples)


    def __init__(self, ex, domain, labeled=True, id='-1') -> None:
        self.ex, self.id = ex, id
        tok = Example.tokenizer

        self.question = ex['input']
        self.question_tok = [tok.cls_token] + tok.tokenize(ex['input']) + [tok.sep_token]
        self.question_id = tok.convert_tokens_to_ids(self.question_tok)
        self.question_len = len(self.question_tok)

        if domain is not None: # check whether the default domains are included
            req_domains = ex['domain'].split('_cross_')
            for d in req_domains: assert d in domain

        if Example.ontology_encoding: # use default domain, pay attention to null.json
            if domain is None: assert ex['domain'] != '', "Need to specify the domain for null.json"
            self.domain = ex['domain'] if domain is None else domain
        else: # use the specified domains
            self.domain = domain

        if Example.encode_method in ['gat', 'rgat']:
            self.question_relation = construct_question_relation(self.question_len)

        self.lf_action, self.sl_action, self.clf_action = [], [], []
        if labeled:
            tranx = Example.tranx
            if Example.decode_method == 'lf': # add boundries for domain/intent/slot
                # self.lf_action = tranx.convert_semantic_to_lf(ex, domain)
                self.lf_action = tranx.convert_semantic_to_bart_lf(ex, domain, ontology_copy=False)
            elif Example.decode_method == 'plf': # domain parallel logical form parsing
                self.lf_action = tranx.convert_semantic_to_plf(ex, domain)
            elif Example.decode_method.startswith('gplm'):
                self.lf_action = tranx.convert_semantic_to_bart_lf(ex, domain, ontology_copy=('copy' in Example.decode_method))
            elif Example.decode_method == 'sl': # sequence labeling method
                self.sl_action = tranx.convert_semantic_to_sl(ex, domain, self.question_tok)
            elif Example.decode_method == 'sl+clf': # sequence labeling + classifier method
                self.sl_action, self.clf_action = tranx.convert_semantic_to_sl_plus_clf(ex, domain, self.question_tok)
            else: raise ValueError('[ERROR]: not recognized decoding method %s' % (Example.decode_method))


def construct_question_relation(question_len):
    if question_len <= MAX_RELATIVE_DISTANCE + 1:
        dist_vec = [RELATIONS.index(f'word-{str(i)}-word') for i in range(- MAX_RELATIVE_DISTANCE, MAX_RELATIVE_DISTANCE + 1, 1)]
        starting = MAX_RELATIVE_DISTANCE
    else:
        dist_vec = [RELATIONS.index('word-left-word')] * (question_len - MAX_RELATIVE_DISTANCE - 1) + \
            [RELATIONS.index(f'word-{str(i)}-word') for i in range(- MAX_RELATIVE_DISTANCE, MAX_RELATIVE_DISTANCE + 1, 1)] + \
                [RELATIONS.index('word-right-word')] * (question_len - MAX_RELATIVE_DISTANCE - 1)
        starting = question_len - 1
    return torch.tensor([dist_vec[starting - i: starting - i + question_len] for i in range(question_len)], dtype=torch.long)
