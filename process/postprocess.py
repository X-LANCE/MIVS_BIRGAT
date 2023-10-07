#coding=utf8
import re
from transformers import AutoTokenizer

PLACEHOLDER = '@'


def contain_chinese_char(s):
    return any(map(lambda x: u'\u4e00' <= x <= u'\u9fff', s))


def convert_tokens_to_value(tokens, tokenizer, input_string=None, dataset='aispeech'):
    """ Convert tokens list into a single string. Take care of whitespaces in Chinese and lower/upper cases of English.
    """
    raw_string = tokenizer.convert_tokens_to_string(tokens)
    if dataset == 'aispeech': # only preserve whitespaces between English/Number characters
        recovered_string = re.sub(r"([a-zA-Z0-9])\s([a-zA-Z0-9])", lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), raw_string, count=1)
        while recovered_string != raw_string:
            raw_string = recovered_string
            recovered_string = re.sub(r"([a-zA-Z0-9])\s([a-zA-Z0-9])", lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), raw_string, count=1)
        recovered_string = re.sub(r'\s+', '', recovered_string).replace(PLACEHOLDER, ' ')

        if '[UNK]' in recovered_string: # special chinese char not in the vocabulary
            unk_char = get_unk_char(tokenizer, input_string)
            recovered_string = recovered_string.replace('[UNK]', unk_char)
    else:
        recovered_string = re.sub(r"\s?'s", " 's", raw_string)
        patt_repl = [
            (r'([a-z])\.', lambda m: m.group(1) + ' .'),
            (r'a\s?\.\s?m', 'a.m'), (r'p\s?\.\s?m', 'p.m'), (r'l\s?\.\s?a', 'l.a'),
            (r'b\s?\.\s?c', 'b.c'), (r'd\s?\.\s?c', 'd.c'), (r'p\s?\.\s?s', 'p.s'),
            (r'u\s?\.\s?s', 'u.s'), (r'n\s?\.e\s?\.\s?r\s?\.\s?d', 'n.e.r.d'),
            (r'([0-9])\s?\.\s?([0-9])', lambda m: m.group(1) + '.' + m.group(2)),
            (r'([a-z0-9]),', lambda m: m.group(1) + ' ,')
        ]
        for patt, repl in patt_repl:
            recovered_string = re.sub(patt, repl, recovered_string, flags=re.I)
    return recovered_string


def get_unk_char(tokenizer: AutoTokenizer, inp: str):
    inp = inp.replace(' ', '')
    tokens = tokenizer.tokenize(inp)
    if '[UNK]' in tokens:
        index = tokens.index('[UNK]')
        prefix = tokenizer.convert_tokens_to_string(tokens[:index]).replace(' ', '')
        char = inp[len(prefix)]
        return char
    return '[UNK]'


def flatten_semantic(semantic):
    tuples = []
    for struct in semantic:
        domain = struct['domain']
        for sf in struct['intents']:
            intent = sf['intent']
            if len(sf['slots']) == 0:
                tuples.append([domain, intent])
            for slot in sf['slots']:
                slot_name, slot_value = slot['name'], slot['value']
                tuples.append([domain, intent, slot_name, slot_value])
    return tuples


def reconstruct_semantic_from_quadruples(quadruples, semantic=None):
    semantic = {'semantics': []} if semantic is None else semantic

    def search_domain(d):
        for struct in semantic['semantics'][::-1]: # check in the reverse order
            if struct['domain'] == d:
                return struct
        struct = {'domain': d, 'intents': []}
        semantic['semantics'].append(struct)
        return struct

    def search_intent(struct, i):
        for sf in struct['intents'][::-1]:
            if i == sf['intent']:
                return sf
        sf = {'intent': i, 'slots': []}
        struct['intents'].append(sf)
        return sf

    def search_intent_slot(struct, i, s, v):
        for sf in struct['intents'][::-1]: # check in the reverse order
            if i == sf['intent']:
                for slot in sf['slots']:
                    if slot['name'] == s:
                        # already has the same intent-slot, create a new semantic frame
                        sf = {'intent': i, 'slots': [{'name': s, 'value': v}]}
                        struct['intents'].append(sf)
                        return sf
                sf['slots'].append({'name': s, 'value': v})
                return sf
        sf = {'intent': i, 'slots': [{'name': s, 'value': v}]}
        struct['intents'].append(sf)
        return sf

    for tup in quadruples:
        if len(tup) == 4:
            d, i, s, v = tup
            search_intent_slot(search_domain(d), i, s, v)
        else:
            d, i = tup
            search_intent(search_domain(d), i)
    return semantic