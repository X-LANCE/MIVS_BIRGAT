#coding=utf8
import os, sys, re, json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.configs import DATASETS


def save_jsonl(data, fp):
    with open(fp, 'w') as of:
        for ex in data:
            of.write(json.dumps(ex, ensure_ascii=False) + '\n')
    return


def retrieve_domains_from_filepath(fp):
    domain_str = os.path.basename(os.path.splitext(fp)[0])
    domain_str = domain_str.replace('_multi_5_10', '').replace('_multi', '')
    domains = sorted([d for d in domain_str.split('_cross_') if d != 'null'])
    return '_cross_'.join(domains)


def load_jsonl(fp):
    lines = []
    print(f'Read data from file path: {fp}')
    with open(fp, 'r') as inf:
        for line in inf:
            if line.strip() == '': continue
            ex = json.loads(line.strip())
            if 'domain' not in ex:
                ex['domain'] = retrieve_domains_from_filepath(fp)
            lines.append(ex)
    return lines


def read_data_recursively(directory: str, read_func=lambda fp: True, skip_func=lambda fp: 'multi_5_10' in fp or 'intent_num' in fp):
    files, data = os.listdir(directory), []
    for fp in files:
        if fp.lower() in ['ontology.json', 'readme.txt']: continue
        fp = os.path.join(directory, fp)
        if skip_func(fp): continue
        if os.path.isdir(fp): data.extend(read_data_recursively(fp, read_func, skip_func))
        elif read_func(fp) and os.path.splitext(fp)[1] == '.json': data.extend(load_jsonl(fp))
    return data


def get_lambda_func_from_files(files, data_split='train'):
    """ Construct the read and skip function according to required filenames, ignoring .json suffix.
    Some abbrev. are also defined for ease of use, e.g., all, all_except_null, single_domain, cross_domain
    """
    read_func = lambda fp: data_split in fp and os.path.splitext(os.path.basename(fp))[0] in files
    skip_func = lambda fp: 'multi_5_10' in fp or 'intent_num' in fp
    if 'all_except_null' in files:
        read_func = lambda fp: data_split in fp
        skip_func = lambda fp: 'multi_5_10' in fp or 'null' in fp or 'intent_num' in fp
    elif 'all' in files:
        read_func = lambda fp: data_split in fp
    elif 'cross_domain' in files:
        skip_func = lambda fp: 'multi_5_10' in fp or 'null' in fp or 'cross' not in fp or 'intent_num' in fp
    elif 'single_domain' in files:
        skip_func = lambda fp: 'multi_5_10' in fp or 'null' in fp or 'cross' in fp or 'intent_num' in fp
    return read_func, skip_func


def split_data_by_intent_number(dataset: str = 'aispeech', split_number: int = 3, split_domains: list = [], seed=999):
    split_domains = [d for d in split_domains if d in DATASETS[dataset]['domains']] if split_domains is not None else []
    if len(split_domains) > 0:
        read_func = lambda x: any(d in x for d in split_domains)
    else:
        read_func = lambda x: True
    skip_func = lambda fp: 'multi_5_10' in fp or 'null' in fp or 'cross' in fp or 'intent_num' in fp
    datas = read_data_recursively(DATASETS[dataset]['data'], read_func=read_func, skip_func=skip_func)
    train, test = [], []
    for ex in datas:
        if sum([len(struct['intents']) for struct in ex['semantics']]) <= split_number: train.append(ex)
        else: test.append(ex)

    indexes, split_point = np.arange(len(test)), int(len(test) / 2)
    np.random.seed(seed)
    np.random.shuffle(indexes)
    valid, test = [test[idx] for idx in indexes[:split_point]], [test[idx] for idx in indexes[split_point:]]
    print(f'Training dataset size with intent number <= {split_number:d} is {len(train):d} .')
    print(f'Valid/Test dataset size with intent number > {split_number:d} is {len(valid):d}/{len(test):d} .')

    domain = '_cross_'.join(sorted(set(split_domains))) if len(split_domains) > 0 else 'all'
    for choice in ['train', 'valid', 'test']:
        output_path = os.path.join(DATASETS[dataset]['data'], choice, domain + '_intent_number_' + str(split_number) + '.json')
        datas = train if choice == 'train' else valid if choice == 'valid' else test
        save_jsonl(datas, output_path)
        print(f'Write {choice:s} dataset to path: {output_path:s} .')
    return


def convert_topv2_data_format(in_dir: str = 'data/TOPv2_Dataset', out_dir: str = 'data/topv2'):
    """ Convert the original data format in TOPv2_Dataset into new data format, e.g.,
    domain: 'timer', logical form: [IN:DELETE_TIMER Cancel [SL:AMOUNT all ] current [SL:METHOD_TIMER timers ] . ]
    => {'domain': 'timer', 'intents': [{'intent': 'delete timer', 'slots': [{'name': 'amount', 'value': 'all'}, {'name': 'method timer', 'value': 'timers'}]}]}
    Some transformation details:
    1. for clarity, intent and slot names are tranformed into lowercase text, _ replaced with whitespace
    2. intent may have empty slots, {'intent': xxx, 'slots': []}
    3. recursive or sub- intent is treated as a separate semantic frame {'intent': xxx, 'slots': [xxx]}
    4. special case [SL:TIMER_NAME [IN:GET_TIME [SL:DATE_TIME 40 minute ] ] ], the slot value '40 minute' is used twice both for the inner slot 'DATE_TIME' and outer slot 'TIMER_NAME'
    5. examples which contain slots without slot values are removed from the dataset
    """
    files = os.listdir(in_dir)
    for split in ['train', 'valid', 'test']:
        for category in ['one_domain_data', 'cross_data']:
            split_dir = os.path.join(out_dir, split, category)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

    def filter_empty_slot_value(data):
        for struct in data['semantics']:
            for sf in struct['intents']:
                for slot in sf['slots']:
                    if len(slot['value']) == 0: return False
        return True


    def convert_data_from_filepath(fp):
        dataset = []
        with open(fp, 'r') as inf:
            inf.readline() # skip the first line: header
            for line in inf:
                if line.strip() == '': continue
                domain, _, parse = line.split('\t')
                inputs, semantics = parse_semantics(parse)
                data = {
                    'input': inputs,
                    'semantics': [{
                        'domain': domain.strip(),
                        'intents': semantics
                    }]
                }
                if filter_empty_slot_value(data):
                    dataset.append(data)
        return dataset

    count = 0
    for name in files:
        if not name.endswith('.tsv'): continue
        fp = os.path.join(in_dir, name)
        domain, split = os.path.splitext(name)[0].split('_')
        if split == 'eval': split = 'valid'
        out_fp = os.path.join(out_dir, split, 'one_domain_data', domain + '.json')
        datas = convert_data_from_filepath(fp)
        save_jsonl(datas, out_fp)
        count += len(datas)
    print(f'In total, convert {count:d} samples into new data format .')
    return count


def parse_semantics(parse):
    parse = parse.replace('[', ' [ ').replace(']', ' ] ').strip()
    toks = re.sub(r'\s+', ' ', parse).split(' ')
    inputs = extract_input_from_parse(toks)
    assert toks[0] == '[', f'The logical form to be parsed should startswith "[": {" ".join(toks):s} .'
    semantics, _, _ = parse_intent(toks, 0)
    return inputs, semantics


def parse_intent(toks, idx):
    """ Extract the intent name (starting from toks[idx], toks[idx] == '[') and its direct slots if exists, return the parsed List of semantic frames,
    the continuous value span of the current intent, and the ending position index. If nested intent exists, create a new semantic frame and append to the returned List.
    @return:
        frames: list of semantic frames, each is of the structure {'intent': intent_name, 'slots': [{'name': slot_name, 'value': slot_value}, ...]}
        value_buffer: the continuous span of the current intent, used as slot value in higher-level slots
        idx: ending position idx in toks, toks[idx] should be ']'
    """
    parsed_frames, start_idx = [], idx
    intent = toks[idx + 1][3:].replace('_', ' ').lower()
    frame = {'intent': intent, 'slots': []} # this is the current semantic frame
    parsed_frames.append(frame)
    idx += 2
    while idx < len(toks):
        w = toks[idx]
        if w == '[':
            prompt = toks[idx + 1]
            if prompt.startswith('IN:'):
                sub_frames, _, idx = parse_intent(toks, idx)
                parsed_frames.extend(sub_frames)
            elif prompt.startswith('SL:'):
                slot, sub_frames, idx = parse_slot(toks, idx)
                frame['slots'].append(slot)
                parsed_frames.extend(sub_frames)
            else: raise ValueError('[ERROR]: unexpected situation in parsed sequence, [ must be followed by IN:xxx or SL:xxx !')
        elif w == ']':
            break
        else: pass # ignore direct content in intent, intent name already conveys the semantic
        idx += 1
    value_buffer = list(filter(lambda x: x not in ['[', ']'] and (not x.startswith('IN:')) and (not x.startswith('SL:')), toks[start_idx:idx]))
    return parsed_frames, value_buffer, idx


def parse_slot(toks, idx):
    """ Extract the slot name and its slot value (starting from toks[idx], toks[idx] == '['), as well as nested semantic frame structure if exists.
    If the slot value is recursively defined and organized as a new [IN:xxx [SL:xxx xxx] ] structure,
    use the buffered value of the entire [IN:xxx [SL:xxx xxx] ] sub-structure as the slot value.
    @return:
        slots: list of slot-value pairs, each is of the strucutre {'name': slot_name, 'value': slot_value}
        frames: list of nested semantic frames, optional
        idx: ending position idx in toks, toks[idx] should be ']'
    """
    parsed_frames = []
    slot_name = toks[idx + 1][3:].replace('_', ' ').lower()
    slot = {'name': slot_name, 'value': [], 'pos': -100}
    idx += 2
    while idx < len(toks):
        w = toks[idx]
        if w == '[':
            prompt = toks[idx + 1]
            if prompt.startswith('IN:'):
                if slot['pos'] == -100:
                    slot['pos'] = extract_value_pos(toks, idx)
                _, value_toks, idx = parse_intent(toks, idx)
                # parsed_frames.extend(sub_frames) # ignore the intent that only functions as a slot value
                slot['value'].extend(value_toks) # actually this slot value is used multiple times, both in the current slot and nested semantic frame
            else: raise ValueError('[ERROR]: unexpected situation in parsed sequence, SL:xxx can not be nested !')
        elif w == ']':
            break
        else:
            slot['value'].append(w)
            if slot['pos'] == -100:
                slot['pos'] = extract_value_pos(toks, idx)
        idx += 1
    slot['value'] = ' '.join(slot['value'])
    slot['pos'] = [slot['pos'] + 1, slot['pos'] + 1 + len(slot['value'])]
    return slot, parsed_frames, idx


def extract_input_from_parse(toks):
    toks = list(filter(lambda w: w not in ['[', ']'] and (not w.startswith('IN:')) and (not w.startswith('SL:')), toks))
    return ' '.join(toks)


def extract_value_pos(toks, idx):
    prefix = list(map(lambda w: len(w), filter(lambda w: w not in ['[', ']'] and (not w.startswith('IN:')) and (not w.startswith('SL:')), toks[:idx])))
    whitespaces = len(prefix) - 1
    return sum(prefix) + whitespaces


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_number', type=int, default=3, help='split intent number')
    parser.add_argument('--split_domains', nargs='+', type=str, help='domains that used to re-split all datas by intent number')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    args = parser.parse_args(sys.argv[1:])

    # convert_topv2_data_format(in_dir=DATASETS['topv2']['raw_data'], out_dir=DATASETS['topv2']['data'])
    split_data_by_intent_number('aispeech', args.split_number, args.split_domains, args.seed)
