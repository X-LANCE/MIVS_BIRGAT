#coding=utf8

MAX_RELATIVE_DISTANCE = 3


RELATIONS = ['padding-relation'] + [f'word-{i:d}-word' for i in range(- MAX_RELATIVE_DISTANCE, MAX_RELATIVE_DISTANCE + 1)] + ['word-left-word', 'word-right-word'] + \
    ['domain-identity-domain', 'intent-identity-intent', 'slot-identity-slot', 'value-identity-value'] + \
    ['domain-has-intent', 'intent-belongsto-domain', 'domain-has-slot', 'slot-belongsto-domain', 'intent-has-slot', 'slot-belongsto-intent'] + \
    ['slot-has-value', 'value-belongsto-slot', 'intent-exactmatch-intent', 'slot-exactmatch-slot']


DATASETS = {
    'plm_dir': 'pretrained_models/',
    'aispeech': {
        'data': 'data/aispeech',
        'ontology': 'data/aispeech/ontology.json',
        'domains': ['地图', '音乐', '天气', '打电话', '车载控制'],
    },
    'topv2': {
        'raw_data': 'data/TOPv2_Dataset',
        'data': 'data/topv2',
        'ontology': 'data/topv2/ontology.json',
        'domains': ['alarm', 'event', 'messaging', 'music', 'navigation', 'reminder', 'timer', 'weather']
    }
}