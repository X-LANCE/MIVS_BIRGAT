#coding=utf8
import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.configs import DATASETS
from process.transition_system import TransitionSystem
from eval.evaluation import evaluate


class Evaluator():

    def __init__(self, tranx: TransitionSystem) -> None:
        self.tranx, self.ontology = tranx, tranx.ontology
        self.checker = PostChecker(self.ontology)


    def accuracy(self, pred_hyps, dataset, output_path=None):
        golds = [entry.ex for entry in dataset]

        if type(pred_hyps[0]) == list:
            return self.beam_accuracy(pred_hyps, golds)

        return self.evaluate_function(pred_hyps, golds, output_path)


    def beam_accuracy(self, pred_hyps, golds):
        count = 0
        assert len(pred_hyps) == len(golds)
        for hyps, g in zip(pred_hyps, golds):
            count += int(self.evaluate_function(hyps, [g] * len(hyps), None) > 0)
        score = count / float(len(golds))
        return score


    def postprocess(self, hyps, dataset, checker=False, decode_method='lf', top_ranked=True):
        preds = []
        for hyp_list, entry in zip(hyps, dataset):
            tmp_pred = []

            for hyp in hyp_list:
                if decode_method == 'lf':
                    # pred, flag = self.tranx.convert_lf_to_semantic(hyp.action, entry.ex, entry.domain)
                    pred, flag = self.tranx.convert_bart_lf_to_semantic(hyp.action, entry.ex, entry.domain, False)
                elif decode_method == 'sl':
                    pred, flag = self.tranx.convert_sl_to_semantic(hyp.action, entry.ex, entry.domain, input_tokens=entry.question_tok)
                elif decode_method == 'sl+clf':
                    pred, flag = self.tranx.convert_sl_plus_clf_to_semantic(hyp.sl_action, hyp.clf_action, entry.ex, entry.domain, input_tokens=entry.question_tok)
                elif decode_method == 'plf':
                    pred, flag = self.tranx.convert_plf_to_semantic(hyp.action, entry.ex, entry.domain, (self.checker if checker else None))
                elif decode_method.startswith('gplm'):
                    pred, flag = self.tranx.convert_bart_lf_to_semantic(hyp.action, entry.ex, entry.domain, 'copy' in decode_method)
                else: raise ValueError(f'[ERROR]: Unknown decoding method {decode_method}')

                if not top_ranked: # preserve each hypothesis in the beam
                    tmp_pred.append(pred)
                    continue
                if not checker: # if not execution-guided, directly use the top-ranked result
                    preds.append(pred)
                    break
                if flag and self.checker(pred): # if eg-acc, use the top-ranked and correct result
                    preds.append(pred)
                    break
                tmp_pred.append(pred) # temporarily save the prediction
            else:
                if top_ranked:
                    preds.append(tmp_pred[0]) # all hyps in the beam are wrong, directly use top of beam
                else:
                    preds.append(tmp_pred) # add all preds, for beam-level accuracy

        return preds


    def evaluate_function(self, preds, golds, output_path=None):
        if output_path is not None:
            with open(output_path, 'w') as of:
                old_print = sys.stdout
                sys.stdout = of
                score = evaluate(preds, golds, verbose=True)
                sys.stdout = old_print
        else: score = evaluate(preds, golds, verbose=False)
        return score


class PostChecker():

    def __init__(self, ontology) -> None:
        self.ontology = ontology


    def __call__(self, semantics: dict) -> bool:
        try:
            semantics = semantics['semantics'] if 'semantics' in semantics else semantics
            for struct in semantics:
                domain = struct['domain']
                if domain not in self.ontology: return False
                allowed_intents = self.ontology[domain]['hierarchy']
                for sf in struct['intents']:
                    intent = sf['intent']
                    if intent not in allowed_intents: return False
                    allowed_slots = allowed_intents[intent]
                    for slot in sf['slots']:
                        slot_name = slot['name']
                        if slot_name not in allowed_slots: return False
            return True
        except: return False


if __name__ == '__main__':

    from process.dataset_utils import read_data_recursively

    dataset = 'aispeech' # topv2
    ontology = json.load(open(DATASETS[dataset]['ontology'], 'r'))
    checker = PostChecker(ontology)
    dataset = read_data_recursively(DATASETS[dataset]['data'])
    cnt = 0
    for ex in dataset:
        if not checker(ex): cnt += 1
    print(f'{cnt} examples among {len(dataset)} examples failed to pass the semantic check: domain -> intent -> slot')
