import json
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag_sents
import re 

nltk.download('averaged_perceptron_tagger')

def getPron(text, predictor, tokenized=False):
    pron_set = []
    pron_p = []  # indexes of the pronouns
    if tokenized:
        output = predictor.predict_tokenized(text)
    else:
        output = predictor.predict(document=text)

    if output['clusters']:
        cluster_len = [len(i) for i in output['clusters']]
        for tup in output['clusters'][cluster_len.index(max(cluster_len))]:
            s = tup[0]
            t = tup[1]
            if s == t:
                pron_p.append((s, s))
            else:
                pron_p.append((s, t + 1))
            pron_set.append(output['document'][s:t + 1])
    else:
        return None, None
    return pron_p, pron_set


def extract_gender_basic(source_dir, out_dir, predictor,resultfile):
    malePron = ['He', 'he', 'him', 'Him', 'his']
    femalePron = ['She', 'she', 'Her', 'her']

    with open(source_dir) as resultfile:
        json_list = list(resultfile)

    with open(out_dir, 'w') as f_out:
        for json_str in json_list:
            result = json.loads(json_str)
            concat_text = f"{result['context']} {result['question']}"
            try:
                pron_indx, pron_set = getPron(concat_text, predictor)
                prons = [" ".join(i) for i in pron_set]
                if bool(set(prons) & set(malePron)) and not bool(set(prons) & set(femalePron)):
                    ### male
                    result['gender'] = 'M'
                elif bool(set(prons) & set(femalePron)) and not bool(set(prons) & set(malePron)):
                    ### female
                    result['gender'] = 'F'
                else:
                    ### unknown
                    result['gender'] = 'UNK'
            except:
                result['gender'] = 'NULL'

            f_out.write(json.dumps(result) + '\n')


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def get_PPN(text):
    output = preprocess(text)
    result = [o for o in output if o[1] in ['NNP', 'NNPS', 'PRP$']]
    return result