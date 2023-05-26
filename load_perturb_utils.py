import torch
from torch import nn 
import torch.nn.functional as F 
import json
import sys
import nltk 
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag_sents
import random 
import string 
import copy
from string import Template
import nltk
from nltk.tokenize import wordpunct_tokenize
import numpy 
import numpy as np
import math
import pandas as pd
from scipy.special import rel_entr 
nltk.download('averaged_perceptron_tagger')


def getPron(text,predictor,tokenized=False):
  pron_set=[]
  pron_p=[] # indexes of the pronouns 
  if tokenized:
    output = predictor.predict_tokenized(text)
  else:
    output = predictor.predict(document=text)

  if output['clusters']!=[]:
    cluster_len = [len(i) for i in output['clusters']]
    for tup in output['clusters'][cluster_len.index(max(cluster_len))]:
      s=tup[0]
      t=tup[1]
      if s==t:
        pron_p.append((s,s))
      else:
        pron_p.append((s,t+1))
      pron_set.append(output['document'][s:t+1])
  else:
    return None, None 
  return pron_p,pron_set 

def extract_gender_basic(source_dir,out_dir,predictor):

  malePron = ['He','he','him','Him','his']
  femalePron = ['She','she','Her','her']

  with open(source_dir) as resultfile:
    json_list = list(resultfile)
  
  with open(out_dir,'w') as f_out:
    for json_str in json_list:
      result = json.loads(json_str)
      concat_text=f"{result['context']} {result['question']}"
      try:
        pron_indx, pron_set=getPron(concat_text,predictor)
        prons=[" ".join(i) for i in pron_set]
        if bool(set(prons) & set(malePron)) and not bool(set(prons) & set(femalePron)):
          ### male 
          result['gender']='M'
        elif bool(set(prons)&set(femalePron)) and not bool(set(prons) & set(malePron)):
          ### female 
          result['gender']='F'
        else:
          ### unknown 
          result['gender']='UNK'
      except:
        result['gender']='NULL'
      
      f_out.write(json.dumps(result)+'\n')

def preprocess(sent):
  sent= nltk.word_tokenize(sent)
  sent = nltk.pos_tag(sent)
  return sent

def get_PPN(text):
  output=preprocess(text)
  result=[o for o in output if o[1] in ['NNP','NNPS','PRP$']]
  return result

def get_name_lists(name_gender,sort_by='most',num=200):
  '''Args:
  name_gender = name_gender dictionary : {('Name','gender'):Frequency,}
  freq = 'most' or 'least' if most ==> get the most freq names 
  num = number of name lists to return 
  '''
  f_gender={}
  m_gender={}
  for k, freq in name_gender.items():
    if k[1]=='M':
      m_gender[k[0]]=freq
    else:
      f_gender[k[0]]=freq

  if sort_by=='most':
    f_gender={k: v for k, v in sorted(f_gender.items(), key=lambda item: item[1],reverse=True)}
    m_gender={k: v for k, v in sorted(m_gender.items(), key=lambda item: item[1],reverse=True)}
    f_instances=list(f_gender)[:num]
    m_instances=list(m_gender)[:num]
  elif sort_by=='least':
    f_gender={k: v for k, v in sorted(f_gender.items(), key=lambda item: item[1])}
    m_gender={k: v for k, v in sorted(m_gender.items(), key=lambda item: item[1])}
    f_instances=list(f_gender)[:num]
    m_instances=list(m_gender)[:num]
  elif sort_by=='random':
    random.seed(1)
    f_instances=random.sample(f_gender.keys(),num)
    m_instances=random.sample(m_gender.keys(),num)
  
  return f_instances, m_instances 

def filter_pron_dataset(original_data,predictor):
  """given a test data, filter out only the ones where
  predictor can identify names and gegnders

  returns : dataset and idxs
  """
  dataset=[]
  idxs=[]
  malePron=['he','him','He','Him']
  femalePron=['she','She','her','Her']

  for idx,text in enumerate(original_data):
    concat_text=f"{text[0]} {text[1]}"
    try:
      pron_indx,pron_set=getPron(concat_text,predictor)
      prons=[" ".join(i) for i in pron_set]

      if bool(set(prons) & set(malePron)) or bool(set(prons) & set(femalePron)):
        dataset.append(text)
        idxs.append(idx)
    except: 
      pass

  return dataset, idxs 

def get_template(data,predictor):
  """args:
  :param data: instance of a data
  :param predictor :  

  Returns: 
  : template 
  """

  text_list = [t.lower() for t in data]
  text_list = [wordpunct_tokenize(sen) for sen in text_list] 
  pron_indx,prons = getPron(text_list[0]+text_list[1]+text_list[2]+text_list[3]+text_list[4],predictor,tokenized=True)
  if pron_indx is None: 
    print('No names / pronouns found, try again!')
    return None
  else:
    index = get_list_element_index(text_list,pron_indx)
    text_list_copy = copy.deepcopy(text_list)

    for i, idx in enumerate(index):
      mod = check_modify(prons[i][0])
      text_list_copy[idx[0]][idx[1]]=mod 

    print([" ".join(x) for x in text_list_copy])
    return [Template(" ".join(x)) for x in text_list_copy]

def check_modify(text):
  """Check if text is name, pronoun1, pronoun2, pronoun3 
  """
  pronoun1=['he','she']
  pronoun2=['his','her']
  pronoun3=['him','her']
  
  rtx=""
  if text in pronoun1:
    rtx="$pronoun1"
  elif text in pronoun2:
    rtx="$pronoun2"
  elif text in pronoun3:
    rtx="$pronoun3"
  else:
    rtx="$name"
  
  return rtx


def get_list_element_index(list_of_lists,index_list):
  '''Args:
  list_of_lists: nested lists [['','',,],['',],]
  index_list: flattened element list (tuples)

  returns: list index, element index 
  '''
  lens=[len(t) for t in list_of_lists]
  total=0 
  for i, l in enumerate(lens):
    total+=l
    lens[i]=total 

  new_idx=[]
  for i,idx in enumerate(index_list):
    for j,l in enumerate(lens):
      if j==0 and idx[0]<l:
        new_idx.append((0,idx[0]))
        break
      if idx[0]<l and idx[0]>lens[j-1]:
        #print(j)
        new_idx.append((j,idx[0]-lens[j-1]))
        break
  
  return new_idx

def eval_instances(model,data,device):
  model.to(device)
  model.eval() 
  loss = None 
  with torch.no_grad():
    input_ids=data[0]['input_ids'].unsqueeze(0).to(device)
    attention_mask = data[0]['attention_mask'].unsqueeze(0).to(device)
    labels=torch.tensor(data[1]).unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
    loss,logits = outputs[0],outputs[1]
    return torch.argmax(nn.Softmax(dim=1)(logits),dim=1).tolist() 

def run_prediction(test0,label,tokenizer,model,device):
  input_context_question=[test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] +tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1]]
  input_answers=[test0[2],test0[3],test0[4]]
  encoded_text_train= tokenizer(input_context_question,input_answers,return_tensors='pt',padding=True)
  data= (encoded_text_train,label)
  pred=eval_instances(model,data,device)
  return pred

def perturb_instances(dataset, test_labels, tokenizer,model,device, Female_instances, Male_instances,Reverse=False):
  pronoun1=['he','she']
  pronoun2=['his','her']
  pronoun3=['him','her']

  test_template=[]
  labels=[]
  errors=[]
  error_verboses=[]

  for idx,data in enumerate(dataset): 
    fe_error=0
    me_error=0
    error_verbose=[]
    test_label=test_labels[idx]
    print(f"***{idx}***") 
    print('Female instances:')
    for j,name in enumerate(Female_instances):
      test0=copy.deepcopy(data)
      for i,line in enumerate(data):
          try:
            if Reverse:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
            else:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0]))
          except:
            break
      try:
        pred=run_prediction(test0,test_label,tokenizer,model,device)
        if pred[0]!=test_label:
          print(f"{name}: {pred[0]}")
          error_verbose.append([name, pred[0]])
          fe_error+=1
        else:
          error_verbose.append([name,pred[0]])
      except:
        break

    print("Male instaces:")
    for j,name in enumerate(Male_instances):
      test0=copy.deepcopy(data)
      for i,line in enumerate(data):
          try:
            if Reverse:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0]))
            else:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
          except:
            break
      try:
        pred=run_prediction(test0,test_label,tokenizer,model,device)
        if pred[0]!=test_label:
          print(f"{name}: {pred[0]}")
          error_verbose.append([name, pred[0]])
          me_error+=1
        else:
          error_verbose.append([name, pred[0]])
      except:
        break

    print(test0)
    error_verboses.append(error_verbose)
    error_txt=f"Error rate for female: {fe_error}/200, male: {me_error}/200"
    errors.append(error_txt)
    labels.append(test_label)
    test_template.append([t.template for t in data])
    print("Test statistics: ")
    print(f"Error rate for female: {fe_error}/200, male: {me_error}/200")
  
  return test_template, labels, errors, error_verboses


def perturb_instances_with_ece(dataset, test_labels, tokenizer,model,device, Female_instances, Male_instances,Reverse=False):
  pronoun1=['he','she']
  pronoun2=['his','her']
  pronoun3=['him','her']

  test_template=[]
  labels=[]
  errors=[]
  error_verboses=[]
  all_confs = [] 

  for idx,data in enumerate(dataset): 
    fe_error=0
    me_error=0
    error_verbose=[]
    confs=[]
    test_label=test_labels[idx]
    print(f"***{idx}***") 
    print('Female instances:')
    for j,name in enumerate(Female_instances):
      test0=copy.deepcopy(data)
      for i,line in enumerate(data):
          try:
            if Reverse:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
            else:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0]))
          except:
            break
      try:
        probs=run_prediction_with_ece(test0,test_label,tokenizer,model,device)
        conf, pred=probs.max(-1)
        #print(conf,pred)
        #pred = pred.detach().cpu().numpy()
        #conf = pred.detach().cpu().numpy() 
        #print(conf.detach().cpu().numpy()[0])
        confs.append(conf.detach().cpu().numpy()[0])
        if pred[0]!=test_label:
          print(f"{name}: {pred[0]}")
          error_verbose.append([name, pred.detach().cpu().numpy()[0]])
          fe_error+=1
        else:
          error_verbose.append([name,pred.detach().cpu().numpy()[0]])
      except:
        break

    print("Male instaces:")
    for j,name in enumerate(Male_instances):
      test0=copy.deepcopy(data)
      for i,line in enumerate(data):
          try:
            if Reverse:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0]))
            else:
              test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
          except:
            break
      try:
        probs=run_prediction_with_ece(test0,test_label,tokenizer,model,device)
        conf, pred = probs.max(-1)
        #pred = pred.detach().cpu().numpy()
        #conf = pred.detach().cpu().numpy()
        confs.append(conf.detach().cpu().numpy()[0])
        if pred[0]!=test_label:
          print(f"{name}: {pred[0]}")
          error_verbose.append([name, pred.detach().cpu().numpy()[0]])
          me_error+=1
        else:
          error_verbose.append([name, pred.detach().cpu().numpy()[0]])
      except:
        break

    print(test0)
    error_verboses.append(error_verbose)
    error_txt=f"Error rate for female: {fe_error}/200, male: {me_error}/200"
    errors.append(error_txt)
    labels.append(test_label)
    test_template.append([t.template for t in data])
    all_confs.append(confs)
    print("Test statistics: ")
    print(f"Error rate for female: {fe_error}/200, male: {me_error}/200")
  
  return test_template, labels, errors, error_verboses , all_confs

def eval_instances_with_ece(model,data,device):
  model.to(device) 
  model.eval() 
  loss = None
  with torch.no_grad():
    input_ids=data[0]['input_ids'].unsqueeze(0).to(device)
    attention_mask = data[0]['attention_mask'].unsqueeze(0).to(device)
    labels=torch.tensor(data[1]).unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
    loss,logits = outputs[0],outputs[1]
    return F.softmax(logits, dim=-1) ## prob


def run_prediction_with_ece(test0,label,tokenizer,model,device): 
  input_context_question=[test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] +tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1]]
  input_answers=[test0[2],test0[3],test0[4]]
  encoded_text_train= tokenizer(input_context_question,input_answers,return_tensors='pt',padding=True)
  data= (encoded_text_train,label)
  probs=eval_instances_with_ece(model,data,device)
  return probs

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def calc_agreement_ratio(prediction_matrix):
  """
  :param: prediction_matrix, dataframe 
  """
  N = len(prediction_matrix) # total instances
  agreement_ratio = []
  for i, row in prediction_matrix.iterrows(): 
    n = sum(row)
    k = len(row) # number of predictions
    mean = n/k
    means = [np.round(1/k,2) for t in row]
    std = np.std(row)
    #print(means) 
    try: 
      #emp_prob = [np.round(t/n,2) for t in row]
      #print(emp_prob)
      #dist=rel_entr(emp_prob,means)
      #print(dist)
      #ratio = max(dist)
      ratio = (1/((n-1)*n))*(sum([k*k for k in row])-n)
      #ratio = (1/((n-1)*n))*(((k/n)*sum([t*t for t in row]))-n)
      #ratio  = (sum([(t-mean)**3 for t in row]) / ((k-1)*(std**3))) #skweness 
    except: 
      #print(i)
      #print(row)  # capture zero division error #
      ratio =1 
    
    #print(np.round(ratio,2))
    agreement_ratio.append(ratio)
  
  softmaxed_agreement_ratio= softmax(agreement_ratio)
  prediction_matrix['ratio'] = agreement_ratio
  
  agreement_mean = prediction_matrix['ratio'].mean()
  
  return prediction_matrix, agreement_mean

def get_prediction_matrix(verbose_list):
  """
  :args : verbose_list [List[],List[]]
  """
  from collections import defaultdict
  df = pd.DataFrame() 
  for i, instance in enumerate(verbose_list): # 0~614 # of templates
    counter = defaultdict(int)
    for j, errs in enumerate(instance): # # of perturbations for each template 
      counter[errs[1]]+=1 
    inst_count = pd.Series(counter)
    new_row = inst_count.to_frame().T
    df=pd.concat([df,new_row],ignore_index=True)
    df=df.fillna(0)
  
  return df 



