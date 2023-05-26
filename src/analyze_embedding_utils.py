import torch
from transformers import GPT2Model,BertModel,RobertaModel, AutoTokenizer, RobertaTokenizer
import copy 
from typing import DefaultDict
import numpy as np 
from collections import defaultdict

def sub_template_instance_sole(template,names,genders):
   pronoun1={'f':'she','m':'he'}
   pronoun2={'f':'her','m':'his'}
   pronoun3={'f':'her','m':'him'}

   perturbed_instances = [] 

   for j, name in enumerate(names):
      test0=copy.deepcopy(template)
      for i, line in enumerate(template):
          try:
            test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[genders[j]],pronoun2=pronoun2[genders[j]],pronoun3=pronoun3[genders[j]]))
          except:
            print('except entered')
            break
      perturbed_instances.append(test0) 

  
   return perturbed_instances
         

def sub_template_instances(template,Female_instances, Male_instances,Reverse=False):
  """
  Args:
  :param: template: the template to perturb

  Returns: 
  :list of perturbed instances Female and Male
  """
  pronoun1=['he','she']
  pronoun2=['his','her']
  pronoun3=['him','her']

  perturbed_instances_f =[]
  perturbed_instances_m =[]

  for j, name in enumerate(Female_instances):
    test0=copy.deepcopy(template)
    for i, line in enumerate(template):
      try:
        if Reverse:
          test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
        else:
          test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0])) 
      except:
        break
    perturbed_instances_f.append(test0)

  for j,name in enumerate(Male_instances):
    test0=copy.deepcopy(template)
    for i,line in enumerate(template):
      try:
        if Reverse:
          test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[0],pronoun2=pronoun2[0],pronoun3=pronoun3[0]))
        else:
          test0[i]=str(line.substitute(name=name,pronoun1=pronoun1[1],pronoun2=pronoun2[1],pronoun3=pronoun3[1]))
      except:
        break
    perturbed_instances_m.append(test0)      

  return perturbed_instances_f, perturbed_instances_m

def get_encoded_tokens(test0, tokenizer,device): 
    input_context_question=[test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] + tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1], 
                        test0[0] +tokenizer.sep_token + 
                        tokenizer.sep_token + test0[1]]
    input_answers=[test0[2],test0[3],test0[4]]
    encoded_text_train= tokenizer(input_context_question,input_answers,return_tensors='pt',padding=True)
    return encoded_text_train

def get_pretok_info(tokenizer,input):
  """
  : param tokenizer: must be tokenizer(use_fast=True)
  : input : here, perturbed instances
  Returns the tokenizer info with spans (for sub tokenization) 
  """
  return tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(input)

def get_name_idx(instances, name, tokenizer): 
    #print(name)
    subtok_name = f"Ġ{name}"
    if len(instances)>0 : 
        tokenized_words = get_pretok_info(tokenizer, instances[0])
    else:
        tokenized_words = get_pretok_info(tokenizer,instances)
    
    for i in tokenized_words: 
        if(i[0]==subtok_name) or (i[0]==name): 
            #print(i)
            return i[1] # return the slice of index
        
def return_hidden_embeddings (model, input): 
    output = model.model (input_ids=input['input_ids'], attention_mask = input['attention_mask'])
    return output[2] # output hidden states 

def return_sentence_embeddings_layer_concat(template, female_names, male_names, model, tokenizer, device ): 
    f_instances, f_instances_least=sub_template_instances(template,female_names, male_names)

    f_encoded_tokens = [] 
    fl_encoded_tokens = [] 

    for i in np.arange(0,len(f_instances)): 
        f_encoded_tokens.append(get_encoded_tokens(f_instances[i],tokenizer, device))
        fl_encoded_tokens.append(get_encoded_tokens(f_instances_least[i], tokenizer, device ))

    f_embeddings = defaultdict(list)
    fl_embeddings = defaultdict(list)


    for i in np.arange(0,len(f_instances)): 
        femb=return_hidden_embeddings(model, f_encoded_tokens[i])
        flemb = return_hidden_embeddings(model,fl_encoded_tokens[i]) 
        for layer in np.arange(0,13): 
            f_embeddings[layer].append(femb[layer])
            fl_embeddings[layer].append(flemb[layer])
        del femb 
        del flemb 
    
    del f_encoded_tokens 
    del fl_encoded_tokens

    return f_embeddings, fl_embeddings     



def return_embeddings_layer_concat(template, names1, genders1, names2, genders2,model, tokenizer,device):
    f_instances =sub_template_instance_sole(template,names1, genders1) 
    fl_instances=sub_template_instance_sole(template,names2,genders2)

    f_encoded_tokens = [] 
    fl_encoded_tokens = [] 

    for i in np.arange(0,len(f_instances)): 
        f_encoded_tokens.append(get_encoded_tokens(f_instances[i],tokenizer, device))
        fl_encoded_tokens.append(get_encoded_tokens(fl_instances[i], tokenizer, device ))

    f_name_idxs = [] 
    fl_name_idxs = [] 

    for i in np.arange(0,len(f_instances)):
        f_name_idxs.append(get_name_idx(f_instances[i],names1[i],tokenizer))
        fl_name_idxs.append(get_name_idx(fl_instances[i], names2[i],tokenizer))

    f_embeddings = defaultdict(list)
    fl_embeddings = defaultdict(list)


    for i in np.arange(0,len(f_instances)): 
        femb=return_hidden_embeddings(model, f_encoded_tokens[i])
        flemb = return_hidden_embeddings(model,fl_encoded_tokens[i]) 
        for layer in np.arange(0,13): 
            f_embeddings[layer].append(femb[layer][:,f_name_idxs[i][0]:f_name_idxs[i][1],:])
            fl_embeddings[layer].append(flemb[layer][:,fl_name_idxs[i][0]:fl_name_idxs[i][1],:])
        del femb 
        del flemb 
    
    del f_encoded_tokens 
    del fl_encoded_tokens

    return f_embeddings, fl_embeddings 

def match_emb_size(embedding_list):
  ## returns a size-matched embeddings in embedding list
  new_list=[]
  for i, emb in enumerate(embedding_list):
    if len(emb.size())>=3 and emb.size()[1]>1:
      emb=emb.mean(axis=1).unsqueeze(1)

    new_list.append(emb.squeeze(1).reshape(1,-1)) #

  return new_list

def partitionFunc(c,features):
    #:param c : a unit vector
    #:param features
    #:return
  summed = 0 
  for feature in features:
    summed +=np.exp(np.dot(c,feature))
  return summed 

# collect partition function value. It is a statistic to measure the isotropy of embedding space 
# :param : model 
def collectIsotropyMeasure(embeddingList,centralize=False):
  # 1. get sentence of word embeddings in the test dataset 
  #embedding_list=embeddingList.clone()
  # 1-1. match size - if the emb have different sizes
  if type(embeddingList[0]) is torch.Tensor:
    #print(embeddingList[0].size())
    embedding_list=match_emb_size(embeddingList)
    #print(embedding_list[0].size())
    embeddingsNp= torch.cat(embedding_list).cpu().detach().numpy() 
  else: 
    embeddingsNp = np.concatenate(embedding_list)
  
  # 2.0 centralize
  if centralize: 
    embeddingsNp = embeddingsNp - embeddingsNp.mean(0) 

  # 2.1 collect eigenvectors of V: feature matrix 
  square = embeddingsNp.transpose() @ embeddingsNp
  eigenValues, eigenV = np.linalg.eig(square) 

  # 3. for each eigenvector c, calculate Z(c)
  funcValues = [] 
  for v in eigenV: 
    funcValue = partitionFunc(v,embeddingsNp)
    funcValues.append(funcValue)

  # 4. measure = min Z(c) / max Z(c)
  measure = min (funcValues) / max(funcValues)

  return np.round(measure.real,3)

import itertools
def calc_cosine_sim_intra_layer(embeddingList):
  """
  given a layer, calc similarity of embeddings within a layer

  sample size : = length of the list 
  """
  embedding_list = match_emb_size(embeddingList)
  ### get the combination cosine-similarity 
  ind_indices=list(itertools.combinations(range(len(embeddingList)), 2))
  result=[]
  for ind in ind_indices:
    result.append((torch.cosine_similarity(embedding_list[ind[0]],embedding_list[ind[1]])).item())
  
  del embedding_list
  return sum(result)/len(result)

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def calc_CKA_sim_intra_layer(embeddingList):
  """
  given a layer, calc similarity of embeddings within a layer

  sample size : = length of the list 
  """
  embedding_list = match_emb_size(embeddingList)
  ### get the combination cosine-similarity 
  ind_indices=list(itertools.combinations(range(len(embeddingList)), 2))
  result=[]
  for ind in ind_indices:
    result.append(linear_CKA(embedding_list[ind[0]].squeeze(0).unsqueeze(1).cpu().detach().numpy(),embedding_list[ind[1]].squeeze(0).unsqueeze(1).cpu().detach().numpy()))
  
  del embedding_list
  return sum(result)/len(result)
