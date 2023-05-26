

from allennlp.predictors.predictor import Predictor


import argparse 

import pickle


from load_perturb_utils import *
from models import *
from transformers import GPT2Model,BertModel,RobertaModel, AutoTokenizer, RobertaTokenizer


def argument_parser():
  parser = argparse.ArgumentParser() 
  parser.add_argument(
    "--model_name",
    default='roberta',
    type=str,
    required=False, 
    choices = ['roberta','bert','gpt2'],
    help="Path to pre-trained model or shortcut name selected"
  )

  parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=False,
    help="Path to checkpoint directory"
  )
  parser.add_argument(
    "--check_point_epoch",
    default=1, 
    type = int
  )

  parser.add_argument(
    "--test_path",
    type=str,
    required=False
  )

  parser.add_argument(
    "--template_path",
    type=str,
    required=False,
    help='Pickle file that stores template'
  )

  parser.add_argument(
    "--template_label_path",
    type=str,
    required=False,
    help='Pickle file that stores template labels'
  )

  parser.add_argument(
    '--perturb_every',
    action='store_true',
    default=False,
    help='if True, perturb the names by all choices'
  )

  parser.add_argument(
    "--perturb_names_by",
    type=str,
    required=False,
    choices=['most','least','random'],
    help='Names to sort by'
  )

  parser.add_argument(
    '--test_all_epoch',
    action='store_true',
    default=False,
    help='True if to test all epoch'
  )

  parser.add_argument(
    '--get_ece',
    action='store_true',
    default=False,
    help='True if to compute ece'
  )

  parser.add_argument(
    "--save_location",
    type=str
  )

  return parser.parse_args() 

def main():
  args = argument_parser()

  # 1. Get test template 
  if args.template_path is not None and args.template_label_path is not None: 
    # retrieve cache 
    with open(args.template_path ,'rb') as f: 
      template_data = pickle.load(f)
    with open(args.template_label_path, 'rb') as f: 
      template_test_labels = pickle.load(f)
  else: 
    # create template from test data 
    if args.test_path is None: 
      print('Warning: You need to provide test data path')
      return 
    else: 
      json_test=pd.read_json(path_or_buf=args.test_path,lines=True)

      test_data_siqa=[elem for elem in zip(
                      json_test['context'].tolist(),
                      json_test['question'].tolist(),
                      json_test['answerA'].tolist(),
                      json_test['answerB'].tolist(),
                      json_test['answerC'].tolist())]
      test_labels_siqa = []
      with open(args.test_path) as f:
        for line in f:
          lin=json.loads(line.strip())
          label=lin['correct']
          test_labels_siqa.append(ord(label)-65)
      
      predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device=0)
      predictor._model = predictor._model.cuda()

      dataset=[]
      idxs=[]
      malePron=['he','him','He','Him']
      femalePron=['she','She','her','Her']

      for idx,text in enumerate(test_data_siqa):
        concat_text=f"{text[0]} {text[1]}"
        try:
          pron_indx,pron_set=getPron(concat_text,predictor)
          prons=[" ".join(i) for i in pron_set]

          if bool(set(prons) & set(malePron)) or bool(set(prons) & set(femalePron)):
            dataset.append(text)
            idxs.append(idx)
        except: 
          pass

      template_data=[]
      template_labels=[]
      for id,text in enumerate(dataset):

        #print(id)
        #print(text)
        text_list=[t.lower() for t in text]
        text_list=[wordpunct_tokenize(sen) for sen in text_list]

        pron_indx,prons = getPron(text_list[0]+text_list[1]+text_list[2]+text_list[3]+text_list[4],
                                  predictor,tokenized=True)
        if pron_indx is None:
          continue
        else:
          index=get_list_element_index(text_list,pron_indx)

          text_list_copy=copy.deepcopy(text_list)

          for i, idx in enumerate(index):
            #_last_ch=prons[i][0][-1]
            #if _last_ch in string.punctuation:
            #  mod=check_modify(prons[i][0][-1])
            #  mod=mod+_last_ch
            #else:
            mod=check_modify(prons[i][0]) # check and modify the pronoun to template grammer
            text_list_copy[idx[0]][idx[1]]=mod
            

          #template_data.append([Template(" ".join(x)) for x in text_list])
            
          #print([" ".join(x) for x in text_list_copy])
          template_labels.append(idxs[id])
          template_data.append([Template(" ".join(x)) for x in text_list_copy])
      template_test_labels = [test_labels_siqa[i] for i in template_labels]


  # 2. get female, male names to augment
  with open('data/names/name_gender','rb') as f: 
  #with open('/content/gdrive/MyDrive/Comsense_eval/Data/names/name_gender','rb') as f: 
    name_gender = pickle.load(f) # place the name_gender dir 

  # 3. Load model and run perturbation
  DEVICE = torch.device('mps') if torch.has_mps else torch.device('cpu')
  #DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  if args.model_name=='roberta': 
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_base = RobertaModel.from_pretrained('roberta-base')
    model = Multiple_Choice_Model_roberta(model_base)
  elif args.model_name=='bert': 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',use_fast=True)
    model_base = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
    model = Multiple_choice_model_bert(model_base,DEVICE)
  elif args.model_name=='gpt2': 
    tokenizer = AutoTokenizer.from_pretrained('gpt2',use_fast=True)
    model_base = GPT2Model.from_pretrained('gpt2',output_hidden_states=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    setattr(model_base.config,'pad_token_id',tokenizer.pad_token_id)
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    setattr(model_base.config,'sep_token_id',tokenizer.sep_token_id)
    model_base.resize_token_embeddings(len(tokenizer))
    model=Multiple_choice_model_gpt2(model_base,DEVICE)

  

  if args.test_all_epoch:
    for i in np.arange(args.check_point_epoch,11):
      print(f'Epoch:{i}')
      checkpoint =f'checkpoints/socialiqa/model-{args.model_name}-checkpoint-epoch{i}.pt'
      #checkpoint = f'/content/gdrive/MyDrive/Comsense_eval/checkpoints/socialiqa/model-{args.model_name}-checkpoint-epoch{i}.pt'
      restore_dict = torch.load(checkpoint,map_location=DEVICE)
      model.load_state_dict(restore_dict)
      result_df=pd.DataFrame()
      if args.perturb_every:
        perturbs = ['most','least','random']
        for perturb in perturbs: 
          print(f'Processing..{perturb}')
          Female_instances,Male_instances=get_name_lists(name_gender,sort_by=perturb,num=200)
          temp, label, reverse_error, reverse_error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances,Reverse=True)
          temp, label, error, error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances)
          if perturb=='most':
            result_df['Template']=temp
            result_df['label'] = label 
          result_df[f"{perturb}_reverse_error"]=reverse_error 
          result_df[f"{perturb}_reverse_error_verbose"]=reverse_error_verbose 
          result_df[f"{perturb}_error"]=error 
          result_df[f"{perturb}_error_verbose"]=error_verbose 
        with open(f"{args.save_location}/{args.model_name}-epoch{i}",'wb') as f:
          pickle.dump(result_df,f)
      else:
        print(f'Processing..{args.perturb_names_by}')
        Female_instances,Male_instances=get_name_lists(name_gender,sort_by=args.perturb_names_by,num=200)
        temp, label, reverse_error, reverse_error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances,Reverse=True)
        temp, label, error, error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances)
        result_df[f"{args.perturb_names_by}_reverse_error"]=reverse_error
        result_df[f"{args.perturb_names_by}_reverse_error_verbose"]=reverse_error_verbose 
        result_df[f'{args.perturb_names_by}_error']=error 
        result_df[f'{args.perturb_names_by}_error_verbose']=error_verbose 
        with open(f"{args.save_location}/{args.model_name}-epoch{i}-{args.perturb_names_by}",'wb') as f: 
          pickle.dump(result_df,f)
  else: 
    i=0 # for untuned
    if args.checkpoint_dir is not None: 
      restore_dict = torch.load(args.checkpoint_dir,map_location=DEVICE) 
      model.load_state_dict(restore_dict) 
      i = args.check_point_epoch
    result_df=pd.DataFrame()
    if args.perturb_every:
        perturbs = ['most','least','random']
        for perturb in perturbs: 
          print(f'Processing..{perturb}')
          Female_instances,Male_instances=get_name_lists(name_gender,sort_by=perturb,num=200)
          temp, label, reverse_error, reverse_error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances,Reverse=True)
          temp, label, error, error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances)
          if perturb=='most':
            result_df['Template']=temp
            result_df['label'] = label 
          result_df[f"{perturb}_reverse_error"]=reverse_error 
          result_df[f"{perturb}_reverse_error_verbose"]=reverse_error_verbose 
          result_df[f"{perturb}_error"]=error 
          result_df[f"{perturb}_error_verbose"]=error_verbose 
        with open(f"{args.save_location}/{args.model_name}-epoch{i}",'wb') as f:
          pickle.dump(result_df,f)
    else:
      #### modify the algorithms #### 
      if args.get_ece:
        print(f"processing..{args.perturb_names_by}- with ece")
        Female_instances,Male_instances=get_name_lists(name_gender,sort_by=args.perturb_names_by,num=200)
        temp, label, reverse_error, reverse_error_verbose, reverse_confs =perturb_instances_with_ece(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances,Reverse=True)
        temp, label, error, error_verbose, all_confs =perturb_instances_with_ece(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances)
        result_df[f"{args.perturb_names_by}_reverse_error"]=reverse_error
        result_df[f"{args.perturb_names_by}_reverse_error_verbose"]=reverse_error_verbose 
        result_df[f'{args.perturb_names_by}_error']=error 
        result_df[f'{args.perturb_names_by}_error_verbose']=error_verbose 
        result_df[f'{args.perturb_names_by}_conf']=all_confs 
        result_df[f'{args.perturb_names_by}_reverse_conf']=reverse_confs
        #result_df[f'{args.perturb_names_by}_label'] = label
        with open(f"{args.save_location}/{args.model_name}-epoch{i}-{args.perturb_names_by}-with_ece",'wb') as f: 
          pickle.dump(result_df,f) 

      
      else: 
        print(f'Processing..{args.perturb_names_by}')
        Female_instances,Male_instances=get_name_lists(name_gender,sort_by=args.perturb_names_by,num=200)
        temp, label, reverse_error, reverse_error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances,Reverse=True)
        temp, label, error, error_verbose =perturb_instances(template_data,template_test_labels, tokenizer, model, DEVICE, Female_instances,Male_instances)
        result_df[f"{args.perturb_names_by}_reverse_error"]=reverse_error
        result_df[f"{args.perturb_names_by}_reverse_error_verbose"]=reverse_error_verbose 
        result_df[f'{args.perturb_names_by}_error']=error 
        result_df[f'{args.perturb_names_by}_error_verbose']=error_verbose 
        with open(f"{args.save_location}/{args.model_name}-epoch{i}-{args.perturb_names_by}",'wb') as f: 
          pickle.dump(result_df,f) 




if __name__ =="__main__":
  main() 