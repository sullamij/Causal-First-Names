import sys
sys.path.insert(0,'/content/gdrive/MyDrive/Comsense_eval')

#from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from transformers import GPT2Model
from transformers.modeling_outputs import (QuestionAnsweringModelOutput)
#from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaTokenizer, AutoTokenizer, RobertaModel, RobertaConfig
from transformers import GPT2PreTrainedModel, GPT2Config, GPT2Model,BertModel, BertPreTrainedModel,BertConfig

# sources: https://github.com/ftarlaci/GPT2sQA/blob/41cd86ef5c2051ad3fda224ac912d97d07f73f61/gpt2sqa/modeling_gpt2.py
class GPT2ModelForQuestionAnswering():
  """A linear layer on top of pre-trained GPT-2 output that computes start_logits and end_logits
  Params:
      `config`: a BertConfig class instance with the configuration to build a new model.
  Inputs:
      `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
          with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
          `extract_features.py`, `run_classifier.py` and `run_squad.py`)
      `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
          types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
          a `sentence B` token (see BERT paper for more details).
      `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
          selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
          input sequence length in the current batch. It's the mask that we typically use for attention when
          a batch has varying length sentences.
      `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
          Positions are clamped to the length of the sequence and position outside of the sequence are not taken
          into account for computing the loss.
      `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
          Positions are clamped to the length of the sequence and position outside of the sequence are not taken
          into account for computing the loss.
  Outputs:
      if `start_positions` and `end_positions` are not `None`:
          Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
      if `start_positions` or `end_positions` is `None`:
          Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
          position tokens of shape [batch_size, sequence_length].
  """
  def __init__(self,config,):
    super().__init__(config)
    self.gpt2 = GPT2Model(config)
    self.qa_outputs = nn.Linear(config.n_embed,2)

  def forward(self,input_ids,token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None):
    sequence_output, _ = self.gpt2(input_ids, attention_mask, token_type_ids)
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1,dim=2)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    if start_positions is not None and end_positions is not None:
      if len(start_positions.size())>1: 
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size())>1:
        end_positions = end_positions.squeeze(-1)
      
      ignored_index = start_logits.size(1)
      start_positions.clamp_(0,ignored_index)
      end_positions.clamp_(0,ignored_index)

      loss_function = CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_function(start_logits, start_positions)
      end_loss = loss_function(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2 
      return total_loss 
    else:
      return start_logits, end_logits

class RobertaClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""
  def __init__(self,config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    classifier_dropout = (
        config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self,features,**kwargs):
    x = features [:,0,:] # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x 

class RobertaForSequenceClassification(RobertaModel):
  def __init__(self,model, config, dropout=None):
    super(RobertaForSequenceClassification,self).__init__(config)
     
    self.num_labels = config.num_labels 
    self.config = config 
    
    self.model = model
    self.classifier = RobertaClassificationHead(config)

    #self.post_init() #initialize wieghts and apply final processing 
  
  def forward(self, input_ids, attention_mask, labels):
    outputs = self.model(
        input_ids, attention_mask=attention_mask)
    sequence_output = outputs[0]
    logits = self.classifier(sequence_output)
    #print(logits.size()) # (1,3) 

    loss = None 

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))

    return loss, logits

# https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/roberta/modeling_roberta.py#L1466
class RobertaForQuestionAnswering(RobertaModel):
  def __init__(self,model,config):
    super(RobertaForQuestionAnswering,self).__init__(config)
    
    self.num_labels = config.num_labels
    self.roberta = model
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    #self.post_init() #initialize wieghts and apply final processing 

  def forward(self,input_ids, attention_mask, start_positions=None,end_positions=None):
    """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss. 
    """

    outputs = self.roberta(
      input_ids,
      attention_mask=attention_mask)

    sequence_output = outputs[0]

    logits = self.qa_outputs(sequence_output)
    
    start_logits, end_logits = logits.split(1,dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous() 
    end_logits = end_logits.squeeze(-1).contiguous() 

    total_loss = None 
    if start_positions is not None and end_positions is not None:
      # multi gpu - split andd a dimension
      if len(start_positions.size()) >1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size())>1:
        end_positions = end_positions.squeeze(-1)

      # sometimes the start/end positions are outside our model inputs, we ignore them
      ignored_index = start_logits.size(1)
      start_positions = start_positions.clamp(0,ignored_index)
      end_positions = end_positions.clamp(0,ignored_index)

      loss_fct = nn.CrossEntropyLoss(ignore_index = ignored_index)
      start_loss = loss_fct(start_logits,start_positions)
      end_loss = loss_fct(end_logits,end_positions)
      total_loss = (start_loss + end_loss) /2 

    return total_loss, (start_logits,end_logits) 

class GPT2ForSequenceClassification(GPT2PreTrainedModel):
  def __init__(self,model,config):
    super().__init__(config)
    self.num_labels = config.num_labels 
    self.transformer = model 
    self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

    #self.post_init() # initialize weights and apply final processing 

  def forward(self, input_ids, attention_mask, labels):
    transformer_outputs = self.transformer(
        input_ids, 
        attention_mask = attention_mask, 
    )

    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    batch_size, sequence_length = input_ids.shape[:2]
    
    sequence_lengths = -1
    if self.config.pad_token_id is None:  # as GPT2 use the last seq to classify 
      print('warning : padding tokens cannot be found') 
    else: 
      if input_ids is not None:
        sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1)-1
        #print(f"seq_len:{sequence_lengths}")
      else: 
        sequence_lengths = -1 
        print('warning : padding tokens cannot be found')
    
    pooled_logits = logits [torch.arange(batch_size, device = logits.device), sequence_lengths]

    loss = None

    loss_fct = nn.CrossEntropyLoss() 
    loss = loss_fct(pooled_logits.view(-1,self.num_labels),labels.view(-1))

    return loss, pooled_logits 

class BertForSequenceClassification(BertPreTrainedModel):
  def __init__(self,model,config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config 
    self.bert = model 
    classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
    self.dropout = nn.Dropout(classifier_dropout)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    #self.post_init() # initialize wieghts and apply final processing 

  def forward(
      self, input_ids, attention_mask, labels):
    outputs = self.bert(input_ids, attention_mask = attention_mask)
    pooled_output = outputs [1]
    pooled_output = self.dropout(pooled_output) 
    logits = self.classifier(pooled_output)

    loss = None 
    loss_fct = nn.CrossEntropyLoss() 
    loss = loss_fct(logits.view(-1,self.num_labels), labels.view(-1))

    return loss, logits

class Multiple_choice_model_gpt2(nn.Module):
  def __init__(self,model,device):
    super(Multiple_choice_model_gpt2,self).__init__() 
    self.model = model 

    self.dropout = nn.Dropout(self.model.config.resid_pdrop)
    self.classifier = nn.Linear(self.model.config.hidden_size,1).to(device)
  
  def forward(self,input_ids,attention_mask,labels=None):
    bsz = input_ids.shape[0]
    num_choices = input_ids.shape[1] #input_id shape: (bsz, num_choices, seq_length)
    flat_input_ids = input_ids.view(-1,input_ids.size(-1)) # shape ([bsz*num_choices,seq_length])
    flat_attention_mask = attention_mask.view(-1,attention_mask.size(-1))
    
    outputs = self.model(
        input_ids = flat_input_ids,
        attention_mask=flat_attention_mask,
    )
    
    # https://github.com/uber-research/PPLM/blob/master/paper_code/pytorch_pretrained_bert/modeling_gpt2.py

    #last_hidden_state=outputs['last_hidden_state'] # size(bsz*num_choices,seq_length,hidden_size)
    last_hidden_state=outputs[0]
    resized_hidden_state=last_hidden_state.view(bsz,num_choices,input_ids.size(-1),-1) # (bsz,num_choices,seq_length,hidden_size)

    if self.model.config.pad_token_id is None: # as GPT2 use the last seq token to classify 
      print('waring: padding tokens cannot be found')
    else: 
      mc_token = torch.ne(input_ids,self.model.config.pad_token_id)
      mc_token_ids=mc_token.sum(axis=-1)-1 # (bsz,num_choices)
    
    mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,resized_hidden_state.size(-1)) #
    #print(f'mc_token_ids.shape:{mc_token_ids.shape}') #(bsz,num_choices,1,hidden_size)?
    #mc_token_ids.shape:torch.Size([8, 3, 1, 768])
    multiple_choice_h = resized_hidden_state.gather(2,mc_token_ids).squeeze(2)
    #multiple_choice_h.shape:torch.Size([8, 3, 768])
    #print(f'multiple_choice_h.shape:{multiple_choice_h.shape}') # (bsz,num_choices,hidden_size)
    multiple_choice_logits = self.classifier(multiple_choice_h).squeeze(-1) 
    #multiple_choice_logits.shape:torch.Size([8, 3])
    #print(f'multiple_choice_logits.shape:{multiple_choice_logits.shape}') # (bsz,num_choices)

    loss_fct=nn.CrossEntropyLoss()
    loss = loss_fct(multiple_choice_logits.view(-1,multiple_choice_logits.size(-1)),labels.view(-1))

    return loss,multiple_choice_logits.view(-1,num_choices)

class Multiple_choice_model_bert(nn.Module):
  def __init__(self,model,device):
    super(Multiple_choice_model_bert,self).__init__() 
    self.model = model 

    self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
    self.classifier = nn.Linear(self.model.config.hidden_size,1).to(device)
  
  def forward(self,input_ids,attention_mask,labels=None):
    #print(f'input_ids.shape:{input_ids.shape}')
    num_choices = input_ids.shape[1] #input_id shape: (batch_num_choices,seq_length)
    flat_input_ids = input_ids.view(-1,input_ids.size(-1))
    #print(f'flat_input_ids.shape:{flat_input_ids.shape}')
    flat_attention_mask = attention_mask.view(-1,attention_mask.size(-1))
    #print(f'input_ids:{input_ids}')
    
    outputs = self.model(
        input_ids = flat_input_ids,
        attention_mask=flat_attention_mask,
    )


    #pooled_output = outputs['pooler_output']
    pooled_output = outputs[1]
    #print(f'pooled_output.shape:{pooled_output.shape}')
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    #print(f'logits.shape:{logits.shape}')
    reshaped_logits = logits.view(-1,num_choices)
    #print(f'reshaped_logits.shape:{reshaped_logits.shape}')


    loss_fct=nn.CrossEntropyLoss() 
    loss = loss_fct(reshaped_logits,labels)
    #print(f'labels.shape:{labels.shape}')
    #print(f'loss: {loss}, labels: {labels}')
    return loss, reshaped_logits

class Multiple_Choice_Model_roberta(nn.Module):
    def __init__(self, roberta_model: RobertaModel, dropout: float = None):
          super(Multiple_Choice_Model_roberta, self).__init__()
          self.model = roberta_model
          self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
          self.classifier = nn.Linear(self.model.config.hidden_size, 1)
   
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels=None):
          num_choices = input_ids.shape[1] 
          flat_input_ids = input_ids.view(-1, input_ids.size(-1))
          flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

          outputs = self.model(
              input_ids = flat_input_ids,
              attention_mask=flat_attention_mask,
          )
          pooled_output = outputs[1] 

          pooled_output = self.dropout(pooled_output)
          logits = self.classifier(pooled_output)
          reshaped_logits = logits.view(-1, num_choices)

          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(reshaped_logits, labels)

          return loss, reshaped_logits