

from collections import defaultdict
import inspect
import json
import os
import random
import torch
import transformers
from transformers import BatchEncoding
from src.interpreter_output import OutputSeq

import numpy as np
import re
from IPython import display as d
from torch.nn import functional as F
from typing import Optional, Any, List, Tuple, Dict, Union
from operator import attrgetter

from transformers import GPT2Model, AutoTokenizer 
from typing import Any, Dict, Optional, List

class Interpreter(object): 
    """
    Wrapper for the multiple choice language model 
     - model.model : language model base 
     - model. classifier : classifier 
    """

    def __init__(self,
                 model,
                 tokenizer,
                 model_name: str,
                 collect_activations_flag=False,
                 collect_activations_layer_nums=None, 
                 verbose=True, 
                 gpu=True):
        
        self.model_name = model_name 

        self.orig_model = model
        self.model = model.model
        self.classifier = model.classifier  

        if torch.has_mps and gpu: 
            self.model = self.model.to('mps')
            self.classifier =self.classifier.to('mps')
            self.device='mps'
        elif torch.cuda.is_available() and gpu: 
            self.model = self.model.to('cuda')
            self.classifier =self.classifier.to('cuda')
            self.device='cuda'
        else: 
            self.device = 'cpu'
        
        self.tokenizer = tokenizer 
        self.verbose = verbose 
        self._path = os.curdir

        #neuron activation 
        self.collect_activation_flag = collect_activations_flag 
        self.collect_activations_layer_nums = collect_activations_layer_nums

        # For each model, this indicates the layer whose activations will be collected 
        embeddings_layer_name = 'wte.weight'
        embed_retriever = attrgetter(embeddings_layer_name)
        self.model_embeddings = embed_retriever(self.model)
        self.collect_activations_layer_name_sig = 'mlp\.c_proj'

        self._reset() 

    def _reset(self): 
        self._all_activations_dict=defaultdict(dict)
        self.activations = defaultdict(dict) 
        self.all_activations = [] 
        self.generation_activations = [] 
        self.neurons_to_inhibit = {} 
        self.neurons_to_induce = {} 
        self._hooks = {} 

    def to(self,tensor:Union[torch.Tensor, BatchEncoding]): 
        if (self.device == 'cuda') or (self.device == 'mps'): 
            print(f".to sent to {self.device}")
            return tensor.to(self.device)
        return tensor 

    def __call__(self,input_tokens: torch.Tensor,attention_mask:torch.Tensor): 
        """

        """

        if self.model.device.type !=input_tokens.device.type: 
            #input_tokens = self.to(input_tokens)
            input_tokens = input_tokens.to(self.device)
            #attention_mask = self.to(attention_mask)
            attention_mask = attention_mask.to(self.device)
        

        self._attach_hooks(self.orig_model) ### pass orig_model to hook all

        if len(input_tokens.shape)<3:
            input_tokens = input_tokens.unsqueeze(0)
        if len(attention_mask.shape)<3:
            attention_mask = attention_mask.unsqueeze(0)
        
        #### manually perform inference in 2 steps - lm and the classifier
        bsz = input_tokens.shape[0]
        num_choices = input_tokens.shape[1] #input_id shape: (bsz, num_choices, seq_length)

        # Remove downstream. set to batch length 
        #n_input_tokens = len(input_tokens['input_ids'][0])
        n_input_tokens = bsz

        flat_input_ids = input_tokens.view(-1,input_tokens.size(-1)) # shape ([bsz*num_choices,seq_length])
        flat_attention_mask = attention_mask.view(-1,attention_mask.size(-1))
    
        outputs = self.model(
            input_ids = flat_input_ids,
            attention_mask=flat_attention_mask,
        ) ## passes to the lm 

        last_hidden_state=outputs[0]
        resized_hidden_state=last_hidden_state.view(bsz,num_choices,input_tokens.size(-1),-1) # (bsz,num_choices,seq_length,hidden_size)

        if self.model.config.pad_token_id is None: # as GPT2 use the last seq token to classify 
            print('waring: padding tokens cannot be found')
        else: 
            mc_token = torch.ne(input_tokens,self.model.config.pad_token_id)
            mc_token_ids=mc_token.sum(axis=-1)-1 # (bsz,num_choices)
        
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,resized_hidden_state.size(-1)) #
                #mc_token_ids.shape:torch.Size([8, 3, 1, 768])
        multiple_choice_h = resized_hidden_state.gather(2,mc_token_ids).squeeze(2)
             #multiple_choice_h.shape:torch.Size([8, 3, 768])
        multiple_choice_logits = self.classifier(multiple_choice_h).squeeze(-1) 

        multiple_choice_logits.detach().cpu().numpy() # multiple_choice_logits.view(-1,num_choices) 
        self.logit = multiple_choice_logits
         
        activations_dict = self._all_activations_dict 
        #for layer_type, 
        self.activations['type_null'] =activations_dict_to_array(activations_dict)

        encoder_hidden_states = None 
        decoder_hidden_states = getattr(outputs,'hidden_states', None ) 
        if decoder_hidden_states is None: 
            try: 
                decoder_hidden_states = outputs[2]
            except:
                print('warning, hidden_states not loaded')
        embedding_states = decoder_hidden_states[0]

        tokens = [] 
        for inst in input_tokens:
            t=[]
            for sent in inst:
                token =self.tokenizer.convert_ids_to_tokens(sent)
                t.append(token) 
            tokens.append(t)

        attn = getattr(outputs, 'attentions',None)
        return OutputSeq(**{'tokenizer':self.tokenizer,
                            'token_ids': input_tokens, 
                            'n_input_tokens': n_input_tokens,
                            'tokens':tokens,
                            'encoder_hidden_states':encoder_hidden_states,
                            'decoder_hidden_states': decoder_hidden_states,
                            'embedding_states': embedding_states,
                            'attention': attn, 
                            'activations': self.activations,
                            'collect_activations_layer_nums': self.collect_activations_layer_nums,
                            'lm_head': None,
                            'device': self.device,
                            'config': self.model.config
        })
        



    def _attach_hooks(self,model): 
        if self._hooks: 
            return # the hooks are already attached 
        
        for name, module in model.named_modules(): 
            # Add hooks to capture the activations in every FFNN

            if re.search (self.collect_activations_layer_name_sig, name):
                if self.collect_activation_flag:
                    self._hooks[name] = module.register_forward_hook(
                        #### modify for input_-> output_
                        #lambda self_, input_, output, name=name: self._get_activations_hook(name,input_)) # for gpt2 its (batchsize,seq_len,3072)
                        lambda self_, input_, output, name=name: self._get_activations_hook(name,output))
                # Register neuron inhibition hook
                self._hooks[name + '_inhibit'] = module.register_forward_pre_hook(
                    lambda self_, input_, name=name: \
                        self._inhibit_neurons_hook(name,input_)
                )

    def _get_activations_hook(self,name:str, output): 
        """ Collects the activation for all tokens 
        The default activations collection method. 

        Args: 
            input_: activation tuple to capture. A tuple containing one tensor of dimenstions 
            (batchsize, seq_len, 3072) 
        """
        layer_number = re.search("(?<=\.)\d+(?=\.)", name).group(0)
        
        collecting_this_layer = (self.collect_activations_layer_nums is None) or (
            layer_number in self.collect_activations_layer_nums)
        
        if collecting_this_layer: 
            # Initialize the layer's key the first time we encounter it 
            if layer_number not in self._all_activations_dict: 
                self._all_activations_dict[layer_number] = [0] 
            
            #if isinstance(output,tuple):
            #    print(name, len(output),output[0].shape)
            #else:
            #    print(name,output.shape)
            #self._all_activations_dict[layer_number] = output[0].detach().cpu().numpy() 
            self._all_activations_dict[layer_number]=output.detach().cpu().numpy()

    def _remove_hooks(self):
        for handle in self._hooks.values(): 
            handle.remove() 
        del self._hooks
        self._hooks = {} 

    def _inhibit_neurons_hook(self,name:str, input_tensor):
        """
        After being attached as a pre-forward hook, it sets to zero the activation value of the neurons
        indicated in the self.neurons_to_inhibit
        """
        layer_number =  re.search("(?<=\.)\d+(?=\.)", name).group(0)
        if layer_number in self.neurons_to_inhibit.keys(): 
            for n in self.neurons_to_inhibit[layer_number]:
                # print('inhibiting', layer_number,n)
                input_tensor[0][0][-1][n]=0 # tuple, batch, position 
        
        if layer_number in self.neurons_to_induce.keys(): 
            #print('layer_number',layer_number, input_tensor[0].shape )
            for n in self.neurons_to_induce[layer_number]: 
                input_tensor[0][0][-1][n] = input_tensor[0][0][-1][n] * 10 # tuple batch position 

        return input_tensor 
    
def activations_dict_to_array(activations_dict):
    """Converts the dict used to collect activations into an array of the shape 
    (batch, layers, neurons, token positions)
    Args: 
        activations_dict: python dict 
    """
    activations = [] 
    for i in sorted(activations_dict.keys()):
        activations.append(activations_dict[i]) 
    
    activations = np.array(activations)
    # layer, batch, position,  neurons 

    activations = np.swapaxes(activations,2,3)
    activations = np.swapaxes(activations,0,1)
    # (batch, layer, neurons,)
    return activations
    




        
