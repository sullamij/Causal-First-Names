import os 
import json 
from IPython import display as d 
import random 
import matplotlib.pyplot as plt 

import numpy as np
import torch
from torch.nn import functional as F
from sklearn import decomposition
from typing import Dict, Optional, List, Tuple, Union

class OutputSeq: 
    def __init__(self,
                 token_ids =None, 
                 n_input_tokens = None,
                 tokenizer=None, 
                 output_text = None, 
                 tokens = None, 
                 encoder_hidden_states=None, 
                 decoder_hidden_states =None, 
                 embedding_states=None, 
                 attribution=None, 
                 activations=None,
                 collect_activations_layer_nums=None,
                 attention=None,
                 model_type:str='mlm',
                 lm_head=None,
                 device='cpu',
                 config=None):
        
        self.token_ids = token_ids
        self.tokenizer = tokenizer
        self.n_input_tokens = n_input_tokens
        self.output_text = output_text
        self.tokens = tokens
        self.encoder_hidden_states = encoder_hidden_states
        self.decoder_hidden_states = decoder_hidden_states
        self.embedding_states = embedding_states
        self.attribution = attribution
        self.activations = activations
        self.collect_activations_layer_nums = collect_activations_layer_nums
        self.attention_values = attention
        self.lm_head = lm_head
        self.device = device
        self.config = config
        self.model_type = model_type
    
    def _get_encoder_hidden_states(self): 
        return self.encoder_hidden_states if self.encoder_hidden_states is not None else self.decoder_hidden_states
    
    def _get_hidden_states(self)->Tuple[Union[torch.Tensor,None], Union[torch.Tensor,None]]: 
        return (self.encoder_hidden_states, self.decoder_hidden_states)
    
    def __str__(self):
        return "<LMOutput '{}' # of lm outputs: {}>".format(self.output_text, len(self._get_hidden_states()[1][-1]))
    
    def to(self,tensor:torch.Tensor):
        if self.device !='cpu':
            return tensor.to(self.device)
        return tensor 
    
    def __call__(self,position=None, **kwargs):
        if position is not None:
            self.position(position,**kwargs)
        else:
            self.primary_attributions(**kwargs)

    def position(self,position,attr_method='grad_x_input'):
        if (position < self.n_input_tokens) or (position> len(self.tokens)-1):
            raise ValueError("'position' should indicate a position of a generated token. "
                             "Accepted values for this sequence are between {} and {}."
                             .format(self.n_input_tokens, len(self.tokens) - 1))
        
        importance_id = position - self.n_input_tokens 
        tokens = [] 
        ##### TBD ####

    def primary_attributions(self,attr_method: Optional[str]='grad_x_input',
                             style='minimal',ignore_tokens: Optional[List[int]]=[],**kwargs):
        """
        """ ##TBD ##
        return None 
    
    def run_nmf(self,**kwargs): 
        """"""
        return NMF(self.activations,
                   n_input_tokens=self.n_input_tokens,
                   token_ids=self.token_ids,
                   tokens=self.tokens,
                   config = self.config,
                   collect_activations_layer_num=self.collect_activations_layer_nums,
                   **kwargs)



class NMF: 
    """conducts NMF and holds the models and components
    """
    def __init__(self, activations: Dict[str,np.ndarray],
                 n_input_tokens: int = 0 , 
                 token_ids: torch.Tensor = torch.Tensor(0),
                 _path: str="",
                 n_components: int = 10, 
                 from_layer: Optional[int] = None, 
                 to_layer: Optional[int] = None, 
                 tokens: Optional[List[str]] = None, 
                 collect_activations_layer_num: Optional[List[int]] = None, 
                 config=None, 
                 **kwargs
                 ):
        """
        Receives a neuron activations tensor from OutputSeq and decomposes it using NMF into the number 
        of components specified by `n_component`. For example, a model like distilgpt2 neurons. Using NMF to 
        reduce them to 32 components  can reveal interesting underlying firing patterns. 

        Args: 
            activations: (batch, layer, neuron, position)
            n_input_tokens : number of input tokens.
            token_ids: List of token ids. 
            _path : path to find javascript that create interactive explorables 
            n_components: Number of components/ factors to reduce the neuron factos to. 
            tokens: the text of each token. 
            collect_activations_layer_nums: the list of layer ids whosse activations were collected. 
                            If None, then all layers collected.
        """
        if activations == []: 
            raise ValueError (f"No activation data found. make sure `activations=True` passed to f'ecco.from_pretrained().'")
        
        self._path = _path 
        self.token_ids = token_ids 
        self.n_input_tokens = n_input_tokens 
        self.config = config 

        # Joining Encoder and Decoder (if exists) together 
        activations = np.concatenate(list(activations.values()),axis=-1)

        merged_act = self.reshape_activations(activations, 
                                              from_layer, 
                                              to_layer,
                                              collect_activations_layer_num)
        # merged_act (neuron * layer, batch * position) 
        activations = merged_act 
        self.tokens = tokens 
        n_output_tokens = activations.shape[-1] # position * batch
        n_layers = activations.shape[0] # neuron * layer 
        n_components = min([n_components, n_output_tokens]) 
        components = np.zeros((n_layers, n_components, n_output_tokens)) 
        models = [] 

        # Get rid of negative acitvation values 
        # in GPT2 there are some because of GELU 
        self.activations = np.maximum(activations, 0).T # (batch*position, neuron_layer)

        self.model = decomposition.NMF(n_components=n_components,
                                       init='random',
                                       random_state=0, 
                                       max_iter=500)
        self.components = self.model.fit_transform(self.activations).T #(neuron * layer ,  )


    @staticmethod
    def reshape_activations(activations, 
                            from_layer: Optional[int] = None, 
                            to_layer: Optional[int] = None, 
                            collect_activations_layer_num: Optional[List[int]] = None): 
        """Prepares the activations tensors for NMF by reshaping it from four dimensions 
        (batch, layer, neuron, position) -> ( neuron (and layer), position (and batch)). 
        """
        if len(activations.shape) != 4:
            raise ValueError(f"The 'activations' parameter should have four dimensions: "
                            f"(batch, layers, neurons, positions). "
                            f"Supplied dimensions: {activations.shape}", 'activations')
        
        if collect_activations_layer_num is None: 
            collect_activations_layer_num = list(range(activations.shape[1])) 
        
        layer_nums_to_row_ixs = {layer_num: i for i, layer_num in enumerate(collect_activations_layer_num)}

        if from_layer is not None or to_layer is not None: 
            from_layer = from_layer if from_layer is not None else 0 
            to_layer = to_layer if to_layer is not None else activations.shape[0] 

            if from_layer == to_layer:
                raise ValueError(f"from_layer ({from_layer}) and to_layer ({to_layer}) cannot be the same value. "
                                "They must be apart by at least one to allow for a layer of activations.")

            if from_layer > to_layer:
                raise ValueError(f"from_layer ({from_layer}) cannot be larger than to_layer ({to_layer}).")
            
            layer_nums = list(range(from_layer,to_layer))
        else:
            layer_nums = sorted(layer_nums_to_row_ixs.keys())
        
        if any([num not in layer_nums_to_row_ixs for num in layer_nums]):
            available = sorted(layer_nums_to_row_ixs.keys())
            raise ValueError(f"Not all layers between from_layer ({from_layer}) and to_layer ({to_layer}) "
                            f"have recorded activations. Layers with recorded activations are: {available}")
        
        row_ixs = [layer_nums_to_row_ixs[layer_num] for layer_num in layer_nums]
        # Merge 'layers' and 'neuron' dimentions. 
        activation_rows = [activations[:,row_ix] for row_ix in row_ixs] # extracts by layer (batch, neuron, position) * layer_length 
        merged_act = np.concatenate(activation_rows,axis=1) # (batch, num_layers*neuron, pos)
        merged_act = merged_act.swapaxes(0,1) # (neurons* layer, batch, position)
        merged_act = merged_act.reshape(merged_act.shape[0],-1) # (neuron*layer, batch*position)
        
        return merged_act 
        
    def explore(self,input_sequence: int=0, **kwargs): 
        """
        Shows interactive explorable for a single sequence with sparklines to isolate factors. 

        Args: 
            input_sequence: Which sequence in the batch to show. 
        """
        tokens = [] 
        for idx, token in enumerate(self.tokens[input_sequence]): 
            type = 'input' if idx < self.n_input_tokens else 'output' 
            tokens.append({'token': token, 
                           'token_id': int(self.token_ids[input_sequence][idx]),
                           'type': type, 
                           'position':idx})
            
        ### TBD 
    
    def plot(self, n_components=3): 
        for idx, comp in enumerate(self.components): 
            print('Layer {} components'.format(idx))
            comp = comp[:n_components, :].T

            fig, ax1 = plt.subplots(1) 
            plt.subplots_adjust(wspace=.4) 
            fig.set_figheight(2) 
            fig.set_figwidth(17) 

            ## PCA line plot 
            ax1.plot(comp) 
            ax1.set_xticks(range(len(self.tokens))) 
            ax1.set_xticklabels (self.tokens, rotation=-90) 
            ax1.legend(['Component {}'.format(i+1) for i in range(n_components)],
                       loc='center_left',
                       bbox_to_anchor=(1.01,0.5))
            
            plt.show() 


