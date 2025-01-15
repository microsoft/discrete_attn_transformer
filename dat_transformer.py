# dat_transformer.py: our version of a decoder-only Transformer (based on PyTorch Transformer model) with changes for running our TGT parser/generator algorithm.
'''
Changes from vanilla PyTorch Transformer model:
   - no MLP/Feedforward module
   - ability to loop over a set of blocks (layers) 
   - use of PashaNorm in place of LayerNorm
   - use of PashaMax in place of softmax
   - input embeddings and residual stream maintained as a set of 1-hot registers
   - weights are programmed vs. learned (QKV and Output matricies)
   - attention over initial prompt is bidirectional (but causal over generated columns)
   - as each new column is generated, its full residual stream is passed to the next column
   - our output head uses the DatDecoder to decode the symbol register in the final col output

    - during new column generation, only the new column is computed (previous columns are fixed)
    --> need to verify this claim; can't find code that does this

Note: we use the PyTorch code for the TransformerEncoderLayer as the layers for our decoder-only Transformer, 
since it only uses self-attention.  We pass in the neccessary causal mask to have it behave in a 
decoder-only fashion (but with bidirectional attention over the initial prompt).
'''
import os
import sys
import copy
import time
import warnings
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
from torch import dropout, softmax, Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

from dat_common import *
from utils.dat_utils import *
from utils.trace_generator import TraceGenerator

class DatStepDecoder():
    def __init__(self, vocab, embedding, register_name_to_index):
        self.vocab = vocab
        self.embedding = embedding
        self.register_name_to_index = register_name_to_index
        self.num_registers = len(register_name_to_index)

    def decode_steps(self, value, layer_index):
        '''
        decode each of the steps for a column within a transformer layer.
            value: dict of values for each step (input, query, key, value, output, residual)
                --> values within each step are: [seq_len, d_model]  (where d_model = d_register * num_registers)
            layer_index: index of the layer (for debugging)
        '''
        #print("  decoding steps for layer: {}".format(layer_index))
        assert layer_index == value["layer_index"]

        dsteps = {}
        dsteps["input"] = [self.decode_step(v) for v in value["input"]]
        dsteps["query"] = [self.decode_step(v, layer_index, "query") for v in value["query"]]
        dsteps["key"] = [self.decode_step(v) for v in value["key"]]
        dsteps["value"] = [self.decode_step(v) for v in value["value"]]
        dsteps["output"] = [self.decode_step(v) for v in value["output"]]
        dsteps["residual"] = [self.decode_step(v) for v in value["residual"]]

        dsteps["attn_weights"] = value["attn_weights"]
        dsteps["layer_index"] = layer_index

        return dsteps

    def decode_step(self, value, layer_index=None, step_name=None, return_pairs=False, permute=False):
        '''
        decodes a "step" for a column within a transformer layer (steps: input, query, key, value, output, residual).  
            value: [bsz, 1, d_model]
            layer_index: index of the layer (for debugging)
            step_name: name of the step (for debugging)
            return_pairs: return as a dict of register names to values (vs. list of values)
            permute: permute value to "batch first" format before decoding
        '''
        if permute:
            value = value.permute(1, 0)
            
        # reshape into num_registers registers
        values = value.view(self.num_registers, -1)

        # do single logits and indexes calculations (for speed)
        # n=num_registers, r=d_register, v=d_vocab, d=d_register
        indexes = embedding_to_indexes(values, 'nd,vd->nv', self.embedding)

        text_values = [self.vocab[index] for index in indexes]

        # apply to text to rows with only 1 zero value
        if (layer_index == 0) and (step_name == "query"):
            a = 9    # breakpoint

        # above, we quickly processed the most common case: each register has zero or one ON values
        # NOW, we apply the exception patterns, as needed:
        #    - neq constant:          register has exactly 1 OFF value
        #    - in constant_list:      register has 2-9 ON values
        #    - not in constant_list:  register has 2-9 OFF values
        
        on_counts = (values > 0).sum(dim=-1)
        multi_on_counts = (on_counts > 1).any()

        if multi_on_counts:
            def get_in_symbol(row_value):
                index = embedding_to_indexes(row_value.squeeze(), 'd,vd->v', self.embedding)
                sym = self.vocab[index]
                return sym

            # we have an exception pattern
            off_counts = (values == 0).sum(dim=-1)

            for t, tv in enumerate(text_values):
                max_value = values[t].max().item()

                if on_counts[t] > 1 and on_counts[t] < 9:
                    # pattern: IN <constant_list>
                    on_indexes = (values[t] > 0).nonzero().squeeze(1)
                    text = "in [{}]".format(", ".join([self.vocab[i] for i in on_indexes]))
                    text_values[t] = text

                elif off_counts[t] == 1:
                    # pattern: != <constant>
                    not_row_value = (max_value - values[t]).unsqueeze(0)
                    sym = get_in_symbol(not_row_value)
                    text_values[t] = "!=" + sym

                elif off_counts[t] > 1 and on_counts[t] > 1:
                    # pattern: NOT IN <constant_list>
                    flip_values = max_value - values[t]
                    on_indexes = (flip_values > 0).nonzero().squeeze(1)
                    text = "not_in [{}]".format(", ".join([self.vocab[i] for i in on_indexes]))
                    text_values[t] = text

        if return_pairs:
            short_register_names = list(self.register_name_to_index.keys())
            text_values = {short_register_names[i]: text_values[i].strip() for i in range(self.num_registers) if text_values[i].strip()}

        return text_values

    def decode_register(self, value):
        index = embedding_to_indexes(value, 'd,vd->v', self.embedding)
        text_value = self.vocab[index]

        return text_value
    
class DatTransformer(nn.Module):

    def __init__(self, fixed_vocab, embedding, d_register, num_encoder_layers, log_progress=False, register_name_to_index=None, nhead=1, 
        dropout=.1, max_seq_len=5000, pad_idx=0, params=None, system_map=None):

        super(DatTransformer, self).__init__()

        self.decoder = DatStepDecoder(fixed_vocab, embedding, register_name_to_index)

        self.log_progress = log_progress
        self.num_registers = len(register_name_to_index)
        self.register_name_to_index = register_name_to_index
        self.system_map = None

        self.sym_reg_name = None
        self.pos_reg_name = None
        self.parse_reg_name = None
        self.eop_reg_name = None
        self.out_reg_name = None

        d_model = d_register * self.num_registers
        self.d_register = d_register

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.fixed_vocab = fixed_vocab

        self.encoder = TransformerEncoder(self.decoder, num_encoder_layers, d_model, nhead, dropout, 
            n_registers=self.num_registers, d_register=d_register)

        self.d_model = d_model
        self.nhead = nhead
        self.use_fill_masks = False    # True
        self.d_self = d_model
        self.layer_steps = []

        # ensure this is assigned AFTER reset_parameters() call
        self.embedding = embedding

        self.system_map = system_map

        if system_map:
            self.sym_reg_name = system_map["symbol"] if "symbol" in system_map else None
            self.pos_reg_name = system_map["position"] if "position" in system_map else None
            self.parse_reg_name = system_map["parse"] if "parse" in system_map else None
            self.eop_reg_name = system_map["eop"] if "eop" in system_map else None
            self.out_reg_name = system_map["output"] if "output" in system_map else None

        # ensure we default to NOT training mode
        self.eval()

    def forward(self, src, input_embeddings=None, gather_steps=False):
        '''
        When src (token indexes) and input_embeddings (outputs of transformer for previous columns) are both valid, we concatenate them together.
        '''
        src_key_padding_mask = None

        valid_src = src.shape[1] > 0
        if valid_src:
            # encode input and pos
            src_embeddings = self.make_embedding(src)

            # dt = self.decoder.decode_step(src_embeddings[:,0,:], return_pairs=True)
            # print("src_embeddings: {}".format(dt))

        if input_embeddings is not None:
            # overwrite position register in input_embeddings with correct values
            prompt_len = src.shape[1] if valid_src else 0

            input_embed_len = input_embeddings.shape[1]
            pos_text = [str(prompt_len+i) for i in range(1, 1+input_embed_len)]

            pos_indices = torch.tensor([self.s2i[pt] for pt in pos_text]).type(torch.long).unsqueeze(0)
            pos_tensor = get_index_embedding(self.embedding, pos_indices)
            self.set_register(input_embeddings, self.pos_reg_name, pos_tensor, self.d_register)

            if valid_src:
                # concat with input_embeddings
                src_embeddings = torch.cat( [src_embeddings, input_embeddings], dim=1)
            else:
                src_embeddings = input_embeddings

            # dummy up some src indexes (we were passed the actual input embeddings to use)
            if self.use_fill_masks:
                dummy_src = torch.ones(src_embeddings.shape[0], src_embeddings.shape[1], dtype=torch.long, device=input_embeddings.device)
                src_key_padding_mask = self.make_padding_mask(dummy_src)

        else:
            if self.use_fill_masks:
                src_key_padding_mask = self.make_padding_mask(src)

        # permute to "batch second" format
        src_embeddings = src_embeddings.permute(1, 0, 2)

        output, captured_steps = self.encoder(src_embeddings, src_no_peek_mask=None, src_key_padding_mask=src_key_padding_mask, 
            gather_steps=gather_steps)

        # permute output to "batch first" format
        model_output = output.permute(1, 0, 2)
        # output = [batch_size, trg_seq_len, d_h]

        # use reverse-mapping of embedding to get token predictions
        # we will use the first register of information (1/num_registers) for decoding our prediction
        symbol_output = model_output[:,:,0:self.d_register]
        symbol_indexes = embedding_to_indexes(symbol_output, 'btd,vd->btv', self.embedding)
        
        return model_output, symbol_indexes, captured_steps
    
    def load_weights(self, d_register, embedding, fn_weights=None, vocab=None, weights=None):
        if fn_weights:
            attn_matrix_layers = torch.load(fn_weights)
        elif weights:
            attn_matrix_layers = weights
        else:
            raise Exception("neither fn_weights nor weights provided")

        # NOTE: don't keep a reference to the entire (large) attn_matrix_layers object
        self.encoder.load_weights(d_register, self.embedding, vocab, attn_matrix_layers)

    def load_vocab(self, vocab):

        # ensure new vocab is based on the same vocab subset
        # fixed_len = len(self.fixed_vocab)
        # assert self.fixed_vocab == vocab[:fixed_len]
        
        self.decoder.vocab = vocab
        self.s2i = {word: i for i, word in enumerate(vocab)}
        self.vocab = vocab

    @torch.no_grad()
    def generate(self, prompt, device, max_new_tokens, example, return_last_steps=False, tracing_enabled=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """

        prompt_tokens = prompt.split()
        #gold_tokens = gold.split()

        stoi = {t: i for i, t in enumerate(self.vocab)}
        src_indexes = [stoi[t] for t in prompt_tokens]

        # make x look like a batch of 1 tensor
        idx = torch.tensor(src_indexes).to(device).unsqueeze(0)

        generated_outputs = None
        trace_generator = None
        generated_text = None
        sym_register_index = self.register_name_to_index[self.out_reg_name]
        
        # uncomment to enable tracing in IDE
        #tracing_enabled = True
        if tracing_enabled:
            trace_generator = TraceGenerator(example, self.decoder, all_gens=False) 

        for g in range(max_new_tokens):
            # predict next token
            model_output, symbol_indexes, curr_gen_steps = self.forward(idx, generated_outputs, gather_steps=tracing_enabled)
            last_col_output = model_output[:,-1,:].unsqueeze(1)

            # use decoder to get predicted symbol
            decoded_last_col_output = self.decoder.decode_step(last_col_output)
            predicted_token = decoded_last_col_output[sym_register_index]    
            last_step = (predicted_token == "." or g == max_new_tokens-1)

            if last_step and return_last_steps and not curr_gen_steps:
                # call again to get the last generation steps
                _, _, curr_gen_steps = self.forward(idx, generated_outputs, gather_steps=True)

            # append sampled index to the running sequence and continue
            if generated_outputs is None:
                generated_outputs = last_col_output
                generated_text = predicted_token
            else:
                generated_outputs = torch.cat((generated_outputs, last_col_output), dim=1)
                generated_text += " " + predicted_token

            if trace_generator:
                trace_generator.trace(g, curr_gen_steps)

            if last_step:
                break

        if trace_generator:
            trace_generator.close()

        return generated_text, curr_gen_steps
        
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
    def set_register(self, data, name, value, d_reg):
        index = self.register_name_to_index[name]
        data[:, :, index*d_reg:(index+1)*d_reg] = value

    def make_embedding(self, src):

        bsz, seq_len = src.shape

        assert max(src.flatten()) < self.embedding.weight.shape[0], "src tensor contains out-of-range indexes"

        fillers = self.embedding(src)
        one_index = self.s2i["1"]
        one_embeddings = self.embedding(one_index*torch.ones_like(src))
        
        # build EOP embedding (end-of-prompt flag, for last token)
        eop_embeddings = self.embedding(torch.zeros_like(src))
        eop_embeddings[0][-1] = get_symbol_embedding(self.embedding, self.s2i, "EOP")

        pos_text = [str(1+i) for i in range(seq_len)]
        pos_indices = torch.tensor([self.s2i[pt] for pt in pos_text]).type(torch.long).unsqueeze(0)
        pos_tensor = get_index_embedding(self.embedding, pos_indices)

        # start with all columns / all registers as zeros        
        data = torch.zeros( (bsz, seq_len, self.d_model)).type_as(fillers)

        # store fillers in symbol register
        if self.sym_reg_name in self.register_name_to_index:
            self.set_register(data, self.sym_reg_name, fillers, self.d_register)

        # store pos_tensor in position register
        if self.pos_reg_name in self.register_name_to_index:
            self.set_register(data, self.pos_reg_name, pos_tensor, self.d_register)

        # set parse register to 1
        if self.parse_reg_name in self.register_name_to_index:
            self.set_register(data, self.parse_reg_name, one_embeddings, self.d_register)

        # set Z register to EOP for last token
        if self.eop_reg_name in self.register_name_to_index:
            self.set_register(data, self.eop_reg_name, eop_embeddings, self.d_register)

        return data

    def make_padding_mask(self, data):
        src_mask = (data == self.pad_idx)
        return src_mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for i, (name, p) in enumerate(self.named_parameters()):
            if p.dim() > 1:
                #print("i={}, {}, {}".format(i, name, p.shape))
                nn.init.xavier_uniform_(p)

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        src = torch.rand(10, 32, 512)
        out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, decoder, num_layers, d_model, nhead, dropout, n_registers, d_register):
        super(TransformerEncoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(decoder, d_model, nhead, dropout, n_registers=n_registers, d_register=d_register)
        self.layers = _get_clones(encoder_layer, num_layers)

        self.decoder = decoder
        self.num_layers = num_layers
        self.norm = LayerNorm(d_model)

    def load_weights(self, d_register, embedding, vocab, attn_matrix_layers):
        last_repeat_group = None

        for i, layer in enumerate(self.layers):
            attn_matrix = attn_matrix_layers[i]
            layer.load_weights(d_register, embedding, vocab, attn_matrix)

            layer.repeat_layer_count = 0

            if "repeat_layer_count" in attn_matrix:
                repeat_group_count = attn_matrix["repeat_layer_count"] 
                repeat_group = attn_matrix["repeat_group"]

                if repeat_group != last_repeat_group:
                    last_repeat_group = repeat_group
                    layer.repeat_layer_count = repeat_group_count


    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_nopeek_mask(self, data):
        mask = self.generate_square_subsequent_mask(sz=data.shape[0]).to(data.device)
        return mask

    def forward(self, src: Tensor, src_no_peek_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                gather_steps=False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        def get_no_peek_mask(causal_attn):
            if causal_attn == "true":
                return src_no_peek_mask
            elif causal_attn == "false" or causal_attn is None:
                return None
            elif causal_attn == "reverse":
                return reversed_no_peek_mask
            else:
                raise ValueError("unknown causal_attn value: {}".format(causal_attn))

        output = src
        all_layer_steps = []
        src_no_peek_mask = self.make_nopeek_mask(src)
        reversed_no_peek_mask = src_no_peek_mask.flip([1])

        layer_index = 0

        while layer_index < len(self.layers):  

            if layer_index == 19 and gather_steps:
                bp = 9     # breakpoint for layer 20

            mod = self.layers[layer_index]

            repeat_layer_count = mod.repeat_layer_count
            if repeat_layer_count:
                #condition = attn_matrix["until"]

                if True:    # not condition:
                    # repeat specified layers until output matches previous output
                    prev_output = None     # will force us to process layers at least twice
                    
                    for r in range(100):    # max 100 repeats
                        repeat_steps = []

                        for w in range(repeat_layer_count):
                            layer = self.layers[w + layer_index]
                            no_peek_mask = get_no_peek_mask(layer.causal_attn)

                            output, steps = layer(output, src_mask=no_peek_mask, src_key_padding_mask=src_key_padding_mask, gather_steps=gather_steps, layer_index=layer_index + w)
                            repeat_steps.append(steps)

                        if prev_output is not None and torch.equal(prev_output, output):
                            break

                        prev_output = output

                    #print("Layer: {}, REPEATS: {}".format(1+layer_index, 1+r))

                    if r == 100:
                        raise Exception("reached max number of repeats")
                    
                    # skip over layers that we processed
                    layer_index += repeat_layer_count
                    assert len(repeat_steps) == repeat_layer_count     # only accrue the steps for the last pass thru the layers

                    if gather_steps:
                        all_layer_steps += repeat_steps

                else:
                    # repeat until output matches specified condition
                    raise NotImplementedError("until condition not implemented yet")

            else:
                # normal layer
                no_peek_mask = get_no_peek_mask(mod.causal_attn)

                output, layer_steps = mod(output, src_mask=no_peek_mask, src_key_padding_mask=src_key_padding_mask, gather_steps=gather_steps, layer_index=layer_index)
                layer_index += 1

                if gather_steps:
                    all_layer_steps.append(layer_steps)


        # if self.norm is not None:
        #     output = self.norm(output)

        return output, all_layer_steps

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, decoder, d_model, nhead, dropout=0.1, n_registers=0, d_register=0, params=None):
        super(TransformerEncoderLayer, self).__init__()

        self.num_registers = n_registers
        self.d_register = d_register

        self.self_attn = MultiheadAttention(decoder, d_model, nhead, dropout=dropout, params=params)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.repeat_group_count = 0     # of layers in this repeat group
        self.causal_attn = None       # set during load_weights()

        # debug
        self.fake_linear = nn.Linear(d_model, d_model)

    def load_weights(self, d_register, embedding, vocab, attn_matrix_layer):

        self.causal_attn = attn_matrix_layer["causal_attn"]
        self.self_attn.load_weights(d_register, embedding, vocab, attn_matrix_layer)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, gather_steps=False, 
                layer_index=None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # normal single pass over layer
        src2, steps = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, gather_steps=gather_steps)

        #output = src + self.dropout1(src2)
        #output = self.norm1(output)

        use_pasha_norm = True

        if not use_pasha_norm:
            # use MASK to apply src2 to src
            bsz, seq_len, d_model = src.shape

            output = src.view(bsz, -1, self.d_register)
            src2_view = src2.view(bsz, -1, self.d_register)

            mask = (src2_view != 0).any(dim=-1)
            output[mask] = src2_view[mask]

            output2 = output.view(bsz, seq_len, d_model)

        else:
            output = src + src2
            output2 = self.pasha_norm(output, self.d_register)

        if gather_steps:
            steps["residual"] = output2
            steps["layer_index"] = layer_index     # debugging

        return output2, steps

    def pasha_norm(self, data, d_register, threshold=0.55):
        '''
        this function reestablishes the one-hot encoding of the registers, unless the
        register is all zeros, in which case it is left as is.
        '''
        # input shape: [seq_len, bsz=1, d_model] 
        seq_len, bsz, d_model = data.shape

        # reshape to: [seq_len, n_registers, d_register]
        data2 = data.view(seq_len, -1, d_register)

        # Get the max values and indices along the last dimension
        max_values, max_indices = data2.max(dim=-1)

        # Create a mask where max values are greater than or equal to 0.95
        mask = max_values >= threshold

        # Create a tensor filled with zeros, with the same shape as your vectors
        one_hot = torch.zeros_like(data2)

        # For each vector in the batch, only create a one-hot if the max is >= 0.95
        one_hot[mask, max_indices[mask]] = 1

        # reshape to input shape: [seq_len, bsz=1, d_model]
        one_hot2 = one_hot.view(seq_len, bsz, d_model)

        return one_hot2


    
class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, decoder, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, 
        kdim=None, vdim=None, params=None):

        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.decoder = decoder

        #self._qkv_same_embed_dim = False   # treat weights separately for each transfer in load_weights()  
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // 4      # num_heads
        #assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        matrix_dim = self.head_dim * num_heads

        #self.in_proj_weight = Parameter(torch.empty(3 * matrix_dim, embed_dim))
        #self.in_proj_bias = Parameter(torch.empty(3 * matrix_dim))
        #self.out_proj = _LinearWithBias(matrix_dim, embed_dim)
        self.q_proj_weight =  Parameter(torch.empty(1))     # will set this in load_weights()
        self.k_proj_weight =  Parameter(torch.empty(1))     # will set this in load_weights()
        self.v_proj_weight =  Parameter(torch.empty(1))     # will set this in load_weights()

        self.bias_q =  Parameter(torch.empty(1))     # will set this in load_weights()
        self.bias_k =  Parameter(torch.empty(1))     # will set this in load_weights()
        self.bias_v =  Parameter(torch.empty(1))     # will set this in load_weights()

        #self.in_proj_bias = Parameter(torch.empty(1))        # will set this in load_weights()
        self.out_proj_weight = Parameter(torch.empty(1))     # will set this in load_weights()
        self.out_proj_bias = Parameter(torch.empty(1))       # will set this in load_weights()

        # self.register_parameter('q_proj_weight', None)
        # self.register_parameter('k_proj_weight', None)
        # self.register_parameter('v_proj_weight', None)

        self.in_proj_weight = None
        self.in_proj_bias = None

        self.add_zero_attn = add_zero_attn
        self.timer = None   # AvgTimer()

        # set during load_weights()
        self.right_match = False

    def expand_weight_matrix(self, d_register, fixup, factor):
        all_regs_matrix = torch.zeros(self.num_registers*d_register, self.num_registers*d_register).type(EMBED_DATA_TYPE)
        identity_matrix = torch.eye(d_register).type(EMBED_DATA_TYPE)

        for (mat_type, dest_start, src_start, fixup_type) in fixup:

            if mat_type == "pos_decrement":
                core_matrix = get_decrement_matrix(d_register)
            elif mat_type == "pos_increment":
                core_matrix = get_increment_matrix(d_register)
            elif mat_type == "identity":
                core_matrix = identity_matrix
            else:
                raise ValueError("unknown mat_type: {}".format(mat_type))

            if fixup_type == "minus":
                core_matrix = -core_matrix
            elif fixup_type == "one_minus":
                core_matrix = 1 - core_matrix
            elif fixup_type:
                raise ValueError("unknown fixup_type: {}".format(fixup_type))

            all_regs_matrix[dest_start:dest_start+d_register, src_start:src_start+d_register] = core_matrix
        
        # apply factor
        all_regs_matrix = all_regs_matrix * factor

        return all_regs_matrix
        
    def expand_weight_bias(self, d_register, embedding, fixup, factor):
        one_hot_registers = torch.zeros(self.num_registers * d_register).type(EMBED_DATA_TYPE)

        # create and process each register
        for (token_index, dest_start, fixup_type) in fixup:

            if token_index == "ones":
                core_bias = torch.ones(d_register)
            else:
                assert(isinstance(token_index, int))
                core_bias = embedding(torch.tensor(token_index))

            if fixup_type == "one_minus":
                core_bias = 1 - core_bias
            elif fixup_type:
                raise ValueError("unknown fixup_type: {}".format(fixup_type))

            one_hot_registers[dest_start:dest_start+d_register] = core_bias

        # apply factor
        one_hot_registers = one_hot_registers * factor

        return one_hot_registers
        
    def load_weights(self, d_register, embedding, vocab, attn_matrix):

        # HERE: apply attr_matrix weights to our self-attention matrices
        if WEIGHT_TYPE == "direct":
            q_factor, qwfixup, qbfixup = attn_matrix["q"]
            k_factor, kwfixup, kbfixup = attn_matrix["k"]
            vfactor, vwfixup, vbfixup = attn_matrix["v"]
            ofactor, owfixup, obfixup = attn_matrix["output"]

            # expand from direct embedding into 1-hot registers 
            qbias = self.expand_weight_bias(d_register, embedding, qbfixup, q_factor)
            kbias = self.expand_weight_bias(d_register, embedding, kbfixup, k_factor)
            vbias = self.expand_weight_bias(d_register, embedding, vbfixup, vfactor)
            obias = self.expand_weight_bias(d_register, embedding, obfixup, ofactor)

            # expand from direct embedding into matrices
            qw = self.expand_weight_matrix(d_register, qwfixup, q_factor)
            kw = self.expand_weight_matrix(d_register, kwfixup, k_factor)
            vw = self.expand_weight_matrix(d_register, vwfixup, vfactor)
            ow = self.expand_weight_matrix(d_register, owfixup, ofactor)

        else:
            qw, qbias = attn_matrix["q"]
            kw, kbias = attn_matrix["k"]
            vw, vbias = attn_matrix["v"]
            ow, obias = attn_matrix["output"]

        # causal_attn handled in layer.load_weights()
        self.right_match = attn_matrix["right_match"]

        self.q_proj_weight.data = qw  
        self.k_proj_weight.data = kw  
        self.v_proj_weight.data = vw  
        self.out_proj_weight.data = ow 

        self.bias_q.data = qbias  
        self.bias_k.data = kbias
        self.bias_v.data = vbias
        self.out_proj_bias.data = obias  

        pass

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, gather_steps=False):
        # type: (Tensor, Tensor, Tensor, dict, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        # apply attn_matrix to our matrices
        d_head = self.head_dim
        d_model = query.shape[-1]

        # this is the path we take for all our experiments
        result = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_q, self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj_weight, self.out_proj_bias,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
            key_padding_mask=key_padding_mask, need_weights=need_weights, training=self.training,
            attn_mask=attn_mask, timer=self.timer, gather_steps=gather_steps, decoder=self.decoder, 
            right_match=self.right_match)

        return result

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_q: Optional[Tensor],
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 timer=None,
                                 gather_steps=False,
                                 decoder=None,
                                 right_match=None,
                                 ):

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    #assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # HERE: apply Q, K, V weights to inputf
    q = linear(query, q_proj_weight, bias_q)
    k = linear(key, k_proj_weight, bias_k)
    v = linear(value, v_proj_weight, bias_v)

    # queryp = decoder.decode_step(query[0,0,:], return_pairs=True)
    # print(queryp)

    # qp = decoder.decode_step(q[0,0,:], return_pairs=True)
    # print(qp)

    # kp = decoder.decode_step(key[0,0,:], return_pairs=True)
    # print(kp)

    # vp = decoder.decode_step(value[0,0,:], return_pairs=True)
    # print(vp)

    debug_print(q, "q")
    debug_print(k, "k")
    debug_print(v, "v")

    use_pasha_max = True   
    if not use_pasha_max:
        q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    # HERE: attn_output_weights = q * v
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    debug_print(attn_output_weights, "BEFORE mask")

    # apply a bunch of masks
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    debug_print(attn_output_weights, "AFTER attn mask")

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    debug_print(attn_output_weights, "AFTER keypadding mask")

    # NORMALIZE attn_output_weights
    if use_pasha_max:
        attn_output_weights2 = pasha_max(attn_output_weights, right_match=right_match)

        # every step should result in at least 1 column having an output
        has_output = not torch.equal(attn_output_weights2, torch.zeros_like(attn_output_weights2))
        #assert has_output, "no columns have an output for this layer"

    else:
        # normal softmax
        attn_temp = 1  # .01   # 1
        attn_output_weights = softmax(attn_output_weights/attn_temp, dim=-1)
        attn_output_weights2 = dropout(attn_output_weights, p=dropout_p, train=training)

    debug_print(attn_output_weights2, "AFTER softmax")

    # HERE: attn_output = attn_output_weights * v
    attn_output = torch.bmm(attn_output_weights2, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, head_dim)    # embed_dim)

    #show_unique_values(attn_output, "attn_output #1")
    #show_unique_values(out_proj_weight.data, "out_proj_weight")

    # experiment: try skipping the linear output projection
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    #show_unique_values(attn_output, "after linear layer")

    # bulid steps dictionary: 5 steps (input, q, k, v, output) + attn_weights
    # NOTE: "residual" is added later to dict (by TransformerEncoderLayer.forward)
    steps = {}    

    if gather_steps:
        # keep values on GPU for fast decoding
        steps["input"] = query.squeeze(1)
        steps["query"] = q.squeeze(0)
        steps["key"] = k.squeeze(0)
        steps["value"] = v.squeeze(0)
        steps["output"] = attn_output.squeeze(1)

        steps_attn_weights = attn_output_weights2.squeeze(0).detach().cpu().numpy()
        steps["attn_weights"] = steps_attn_weights

    # # debug
    # output_values = attn_output.squeeze(1).chunk(4, dim=-1)
    # dummy = 0
    
    need_weights = False

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, steps

def pasha_max(data, threshold=0.95, right_match=False):
    # input shape: [seq_len, bsz=1, d_model] 
    seq_len, bsz, d_model = data.shape

    # reshape to: [seq_len, n_registers, d_register]
    data2 = data

    if right_match:
        # reverse 
        rev_data2 = torch.flip(data2, dims=[-1])
        max_values, rev_max_indices = rev_data2.max(dim=-1)
        max_indices = data2.size(-1) - 1 - rev_max_indices

    else:
        # Get the max values and indices along the last dimension
        max_values, max_indices = data2.max(dim=-1)

    # Create a mask where max values are greater than or equal to 0.95
    mask = max_values >= threshold

    # Create a tensor filled with zeros, with the same shape as your vectors
    one_hot = torch.zeros_like(data2)

    # For each vector in the batch, only create a one-hot if the max is >= 0.95
    one_hot[mask, max_indices[mask]] = 1

    # reshape to input shape: [seq_len, bsz=1, d_model]
    one_hot2 = one_hot

    return one_hot2



def debug_print(attn_output_weights, context):
    #show_unique_values(attn_output_weights, context)
    pass

    # if attn_output_weights.shape[1] == 5:
    #     print("attn_output_weights: {}".format(context))
    #     for aow in attn_output_weights[0]:
    #         aw = aow.detach().numpy()
    #         txt = ", ".join(["{:.2f}".format(z) for z in aw])
    #         print("   ", txt)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class Runner():
    def __init__(self, d_register):
        self.d_register = d_register
        self.device = None
        self.model = None
        self.register_map = None
        self.constant_map = None
        self.register_name_to_index = None
        self.num_registers = 0

    def get_example(self, example, parser, fn_examples):
        import yaml

        with open(fn_examples, "r") as f:
            data = yaml.safe_load(f)
            examples = data["examples"]

        if example in examples:
            example = examples[example]
            prompt = example["prompt"]
            gold = example["gold"]

        else:
            print("example not found: {}".format(example))
            self.usage(parser)
            sys.exit(1)

        return prompt, gold

    def get_all_examples_names(self, fn_examples):
        import yaml

        with open(fn_examples, "r") as f:
            data = yaml.safe_load(f)
            examples = data["examples"]

        return examples.keys()

    def build_fixed_vocab_and_embedding(self, prompt, gold, embeddings_device, d_register):

        # get fixed part of vocab (and its stoi)
        fixed_vocab = build_vocab_from_example(prompt, gold, self.register_map, self.constant_map)

        # make an embedding to match what is used in DatTransformer
        embedding = build_embedding(d_embedding=d_register, max_vocab_len=d_register, device=embeddings_device)

        return fixed_vocab, embedding

    def compile_weights(self, fixed_vocab, embedding, device, d_register):
        from weights_compiler import WeightsCompiler

        # compile generated JSON weights
        compiler = WeightsCompiler(fixed_vocab, embedding, d_register, self.register_name_to_index)
        matrix_layers = compiler.compile_json_weights(self.fn_weights, device)
        print("compiled weights file: {}".format(self.fn_weights))

        # started = time.time()
        # elapsed = time.time() - started
        # print("total compilation time: {:.2f} sec".format(elapsed))

        return matrix_layers

    def run_all_examples(self, parser, fn_examples):
        names = self.get_all_examples_names(fn_examples)
        correct = 0

        for name in names:
            prompt, gold = self.get_example(name, parser, fn_examples)

            match = self.run_example(prompt, gold, name)
            if match:
                correct += 1

        print("\n{} of {} correct".format(correct, len(names)))

    def run_dataset_examples(self, args):
        import datetime
        import socket

        dataset_name, version = args.dataset.split("/")
        task_name = args.task
        split_name = args.split
        max_examples = args.max_examples

        correct = 0

        # print date/time and machine name
        print("time: \t{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("host: \t{}".format(socket.gethostname()))
        print("model: \tDAT Transformer")

        # print dataset, task, split
        print("dataset: \t{}".format(dataset_name))
        print("task: \t{}".format(task_name))
        print("split: \t{}".format(split_name))
        print()

        examples = load_dataset_task_split(dataset_name, version, task_name, split_name)
        print()

        for e, example in enumerate(examples):
            if max_examples and e >= max_examples:
                break

            parts = example.split("\t")
            prompt, gold, info = parts

            example = "#{}, info: {}".format(1+e, info)
            match = self.run_example(prompt, gold, example)
            #print()

            if match:
                correct += 1

        print("\n{} of {} correct".format(correct, e))
    
    def run_example(self, prompt, gold, example):

        started = time.time()

        if not self.register_map:
            # compile PSL program
            fn_weights, num_dat_layers = compile_psl(self.args.psl)

            # open weights file and load data
            weight_layers, register_map, constant_map, register_name_to_index, watch_list, system_map = load_json_weights(fn_weights)

            self.register_map = register_map
            self.constant_map = constant_map
            self.register_name_to_index = register_name_to_index
            self.num_dat_layers = num_dat_layers
            self.fn_weights = fn_weights
            self.system_map = system_map

        needed_d_register = get_needed_d_register_size(prompt, gold, self.register_map, self.constant_map)
        if not self.model:
            # first example loads the model
            self.d_register = max(DEFAULT_D_REG, needed_d_register)
            self.on_d_register_changed(prompt, gold)

        elif needed_d_register > self.d_register:
            # big example: rebuild the model
            self.d_register = needed_d_register  # int(1.2 * needed_d_register)     # give some extra room
            print("large example encountered; increasing d_register to: {}".format(self.d_register))
            self.on_d_register_changed(prompt, gold)

        vocab = build_vocab_from_example(prompt, gold, self.register_map, self.constant_map)
        self.model.load_vocab(vocab)

        gold_tokens = gold.split()
        max_new_tokens = len(gold_tokens)

        y_hat_text, _ = self.model.generate(prompt, self.device, max_new_tokens=max_new_tokens, example=example, tracing_enabled=self.args.trace)
        correct = (y_hat_text == gold)

        print("\nexample: {}".format(example))
        print("  prompt: {}".format(prompt))
        print("  gold:   {}".format(gold))
        print("  y^:     {}".format(y_hat_text))

        feedback = "CORRECT" if correct else "WRONG"
        elapsed = time.time() - started
        print("  [{}]  ({:.2f} secs)".format(feedback, elapsed))

        return correct

    def parse_args(self):
        fn_psl = "psl_programs/icl_parser_gen.yaml"
        prompt = None    # "Q john loves mary A mary hugs john . Q sue loves bill A"
        gold = None      # "bill hugs sue ."

        parser = argparse.ArgumentParser(add_help=False, description="Run a PSL program (interpreted) with a specified template example.")
        parser.add_argument("--psl", type=str, help="specify the PSL program to interpret", default=fn_psl)
        parser.add_argument("--example", type=str, nargs="*", help="name of example to run (or 'all')", default=None)
        parser.add_argument("--examples", type=str, help="specify the YAML file holding the examples", default="examples.yaml")
        parser.add_argument("--max_examples", type=int, help="max examples to test")
        parser.add_argument("--dataset", type=str, help="dataset name", default="nc_tgt/v11")
        parser.add_argument("--task", type=str, help="tgt dataset task name", default="1_shot_rlw")
        parser.add_argument("--split", type=str, help="tgt dataset split name", default=None)
        parser.add_argument("--prompt", type=str, help="prompt for a custom example")
        parser.add_argument("--gold", type=str, help="gold text for a custom example")
        parser.add_argument("--trace", type=int, help="set to 1 to enable tracing", default=0)
        parser.add_argument("--help", action="store_true", help="print information command line arguments", default=0)

        args = parser.parse_args()

        if not args.example:
            names = list(self.get_all_examples_names(args.examples))
            args.example = names[0]
        elif isinstance(args.example, list):
            args.example = args.example[0]

        args.psl = fixup_psl(args.psl)
            
        return args, parser
        
    def usage(self, parser):

            parser.print_help()
            # print("Usage: python dat_transformer.py [ <options> ]")
            # print("   <options>: <example> | all | --tgt <task_name> | --prompt <prompt> --gold <gold> | --fn_psl <psl_file_path> | --trace")

            print()
            print("examples:")
            print("  > python dat_transformer.py                         (run default example on default weights file)")
            print("  > python dat_transformer.py --example=cross_mult    (run example: cross_mult)")
            print("  > python dat_transformer.py --example=all           (run all of examples from examples file)")  
            print("  > python dat_transformer.py --trace                 (generate a trace file while running the default example)")
            print()
            print('  > python dat_transformer.py  --prompt "Q john loves mary A mary hugs john . Q sue loves bill A" --gold "bill hugs sue ."')
            print("  > python dat_transformer.py --dataset nc_tgt/v11 --task 1_shot_rlw")

    def on_d_register_changed(self, prompt, gold):
        load_started = time.time()

        fixed_vocab, embedding = self.build_fixed_vocab_and_embedding(prompt, gold, embeddings_device=self.device, d_register=self.d_register)

        matrix_layers = self.compile_weights(fixed_vocab, embedding, self.device, self.d_register)

        print("loading model...")
        load_started = time.time()

        self.model = DatTransformer(fixed_vocab, embedding, d_register=self.d_register, num_encoder_layers=self.num_dat_layers, log_progress=False, 
            register_name_to_index=self.register_name_to_index, system_map=self.system_map)

        self.model.load_weights(self.d_register, embedding, vocab=fixed_vocab, weights=matrix_layers)
        
        if WEIGHT_TYPE == "direct":
            self.model.to(self.device)

        load_elapsed = time.time() - load_started
        num_params = self.model.get_num_params()

        print("model loaded ({:.2f} secs), d_register: {}, self.num_registers: {}, d_hidden: {:,}, params: {:,}".format(\
            load_elapsed, self.d_register, self.num_registers, self.model.d_model, num_params))

    def main(self):

        args, parser = self.parse_args()
        self.args = args

        # start processing with args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.d_register = DEFAULT_D_REG

        if args.help or args.example in ["-h", "--help", "help"]:
            self.usage(parser)
            sys.exit(0)

        if args.prompt:
            if not args.gold:
                print("missing --gold <gold text>")
                self.usage(parser)
                sys.exit(1)

            # run custom example
            self.run_example(args.prompt, args.gold, "<custom>")

        elif args.split:
                self.run_dataset_examples(args)

        elif args.example == "all":
            self.run_all_examples(parser, args.examples)

        else:
            # run single example
            example = args.example
            prompt, gold = self.get_example(example, parser, args.examples)
            self.run_example(prompt, gold, example)


if __name__ == "__main__":
    started = time.time()
    
    runner = Runner(d_register=DEFAULT_D_REG)
    runner.main()

    elapsed = time.time() - started
    elapsed_text = get_time_str(elapsed)
    
    print("\ntotal elapsed time: {}".format(elapsed_text))

