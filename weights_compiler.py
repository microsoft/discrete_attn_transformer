# weights_compiler.py: compiles JSON weights into weight matrices for DAT transformer

import os
import sys
import json
import math
import time
import torch
import numpy as np
import argparse

from dat_common import *
from utils.dat_utils import *

class WeightsCompiler():
    '''
    vocab: a list of tokens in the fixed part of the vocabulary (predefined compiler symbols)
    embedding: the embedding function that will be used in the DAT transformer with the generated matrix weights
    '''
    def __init__(self, vocab, embeddings, d_register, register_name_to_index):

        print("WeightsCompiler: vocab size: {}, d_register: {}".format(len(vocab), d_register))

        # if WEIGHT_TYPE == "direct":
        #     d_register = 1

        self.register_name_to_index = register_name_to_index
        self.stoi = {word: i for i, word in enumerate(vocab)}

        self.embed_data_type = EMBED_DATA_TYPE
        self.embedding = embeddings

        self.num_heads = 1
        self.d_model = d_register * len(register_name_to_index)
        self.d_register = d_register

    def load_attn_matrix_layer_from_weights(self, layer_dict, weights_dict, layer_index, device):
        ml = {"layer_index": layer_index, "right_match": layer_dict["right_match"], "causal_attn": layer_dict["causal_attn"]}

        # note: our transformer expects symbol @ register[0] and position @ register[1]
        # so we hardcode those positions here
        register_map = self.register_name_to_index

        d_head = self.d_model // self.num_heads
        d_register = self.d_register

        # make the output matrix just pass thru values from the v matrix
        vd = weights_dict["v"]
        weights_dict["output"] = {k:k for k,v in vd.items()} 

        for key in ["q", "k", "v", "output"]:
            matrix = torch.zeros((d_head, d_head))
            bias = torch.zeros(d_head)
            identity_matrix = torch.eye(d_register)

            weights = weights_dict[key]

            if layer_index == 22 and key == "q":
                pass

            positive_match_count, negative_match_count, mat_fixup_list, bias_fixup_list = \
                self.compute_matrix_and_bias(weights, register_map, d_register, matrix, bias, identity_matrix)

            factor = 1
            if key == "output":
                factor = 2
            elif key in ["q", "k"] and positive_match_count > 1:
                factor = 1/math.sqrt(positive_match_count) 

            if WEIGHT_TYPE == "direct":
                ml[key] = (factor, mat_fixup_list, bias_fixup_list)

            else:
                # apply factor now
                matrix = matrix * factor
                bias = bias * factor

                matrix = matrix.to(self.embed_data_type)
                bias = bias.to(self.embed_data_type)

                if device is not None:
                    matrix = matrix.to(device)
                    bias = bias.to(device)
  
                ml[key] = (matrix, bias)

        return ml

    def compute_matrix_and_bias(self, weights, register_map, d_register, matrix, bias, identity_matrix):

        positive_match_count = 0
        negative_match_count = 0
        mat_fixup_list = []
        bias_fixup_list = []

        def add_bias_reg(token_index, dest_start, fixup_type):
            if WEIGHT_TYPE == "direct":
                bias_fixup_list.append((token_index, dest_start, fixup_type))

            else:
                if token_index == "ones":
                    core_bias = torch.ones(d_register)

                elif isinstance(token_index, torch.Tensor):
                    core_bias = token_index

                else:
                    assert(isinstance(token_index, int))
                    core_bias = self.embedding(torch.tensor(token_index))

                if fixup_type == "one_minus":
                    core_bias = 1 - core_bias
                elif fixup_type:
                    raise ValueError("unknown fixup_type: {}".format(fixup_type))

                bias[dest_start:dest_start+d_register] = core_bias

        def add_matrix_reg(mat_type, dest_start, src_start, fixup_type):
            if WEIGHT_TYPE == "direct":
                mat_fixup_list.append((mat_type, dest_start, src_start, fixup_type))

            else:
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

                matrix[dest_start:dest_start+d_register, src_start:src_start+d_register] = core_matrix

        for lhs, rhs in weights.items():
            if isinstance(rhs, int):
                # ensure everything is a string
                rhs = str(rhs)

            weight_func = None
            if "@" in rhs:
                rhs, weight_func = rhs.split("@")

            using_not_equals = False

            if isinstance(rhs, list):     # our original tuple has been converted to a list
                if rhs[0] == "in" or rhs[0] == "not_in":
                    pass
                else:
                    operator, rhs = rhs
                    # we only support NOT equals here
                    assert operator == "!="
                    using_not_equals = True

            dest_start = register_map[lhs] * d_register

            if isinstance(rhs, list):
                # its a CONSTANT LIST of values (put sum of their embedding vectors into bias)
                # IN or NOT IN operator
                sum_embedding = torch.zeros(d_register).type_as(self.embedding.weight)
                const_values = rhs[1:]

                for token in const_values:
                    if token == "BLANK":
                        token = " "
                    token_index = self.stoi[token]
                    tok_core_bias = self.embedding(torch.tensor(token_index))
                    sum_embedding += tok_core_bias

                if rhs[0] == "in":
                    add_bias_reg(sum_embedding, dest_start, None)
                else:
                    add_bias_reg(sum_embedding, dest_start, "one_minus")
                positive_match_count += 1

            elif not rhs in register_map:
                # its a CONSTANT value (put its embedding vector into bias)
                token = " " if rhs == "BLANK" else rhs
                token_index = self.stoi[token]

                if using_not_equals:
                    # match all 1-hots except this one
                    add_bias_reg(token_index, dest_start, "one_minus")
                else:
                    add_bias_reg(token_index, dest_start, None)

                positive_match_count += 1

            else:
                src_start = register_map[rhs] * d_register

                if weight_func:
                    # reference to a function (e.g. "pos_decrement")
                    if using_not_equals:
                        # match all 1-hots except this one
                        add_matrix_reg(weight_func, dest_start, src_start, "one_minus")
                    else:
                        add_matrix_reg(weight_func, dest_start, src_start, None)

                    positive_match_count += 1

                    # # debug: quick test of matrix
                    # x_embed = self.embed(torch.tensor(2))    # token index of "1"
                    # value = torch.matmul(x_embed, trained_matrix)
                    # print(value)

                else:
                    # use identity matrix (put into matrix)
                    if using_not_equals:
                        # we want "1-contents(reg)" to match everything but contents(reg)
                        # flip the sign of the identity matrix
                        add_matrix_reg("identity", dest_start, src_start, "minus")

                        # set the bias to the value of 1 (not the token)
                        add_bias_reg("ones", dest_start, None)
                        negative_match_count += 1
                    else:
                        add_matrix_reg("identity", dest_start, src_start, None)
                        positive_match_count += 1

        return positive_match_count, negative_match_count, mat_fixup_list, bias_fixup_list

    def compile_json_weights(self, fn_weights, device=None):
        matrix_layers = []
        repeat_group = 1

        if not os.path.exists(fn_weights):
            raise Exception("json weights file not found: {}".format(fn_weights))

        with open(fn_weights, "rt") as file:
            text = file.read()
            data = json.loads(text)
            layer_weights = data["weights"]

        layer_index = 0
        for layer_dict in layer_weights:

            weights = layer_dict["weights"]

            # support for multi-layer repetition
            if "until" in layer_dict:
                # flatten out nested layers, adding repeat_group info to each layer

                for ld in weights:
                    tf_layer = self.load_attn_matrix_layer_from_weights(ld, ld["weights"], layer_index, device)
                    tf_layer["repeat_group"] = repeat_group
                    tf_layer["repeat_layer_count"] = len(weights)
                    tf_layer["until"] = layer_dict["until"]

                    matrix_layers.append(tf_layer)
                    layer_index += 1

                repeat_group += 1

            else:
                tf_layer = self.load_attn_matrix_layer_from_weights(layer_dict, weights, layer_index, device)
                matrix_layers.append(tf_layer)
                layer_index += 1

        return matrix_layers

    def save_weights(self, new_layers, fn_source):
        # save new layers to file
        base_name = os.path.basename(fn_source).split(".")[0]
        fn = "dat_weights/{}.pt".format(base_name)

        os.makedirs("dat_weights", exist_ok=True)
        torch.save(new_layers, fn)

        return fn

def parse_args():
    fn_psl = "psl_programs/icl_parser_gen.yaml"

    parser = argparse.ArgumentParser(add_help=False, description="Compile weights for the specified PSL program.")
    parser.add_argument("--psl", type=str, help="specify the PSL program to be compiled", default=fn_psl)
    parser.add_argument("--help", action="store_true", help="print information command line arguments", default=0)

    args = parser.parse_args()

    args.psl = fixup_psl(args.psl)
    return args, parser

def usage(parser):
    #print("usage: python weights_compiler.py <json_weights_file>")
    parser.print_help()

    print()
    print("examples: ")
    print("  > python weights_compiler.py  (compile weights for deault psl program)")
    print("  > python weights_compiler.py  --psl my_test_program.yaml")
    print("  > python weights_compiler.py  --help (print this help)")

if __name__ == "__main__":  

    args, parser = parse_args()

    if args.help:
        usage(parser)
        sys.exit(0)

    started = time.time()
    fn_psl = args.psl
    print("compiling json weights from: {}".format(fn_psl))

    # compile PSL program
    fn_weights, num_dat_layers = compile_psl(fn_psl)

    # open weights file and load data
    weight_layers, register_map, constant_map, register_name_to_index, _, system_map = load_json_weights(fn_weights)

    # get fixed part of vocab (and its stoi)
    vocab = build_vocab_from_example(prompt="", gold="", register_map=register_map, constant_map=constant_map)

    # make an embedding to match what is used in DatTransformer
    d_register = DEFAULT_D_REG
    embedding = build_embedding(d_embedding=d_register, max_vocab_len=d_register)

    compiler = WeightsCompiler(vocab, embedding, d_register, register_name_to_index)
    matrix_layers = compiler.compile_json_weights(fn_weights)
    fn_saved = compiler.save_weights(matrix_layers, fn_weights)

    elapsed = time.time() - started

    size = os.path.getsize(fn_saved)
    print("compile completed, elapsed: {:.2f} secs, weight matrices saved to: {} (size: {:,} bytes)".format(elapsed, fn_saved, size))





