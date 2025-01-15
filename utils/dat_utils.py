# utils.py: utility functions for working with DAT transformer
import os
import json
import torch
import numpy as np
from dat_common import *

def build_vocab_from_example(prompt, gold, register_map, constant_map):
    prompt_tokens = prompt.split()
    gold_tokens = gold.split()

    # filter out any numbers (they will be added in next step, to keep them contiguous, so decrement function works correctly)
    constant_tokens = [cv for cv in constant_map.values() if not cv.isnumeric()]
    constant_tokens.append("EOP")   # add EOP to constant tokens (reserved by interpreter/transformer)

    vocab = [" "]                                           # zero-valued embeddings will decode to the space token     
    vocab += constant_tokens                                # add next (this represents the fixed part of the vocab)

    max_cols = max(2, len(prompt_tokens) + len(gold_tokens))
    vocab += [str(i) for i in range(max_cols+2)]            # make positions sequential (0-max_cols) and 1 more (for pos_increment use)
    vocab += prompt_tokens

    # remove any duplicates in vocab without changing order
    seen = set()
    vocab = [x for x in vocab if not (x in seen or seen.add(x))]

    # stoi = {word: i for i, word in enumerate(vocab)}
    # itos = {i: word for i, word in enumerate(vocab)}

    return vocab   # , stoi, itos, prompt_tokens, gold_tokens

def get_needed_d_register_size(prompt, gold, register_map, constant_map):
    '''
    should match build_vocab_from_example
    '''
    vocab = build_vocab_from_example(prompt, gold, register_map, constant_map)
    d_reg = len(vocab)
    return d_reg

def build_embedding(d_embedding, max_vocab_len, device=None):
    embed_type = "one_hot" 

    if embed_type == "one_hot":
        # create a 1-hot embedding matrix 
        assert d_embedding >= max_vocab_len
        embedding_weight = torch.eye(d_embedding).type(EMBED_DATA_TYPE)

        if device:
            embedding_weight = embedding_weight.to(device)

        # make it look like a pytorch embedding function
        embed = lambda x: embedding_weight[x]
        embed.weight = embedding_weight     

    else:
        raise ValueError("unknown embed_type: {}".format(embed_type))

    return embed

def embedding_to_indexes(value, einsum_formula, embedding):
    # its a one-hot or dense embedding
    logits = torch.einsum(einsum_formula, value, embedding.weight)
    indexes = torch.argmax(logits, dim=-1)

    return indexes


def get_symbol_embedding(embedding, stoi, symbol):
    index = stoi[symbol]
    return get_index_embedding(embedding, index)

def get_index_embedding(embedding, index):
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index)

    tensor = index.type_as(embedding.weight).type(torch.long)
    return embedding(tensor)

def get_dataroot(path=None):
    if not path or path == "$DATAROOT":
        dp_default = os.path.expanduser('~/.data')
        path = os.getenv('DATAROOT', dp_default)
        
    return path

def show_unique_values(data, name):
    unique_values = torch.unique(data).tolist()
    print("unique values in {}: {}".format(name, unique_values))


def get_decrement_matrix(d_register):
    decrement_matrix = torch.tensor(np.eye(d_register, d_register, k=1))
    return decrement_matrix
        
def get_increment_matrix(d_register):
    decrement_matrix = torch.tensor(np.eye(d_register, d_register, k=-1))
    return decrement_matrix
        
def compile_psl(fn_psl):
    from psl_compiler import PslCompiler

    show_progress = False
    compiler = PslCompiler(show_progress=show_progress)

    layers = compiler.compile(fn_psl)
    weights, num_dat_layers = compiler.generate_weights(layers, print_them=show_progress)

    fn_weights = compiler.save_weights(weights, fn_psl)
    return fn_weights, num_dat_layers

def load_json_weights(fn_weights):
    with open(fn_weights, "r") as f:
        data = json.load(f)

        weight_layers = data["weights"]
        register_map = data["register_map"]
        constant_map = data["constant_map"]
        register_name_to_index = {k: i for i, k in enumerate(register_map.values())}     
        watch_list = data["watch_list"]
        system_map = data["system_map"]

    return weight_layers, register_map, constant_map, register_name_to_index, watch_list, system_map

def fixup_psl(fn_psl):

    # if fn_psl is not found and doesn't specifiy a directory, try adding psl_programs/ to the front
    if not os.path.exists(fn_psl) and not "/" in fn_psl:
        fn_temp = os.path.join("psl_programs", fn_psl)
        if os.path.exists(fn_temp):
            fn_psl = fn_temp

    return fn_psl

def load_dataset_task_split(dataset_name, version, task_name, split_name):
    # this is the only dataset currently supported
    assert dataset_name == "nc_tgt"
    assert version == "v11"

    # load dataset and subset (task) from HuggingFace
    repo_id = "rfernand/templatic_generation_tasks"
    from datasets import load_dataset

    dataset = load_dataset(repo_id, task_name)
    examples = dataset[split_name]

    # convert from HF labels to tab separated format
    examples = [f"{part0}\t{part1}\t{part2}" for part0, part1, part2 in zip(examples["text"], examples["label"], examples["annotation"])]

    return examples

def get_time_str(value):
    if value < 60:
        return f"{value:.2f} secs"
    elif value < 3600:
        return f"{value/60:.2f} mins"
    else:
        return f"{value/3600:.2f} hrs"
    