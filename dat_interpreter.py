# dat_interpreter.py: interpret a DAT program thru its JSON weights file

import sys
import json
import time
import yaml
import argparse

from utils.dat_utils import *
from utils.trace_generator import TraceGenerator

class DatInterpreter():
    '''
    The interpreter's primary data structure is called "registers" and consists of a list of register dictionaries for each column.  Within a register
    dict, the keys are the register names and the values are the stoi[] number for the symbol in the register.  For example, here
    is the value of the input registers fed into the first layer of the interpreter:
        [
            {'s': 35, 'p': 2, 'a': 2}
            {'s': 36, 'p': 3, 'a': 2}
            {'s': 37, 'p': 4, 'a': 2}
            {'s': 38, 'p': 5, 'a': 2}
            {'s': 39, 'p': 6, 'a': 2}
            {'s': 38, 'p': 7, 'a': 2}
            {'s': 40, 'p': 8, 'a': 2}
            {'s': 36, 'p': 9, 'a': 2}
            {'s': 41, 'p': 10, 'a': 2}
            {'s': 35, 'p': 11, 'a': 2}
            {'s': 42, 'p': 12, 'a': 2}
            {'s': 37, 'p': 13, 'a': 2}
            {'s': 43, 'p': 14, 'a': 2}
            {'s': 39, 'p': 15, 'a': 2, 'z': 34}
        ]
    '''
    def __init__(self, fn_psl, log=False, caching_enabled=False):
        self.max_cols = None
        self.prompt_len = None
        self.vocab = None
        self.stoi = None
        self.itos = None
        self.log = log
        self.num_layers = 0
        self.caching_enabled = caching_enabled

        # needed for dat_explorer.py interaction
        self.decoder = self

        # compile PSL program
        fn_weights, num_dat_layers = compile_psl(fn_psl)

        # open weights file and load data
        weight_layers, register_map, constant_map, register_name_to_index, watch_list, system_map = load_json_weights(fn_weights)

        self.weight_layers = weight_layers
        self.register_map = register_map
        self.constant_map = constant_map
        self.system_map = system_map

        # get system register names
        self.sym_reg_name = system_map["symbol"] if "symbol" in system_map else None
        self.pos_reg_name = system_map["position"] if "position" in system_map else None
        self.parse_reg_name = system_map["parse"] if "parse" in system_map else None
        self.eop_reg_name = system_map["eop"] if "eop" in system_map else None
        self.out_reg_name = system_map["output"] if "output" in system_map else None
       
        # # display all layers
        # for i, layer in enumerate(self.weight_layers):
        #     comment = layer["layer_comment"]
        #     weights = layer["weights"]
        #     print("Layer {}\n  {}\n  {}".format(i, comment, weights))
        #     a = 9
        pass

    def build_vocab(self, prompt, gold):
        vocab = build_vocab_from_example(prompt, gold, self.register_map, self.constant_map)
        prompt_tokens = prompt.split()
        gold_tokens = gold.split()

        self.max_cols = len(prompt_tokens) + len(gold_tokens)
        self.prompt_len = len(prompt_tokens)

        self.vocab = vocab
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.max_gen_cols = len(gold_tokens)

    def load_vocab(self, vocab):
        # called from dat_explorer.py
        self.vocab = vocab
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self.itos = {i: word for i, word in enumerate(vocab)}
        #self.max_gen_cols = len(gold_tokens)

    def to(self, device):
        # called from dat_explorer.py
        return self

    def build_input_embeddings(self, prompt):

        prompt_tokens = prompt.split()
        pt_len = len(prompt_tokens)
        registers = [{} for i in range(pt_len)]   # initialize registers for each column

        for c, rd in enumerate(registers):
            pt = prompt_tokens[c]

            if self.sym_reg_name:
                rd[self.sym_reg_name] = self.stoi[pt]             # set symbol register to prompt token

            if self.pos_reg_name:
                rd[self.pos_reg_name] = self.stoi[str(c + 1)]     # set position register to 1-based position

            if self.parse_reg_name:
                rd[self.parse_reg_name] = self.stoi["1"]            # set parse register to 1"

            if c == pt_len - 1 and self.eop_reg_name:
                rd[self.eop_reg_name] = self.stoi["EOP"]       # set z register to EOP on last token

        return registers

    def apply_weights(self, weights, registers):
        num_cols = len(registers)
        new_registers = [{} for i in range(num_cols)]   # initialize registers for each column

        # process each column (of the current layer)
        for src_rd, dest_rd in zip(registers, new_registers):

            # process each weight (specified by a source and dest register pair)
            # here, we interpret various special cases, such as register expressions, constants, and list comparisons
            # constants are converted to vocab indexes
            for dest, src in weights.items():
                # NOTE: SRC WEIGHT can be any one of the following:
                #  - constant (string)
                #  - register_name 
                #  - register_name@pos_increment
                #  - register_name@pos_decrement
                # - [ "!=", register_name ]
                # - [ "!=" constant ]
                # - [ "in", "constant1", "constant2", ... ]
                # - [ "not_in", "constant1", "constant2", ... ]

                if isinstance(src, list):
                    if src[0] == "!=":
                        # src is an comparison operator (!=), followed by a constant or register
                        assert len(src) == 2
                        neq_src = src[1]

                        if neq_src[0].islower():
                            # src is a register
                            # new: now we have to ensure that the register is in the source registers
                            if neq_src in src_rd:
                                dest_rd[dest] = -src_rd[neq_src]
                            else:
                                #match_None = ["in", 0]
                                dest_rd[dest] = -0.0    # match_None
                        else:
                            dest_rd[dest] = -self.stoi[neq_src]     # cheap interperter trick to make != work

                    elif src[0] == "in" or src[0] == "not_in": 
                        # src is a list of constant values
                        new_src = [src[0]]  # copy the operator

                        for s in range(1, len(src)):
                            # convert each entry from string to vocab index
                            symbol_name = src[s]
                            symbol_value = symbol_name     # self.constant_map[symbol_name]
                            value = self.stoi[symbol_value]
                            new_src.append(value)

                        dest_rd[dest] = new_src  

                    else:
                        # list has already been converted to vocab indexes
                        dest_rd[dest] = src
                
                elif src[0].islower():
                    # src is a register expression
                    if "@" in src:
                        # src: register@func
                        src, func = src.split("@")
                        
                        if func == "pos_decrement":
                            dest_rd[dest] = src_rd[src] - 1
                        elif func == "pos_increment":
                            dest_rd[dest] = src_rd[src] + 1
                        else:
                            raise ValueError("unknown function: {}".format(func))

                    else:
                       # src is register name
                        dest_rd[dest] = src_rd[src] if src in src_rd else 0

                else:
                    # src is a constant
                    symbol_value = src    #self.constant_map[src]
                    value = self.stoi[symbol_value]
                    dest_rd[dest] = value

        return new_registers

    def query_key_match(self, qd, kd):
        # for each key in query dict, immediately return False if NO MATCH is found

        # NOTE: REGISTERS normally just contain constant vocab indexes
        # but query and key registers can also contain any of the following:
        # - negative vocab index (to indicate NOT EQUALS comparison)
        # - [ "in", "constant1 vocab index", "constant2 vocab index", ... ]
        # - [ "not_in", "constant1 vocab index", "constant2 vocab index", ... ]

        for key, value in qd.items():
            assert not isinstance(key, list)

            if isinstance(key, int) and key < 0:
                pass

            if (not key in kd):   #  or kd[key] != value:
                return False

            kd_entry = kd[key]

            #assert not isinstance(value, list)
            if isinstance(value, list):
                which = value[0]
                found = (kd_entry in value[1:])
                if which == "not_in":
                    found = not found
                if not found:
                    return False
                continue

            # list is the "in" operator
            if isinstance(kd_entry, list):
                if kd_entry[0] == "not_in":
                    found = (not value in kd_entry[1:])
                else:
                    found = (value in kd_entry)
                if not found:
                    return False
                continue

            # negative value means do NOT equals comparison
            if value < 0 or kd_entry < 0:
                found = abs(value) != abs(kd_entry)
                if not found:
                    return False
                continue

            elif kd_entry != value:
                return False    

        # only return True after checking all keys in qd
        return True

    def apply_attn_to_values(self, q_registers, k_registers, v_registers, causal_attn, right_match, current_layer_cache=None, layer_index=None):
        num_cols = len(q_registers)
        new_registers = [{} for i in range(num_cols)]   # initialize registers for each column
        attn_weights_by_col = []
        next_cache_index = 0
        skip_each_col_processing = False

        if current_layer_cache != None:
            if len(current_layer_cache) == num_cols:
                # we are REPEATING the last layer to log layers, steps, & registers
                for col in range(num_cols):
                    col_weights = np.zeros(num_cols)                   # currently, we don't cache this, so don't use tracing with caching
                    attn_weights_by_col.append(col_weights)
                    new_registers[col] = dict(current_layer_cache[col])
                    skip_each_col_processing = True
                next_cache_index = num_cols    # filled the new_registers[] with cache

            elif len(current_layer_cache) == num_cols-1:
                # use cache to set set new_registers for all but last column
                for col in range(num_cols-1):
                    col_weights = np.zeros(num_cols)                   # currently, we don't cache this, so don't use tracing with caching
                    attn_weights_by_col.append(col_weights)
                    new_registers[col] = dict(current_layer_cache[col])
                    
                col_weights = np.zeros(num_cols)
                attn_weights_by_col.append(col_weights)
                self.process_attn_current_column(num_cols-1, q_registers, k_registers, v_registers, new_registers, causal_attn, right_match, col_weights, layer_index=layer_index)

                next_cache_index = num_cols-1
                skip_each_col_processing = True

        if not skip_each_col_processing:
            # process each column (no caching)
            for col in range(num_cols):
                col_weights = np.zeros(num_cols)
                attn_weights_by_col.append(col_weights)
                
                self.process_attn_current_column(col, q_registers, k_registers, v_registers, new_registers, causal_attn, right_match, col_weights, layer_index=layer_index)

        if current_layer_cache != None:
            # add newly processed columns to the cache
            #current_layer_cache += new_registers[next_cache_index:]
            for col in range(next_cache_index, num_cols):
                current_layer_cache.append(dict(new_registers[col]))
                #assert len(current_layer_cache) <= num_cols

            assert len(current_layer_cache) == num_cols

        return new_registers, attn_weights_by_col

    def process_attn_current_column(self, col, q_registers, k_registers, v_registers, new_registers, causal_attn, right_match, col_weights, layer_index=None):
       
        num_cols = len(q_registers)

        if causal_attn is None:
            # legacy mode: causal attn turns on when generating new cols
            is_generating = (num_cols > self.prompt_len)
            x_limit = col if is_generating else num_cols-1
            x_range = list(range(0, 1+x_limit))

        else:
            # normal mode - controlled by directives
            if causal_attn == "true":
                x_range = list(range(0, 1+col))
            elif causal_attn == "false":
                x_range = list(range(0, num_cols))
            elif causal_attn == "reverse":
                x_range = list(range(col, num_cols))
            else:
                raise ValueError("unknown causal_attn value: {}".format(causal_attn))

        # left or right match?
        if right_match:
            x_range = list(reversed(x_range))

        if layer_index == 26 and col == 20: # L27:C21
            pass

        for xcol in x_range:
            # process attn for the current column, using all input columns from previous layer
            if self.query_key_match(q_registers[col], k_registers[xcol]):
                new_registers[col] = v_registers[xcol]
                if right_match:
                    col_weights[num_cols - (1+xcol)] = 1
                else:
                    col_weights[xcol] = 1
                break

    def add_register_to_stream(self, registers, stream):
        num_cols = len(registers)
        is_generating = (num_cols > self.prompt_len)

        # NOTE: previously, we only updated the last column when generating new columns
        # but this causes problems, since we don't save every layer of every previous column.
        # the DatTransformer claims it only processes the last column, but this is currently unverified.

        if False:    # is_generating:  
            # just update the last column 
            reg_rd = registers[-1]
            stream_rd = stream[-1]
            for key, value in reg_rd.items():
                stream_rd[key] = value

        else:
            # parsing; update all columns
            for reg_rd, stream_rd in zip(registers, stream):
                for key, value in reg_rd.items():
                    stream_rd[key] = value

    def process_layer(self, registers, weights, layer_index, causal_attn, right_match, nested=False, capture_steps=False, current_layer_cache=None):

        #print("processing layer: {}, causal_attn: {}, right_match: {}".format(1+layer_index, causal_attn, right_match))

        steps = {}
        #registers = self.copy_registers(registers)

        if capture_steps:
            steps["input"] = self.copy_registers(registers)

        q_registers = self.apply_weights(weights["q"], registers)
        k_registers = self.apply_weights(weights["k"], registers)
        v_registers = self.apply_weights(weights["v"], registers)

        o_registers, attn_weights_by_col = self.apply_attn_to_values(q_registers, k_registers, v_registers, causal_attn, right_match, current_layer_cache=current_layer_cache, layer_index=layer_index)

        if self.caching_enabled:
            # update cache for the current layer
            current_layer_cache = o_registers

        self.add_register_to_stream(o_registers, registers)

        if self.log:
            if layer_index > -1:    # final output layer
                for c in range(0, len(registers)):
                    self.decode_and_print_dict(nested, layer_index, c, registers[c])
                print()

        if layer_index == 26 and len(attn_weights_by_col) > 20:   # L27-C21
            weights = attn_weights_by_col[20]
            weights_sum = weights.sum()
            if weights_sum > 0:
                pass

        if capture_steps:
            steps["query"] = q_registers
            steps["key"] = k_registers
            steps["value"] = v_registers
            steps["output"] = o_registers
            steps["residual"] = self.copy_registers(registers)
            steps["attn_weights"] = attn_weights_by_col
            steps["layer_index"] = layer_index

        return registers, steps

    def decode_and_print_dict(self, nested, layer_index, col_index, rd):
        title = "L{}.C{}".format(1+layer_index, 1+col_index)
        nest_indent = "  " if nested else ""
        print("{}{}: ".format(nest_indent, title) + "  ".join(["{}:'{}'".format(k, self.itos[v]) for k, v in rd.items()]))

    def run(self, prompt, gold, example, tracing_enabled=False):
        started = time.time()
        self.build_vocab(prompt, gold)

        print("  prompt: {}".format(prompt))
        print("  gold:   {}".format(gold))

        generated_text, curr_gen_steps = self.generate(prompt, device=None, max_new_tokens=self.max_gen_cols, example=example, return_last_steps=True, tracing_enabled=tracing_enabled)

        elapsed = time.time() - started
        generated_text = generated_text.strip()

        print("  y^:     {}".format(generated_text))

        correct = (gold == generated_text)
        feedback = "CORRECT" if correct else "WRONG"

        print("  [{}]  ({:.2f} secs)".format(feedback, elapsed))
        self.generated_text = generated_text

        return correct
    
    def count_all_layers(self, all_weights):
        outer_count = 0
        inner_count = 0

        for layer_weights in all_weights:
            weights = layer_weights["weights"]
            if isinstance(weights, dict):
                outer_count += 1
            elif isinstance(weights, list):
                inner_count += len(weights)
            else:
                raise ValueError("unknown weights type: {}".format(type(weights)))
            
        count = outer_count + inner_count
        return count

    def generate(self, prompt, device, max_new_tokens, example=None, return_last_steps=False, tracing_enabled=None):

        input_registers = self.build_input_embeddings(prompt)
        generated_text = ""
        curr_gen_steps = None
        trace_generator = None
        total_num_layers = self.count_all_layers(self.weight_layers)

        prompt_tokens = prompt.split()
        self.prompt_len = len(prompt_tokens)
        output_layers_cache = None

        if self.caching_enabled:
            # create an empty column list for each layer
            output_layers_cache = [[] for i in range(total_num_layers)]

        # uncomment to enable tracing in IDE
        #tracing_enabled = True
        if tracing_enabled:
            trace_generator = TraceGenerator(example, self.decoder, all_gens=False) 

        for g in range(max_new_tokens):
            #irx = input_registers    # self.copy_registers(input_registers)
            irx = self.copy_registers(input_registers)

            num_cols = len(irx)
            #print("num_cols: {}".format(num_cols))

            # the CORE call
            last_registers, curr_gen_steps = self.forward(irx, prompt, capture_steps=tracing_enabled, output_layers_cache=output_layers_cache)

            output_rd = last_registers[-1]

            if self.pos_reg_name:
                output_rd[self.pos_reg_name] = self.stoi[str(len(input_registers) + 1)]     # set position register to 1-based position

            if self.log:
                self.decode_and_print_dict(False, self.num_layers-1, len(last_registers)-1, output_rd)

            token = self.itos[output_rd[self.out_reg_name]]
            generated_text += token + " "
            last_step = (token == "." or g == max_new_tokens-1)

            if last_step and return_last_steps and not curr_gen_steps:
                # call again to get the last generation steps
                irx = self.copy_registers(input_registers)

                # we can now handle repeating the last layer, with respect to the cache
                _, curr_gen_steps = self.forward(irx, prompt, capture_steps=True, output_layers_cache=output_layers_cache)   

            # OK to update input_registers now
            input_registers.append(output_rd)

            if trace_generator:
                trace_generator.trace(g, curr_gen_steps)

            if last_step:
                break

        if trace_generator:
            trace_generator.close()

        generated_text = generated_text.strip()
        return generated_text, curr_gen_steps

    def copy_registers(self, registers):
        return [dict(rd) for rd in registers]

    def compare_registers(self, registers1, registers2):
        for rd1, rd2 in zip(registers1, registers2):
            if rd1 != rd2:
                return False

        return True


    def decode_steps(self, value, layer_index):
        '''
        decode each of the steps for a column within an interpreter layer.
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

    def decode_step(self, reg_value, layer_index=None, step_name=None, return_pairs=False, permute=False):
        '''
        decodes a "step" for a column within a transformer layer (steps: input, query, key, value, output, residual).  
            reg_value: [bsz, 1, d_model]
            layer_index: index of the layer (for debugging)
            step_name: name of the step (for debugging)
            return_pairs: return as a dict of register names/values (vs. list of values)
            permute: permute reg_value to "batch first" format before decoding
        '''
        def text_value(value):
            if isinstance(value, list):
                # NOTE: the interpreter processes the JSON weights, so all lists of constants start with "in" or "not_in"
                if value[0] == "not_in":
                    value = "not_in [{}]".format(", ".join(self.vocab[v] for v in value[1:]))
                elif value[0] == "in":
                    value = "in [{}]".format(", ".join(self.vocab[v] for v in value[1:]))
                else:
                    raise ValueError("unknown list value: {}".format(value))
            else:
                if value < 0:
                    value = "!={}".format(self.vocab[abs(value)])
                else:
                    value = self.vocab[value] 
            return value

        register_list = list(self.register_map.values())

        # expand to each register position using text_value() to convert irregular encodings
        text_values = [text_value(reg_value[reg]) if reg in reg_value else " " for reg in register_list]

        if return_pairs:
            #short_register_names = list(self.register_name_to_index.keys())
            num_registers = len(register_list)
            text_values = {register_list[i]: text_values[i].strip() for i in range(num_registers) if text_values[i].strip()}

        return text_values

    def forward(self, input_registers, prompt, capture_steps, output_layers_cache=None):

        # NOTE: layer_index is a unique id for each OUTER or INNER (repeated) layer
        # do NOT use it as an index into the weight_layers list
        layer_index = 0   

        gen_steps = []
        current_layer_cache = None
        #print("last col: {}".format(len(input_registers)-1))

        if self.log:
            print("first 5 input registers:")
            for decode_col in range(5):
                self.decode_and_print_dict(registers, layer_index, decode_col, registers[decode_col])

        for layer_weights in self.weight_layers:
            # don't modify the previous layer's registers
            registers = input_registers   #self.copy_registers(input_registers)

            comment = layer_weights["layer_comment"]
            weights = layer_weights["weights"]

            if isinstance(weights, dict):
                # process simple layer
                #print("SIMPLE layer_index: {}".format(layer_index))

                if self.log:
                    print("processing simple layer: {}".format(1+layer_index))   # 1-relative reporting here 
                    print("  {}".format(comment))
                    print("  {}".format(weights))

                current_layer_cache = output_layers_cache[layer_index] if output_layers_cache else None

                causal_attn = layer_weights["causal_attn"] 
                right_match = layer_weights["right_match"] 

                # turn off caching for causal_attn layers that are false or reverse
                if causal_attn != "true":
                    self.current_layer_cache = None

                registers, steps = self.process_layer(registers, weights, layer_index, causal_attn, right_match, capture_steps=capture_steps, current_layer_cache=current_layer_cache)
                layer_index += 1
                if capture_steps:
                    gen_steps.append(steps)

            else:
                # processing REPEAT block of layers
                block_layers = weights
                repeat_count = 0

                if self.log:
                    print("processing BLOCK layers")
                #print("===== start repeat ========")

                while True:
                    prev_output = self.copy_registers(registers)
                    group_steps = []

                    for b, block_layer in enumerate(block_layers):
                        weights = block_layer["weights"]
                        comment = block_layer["layer_comment"]

                        causal_attn = block_layer["causal_attn"] 
                        right_match = block_layer["right_match"] 

                        if self.log:
                            print("  processing INNER layer: {}".format(1+layer_index+b))
                            print("    {}".format(comment))
                            print("    {}".format(weights))

                        if output_layers_cache:
                            # for repeated inner blocks, we maintain a dynamic list of separate caches, by repeat_count
                            repeat_layer_cache = output_layers_cache[b+layer_index] 

                            # grow if needed
                            if len(repeat_layer_cache) < repeat_count+1:
                                repeat_layer_cache.append([])

                            current_layer_cache = repeat_layer_cache[repeat_count] 

                            # turn off caching for causal_attn layers that are false or reverse
                            if causal_attn != "true":
                                self.current_layer_cache = None

                        #print("REPEAT layer_index: {}".format(b+layer_index))

                        registers, steps = self.process_layer(registers, weights, layer_index+b, causal_attn, right_match, nested=True, capture_steps=capture_steps, current_layer_cache=current_layer_cache)
                        if capture_steps:
                            group_steps.append(steps)

                    repeat_count += 1

                    if self.compare_registers(registers, prev_output):
                        # no change, so break out of loop
                        break

                layer_index += len(block_layers)
                #print("--------------")

                if capture_steps:
                    # just capture final pass of group steps
                    gen_steps += group_steps

                if self.log:
                    print("---- end nesting ----")
        
        self.num_layers = layer_index

        return registers, gen_steps

def get_example(example, parser, fn_examples):
    with open(fn_examples, "r") as f:
        data = yaml.safe_load(f)
        examples = data["examples"]

    if example in examples:
        example = examples[example]
        prompt = example["prompt"]
        gold = example["gold"]

    else:
        print("example not found: {}".format(example))
        usage(parser)
        sys.exit(1)

    return prompt, gold

def get_all_examples_names(fn_examples):
    with open(fn_examples, "r") as f:
        data = yaml.safe_load(f)
        examples = data["examples"]

    return examples.keys()

def usage(parser):
        # print("Usage: python dat_interpreter.py [ <options> ]")
        # print("   <options>: <example> | all | --tgt <task_name> | --prompt <prompt> --gold <gold> | --program_name <psl_file>")
        parser.print_help()

        print()
        print("examples:")
        print("  > python dat_interpreter.py                         (run default example on default weights file)")
        print("  > python dat_interpreter.py --example=cross_mult    (run example: cross_mult)")
        print("  > python dat_interpreter.py --example=all           (run all of examples from examples file)")  
        print('  > python dat_interpreter.py  --prompt "Q john loves mary A mary hugs john . Q sue loves bill A" --gold "bill hugs sue ."')
        print("  > python dat_interpreter.py --dataset=nc_tgt/v11 --task 1_shot_rlw --split ood_lexical")

def parse_args():
    fn_psl = "psl_programs/icl_parser_gen.yaml"

    parser = argparse.ArgumentParser(add_help=False, description="Interpret the specified PSL program with a specified template example.")
    parser.add_argument("--psl", type=str, help="specify the PSL file to compile and interpret", default=fn_psl)
    parser.add_argument("--example", type=str, nargs="*", help="name of example to run (or 'all')", default=None)
    parser.add_argument("--examples", type=str, help="specify the YAML file holding the examples", default="examples.yaml")
    parser.add_argument("--max_examples", type=int, help="max examples to test")
    parser.add_argument("--dataset", type=str, help="dataset name", default="nc_tgt/v11")
    parser.add_argument("--task", type=str, help="tgt dataset task name", default="1_shot_rlw")
    parser.add_argument("--split", type=str, help="tgt dataset split name", default=None)
    parser.add_argument("--caching", type=int, help="set =1 to enable layer caching", default=0)
    parser.add_argument("--prompt", type=str, help="prompt for a custom example")
    parser.add_argument("--gold", type=str, help="gold text for a custom example")
    parser.add_argument("--trace", type=int, help="set to 1 to enable tracing", default=0)
    parser.add_argument("--help", action="store_true", help="print information command line arguments", default=0)

    args = parser.parse_args()

    if not args.example:
        names = list(get_all_examples_names(args.examples))
        args.example = names[0]
    elif isinstance(args.example, list):
        args.example = args.example[0]

    args.psl = fixup_psl(args.psl)

    return args, parser

def run_all_examples(interpreter, parser, fn_examples):
    names = get_all_examples_names(fn_examples)
    correct = 0

    for name in names:
        prompt, gold = get_example(name, parser, fn_examples)
        print("\nexample: {}".format(name))
        match = interpreter.run(prompt, gold, name)
        if match:
            correct += 1

    print("\n{} of {} correct".format(correct, len(names)))

def run_dataset_examples(interpreter, args):
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
    print("model: \tDAT Interpreter")

    # print dataset, task, split
    print("dataset: \t{}".format(dataset_name))
    print("task: \t{}".format(task_name))
    print("split: \t{}".format(split_name))
    print()

    data_root = get_dataroot("$DATAROOT")
    data_dir = get_path_to_task_files(data_root, dataset_name, version, task_name, ok_to_download=True, xt=None)
    fn = os.path.join(data_dir, "{}.xy".format(split_name))

    with open(fn, "rt") as infile:
        examples = infile.read().split("\n")

    for e, example in enumerate(examples):
        if max_examples and e >= max_examples:
            break

        parts = example.split("\t")
        prompt, gold, info = parts

        print("example #{}, info: {}".format(1+e, info))
        example = "tgt_{}_{}_{}".format(task_name, split_name, e)
        match = interpreter.run(prompt, gold, example, tracing_enabled=args.trace)
        print()

        if match:
            correct += 1

    print("\n{} of {} correct".format(correct, e))

def main():

    started = time.time()
    args, parser = parse_args()

    if args.help:
        usage(parser)
        sys.exit(0)

    interpreter = DatInterpreter(args.psl, log=False, caching_enabled=args.caching)

    print("caching: {}".format(interpreter.caching_enabled))

    if args.example == "all":
        run_all_examples(interpreter, parser, args.examples)

    elif args.prompt:
        if not args.gold:
            print("missing --gold <gold text>")
            usage(parser)
            sys.exit(1)

        # run custom example
        interpreter.run(args.prompt, args.gold, "custom", tracing_enabled=args.trace)

    elif args.split:
        run_dataset_examples(interpreter, args)

    else:
        # run single example
        example = args.example
        print("\nexample: {}".format(example))
        prompt, gold = get_example(example, parser, args.examples)
        interpreter.run(prompt, gold, example, tracing_enabled=args.trace)

    elapsed = time.time() - started
    elapsed_text = get_time_str(elapsed)

    print("\ntotal elapsed time: {}".format(elapsed_text))

if __name__ == "__main__":
    main()