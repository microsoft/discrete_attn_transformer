# psl_compiler.py: compiles our Transformer programming language into JSON weights that can be used by the interpreter to run the program, 
# or by weights_compiler to generate the weights for the DAT Transformer model.
import os
import sys
import json
import time
import string
import argparse
from utils.multiline_scanner import MultiLineScanner
import utils.dat_utils as dat_utils
from dat_common import *

PRIME_CHAR = "`"

def dict_str(d):
    text = "{"
    for k, v in d.items():
        if len(text) > 1:
            text += ", "
        text += str(k) + ": " + str(v)
    text += "}"

    return text

class PslLayer:
    def __init__(self, causal_attn, right_match) -> None:
        self.until = None    
        self.where_conditions = []
        self.assignments = []
        self.comment = None
        self.bound_vars = {}
        self.causal_attn = causal_attn
        self.right_match = right_match
        self.error_text = None

class PslRepeatLayer:
    def __init__(self, comment) -> None:
        self.comment = comment
        self.until = None    
        self.layers = []

class PslAssignment:
    def __init__(self, lhs, rhs) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(self.lhs) + " := " + str(self.rhs)

class PslWhereCondition:
    def __init__(self, lhs, rhs) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(self.lhs) + " == " + str(self.rhs)

class PslLhs:
    def __init__(self, register, index_var, has_star=False, weight_func=None, reg_map=None) -> None:
        self.register = register
        self.index_var = index_var
        self.has_star = has_star
        self.weight_func = weight_func
        self.reg_map = reg_map

    def register_str(self, add_func):
        text = self.register
        text = self.reg_map[text]

        if add_func and self.weight_func:
            text += "@" + self.weight_func

        return text

    def __str__(self) -> str:
        text = self.register_str(add_func=False) + "["
        # if self.has_star:
        #     text += "*:"
        text += self.index_var + "]"

        if self.weight_func:
            text += "@" + self.weight_func
        
        return text

class PslRhs:
    def __init__(self, register, index_exp, constant_value=None, weight_func=None, reg_map=None,) -> None:
        self.register = register
        self.index_exp = index_exp   # N, N-1, 0, ...
        self.constant_value = constant_value         # XQ, XA, ...
        self.weight_func = weight_func
        self.operator = None
        self.reg_map = reg_map

    def value_str(self):
        # # can we delete this?  might have to adjust for icl_parser_gen.psl.  rfernand, Jul-30-2024
        # if value == "R_INIT":
        #     value = "R"
        # elif value == "T_INIT":
        #     value = "T"

        return self.constant_value

    def register_str(self, add_func):
        text = self.register
        text = self.reg_map[text]

        if add_func and self.weight_func:
            text += "@" + self.weight_func

        return text

    def __str__(self) -> str:
        if self.register:
            text = self.register_str(add_func=False) + "[" + self.index_exp + "]"
            if self.weight_func:
                text += "@" + self.weight_func
        else:
            text = self.value_str()
        
        return text

class PslCompiler:
    def __init__(self, show_progress=False):
        self.show_progress = show_progress
        
        self.register_map = {}
        self.constant_map = {}
        self.watch_list = []

        # build default system map (in case program doesn't specify it)
        self.system_map = {"symbol": "s", "position": "p", "output": "s", "parse": "a", "eop": "z"}

        #self.register_index_names = ["$", "n", "0"]
        self.weight_func_names = ["pos_increment", "pos_decrement"]

        self.bound_vars = {}
        self.psl_layers = []
        self.causal_attn = None    # by default, kicks in when generating
        self.fn_program = None

    def compile(self, fn_program):

        self.fn_program = fn_program

        with open(fn_program, "r") as f:
            lines = f.readlines()

        scanner = MultiLineScanner(lines, echo_lines=self.show_progress)
        self.scanner = scanner

        # get first token
        scanner.scan()

        self.parse_program(scanner)

        return self.psl_layers

    def parse_program(self, scanner):

        while not scanner.token_type == "eof":
            token, layer = self.parse_statement(scanner)

            if layer:
                self.psl_layers.append(layer)

        return token
    
    def first_layer_init(self):
        '''
        before processing first layer, build register_map and constant_map if program has not specified them (gen 1 compatibility)
        '''
        pass

    def parse_statement(self, scanner):

        last_comment = scanner.last_full_comment
        layer = None
        token = scanner.token

        if token == "registers":
            self.parse_registers_statement(token, scanner)

        elif token == "constants":
            self.parse_constants_statement(token, scanner)

        elif token == "watch":
            self.parse_watch_statement(token, scanner)

        elif token == "system":
            self.parse_system_statement(token, scanner)

        elif token == "causal_attn":
            self.parse_causal_attn_statement(token, scanner)

        elif token in ["where", "where_lm", "where_rm"]:
            if not self.psl_layers:
                self.first_layer_init()
            token, layer = self.parse_where_statement(token, scanner, where_type=token)

        elif token == "repeat":
            if not self.psl_layers:
                self.first_layer_init()
            token, layer = self.parse_repeat(token, scanner)

        else:
            self.show_error("unknown statement: " + token)
        
        if layer:
            layer.comment = last_comment
        return token, layer
    
    def show_error(self, msg):
        scanner = self.scanner
        line_text = scanner.text.strip()
        line_num = scanner.line_index     # usually this is 1 ahead but its an index, so use its exact value

        text = msg + "\n"
        text += "    line {}, program: {}\n".format(line_num, self.fn_program)
        text += "    {}".format(line_text)

        print()
        print("PSL compiler error: {}".format(text))
        print()

        # save this info for caller
        self.error_text = text

        raise Exception(msg)

    def parse_registers_statement(self, token, scanner):
        if self.psl_layers:
            self.show_error("registers declaration must appear before the first statement in the program")

        #scanner.stay_on_current_line = True
        token = scanner.scan()      # skip over "registers"
        self.require_token(":", scanner)
        self.require_token("{", scanner)
        register_map = {}

        # process all tokens on current line
        while scanner.token != "}":
            register_name = scanner.token
            if not scanner.token_type in ["id", "string"]:
                self.show_error("expected register name (id or string) but found: " + scanner.token_type)
            
            scanner.scan()      # skip over register name
            
            short_register_name = self.require_token(":", scanner)
            if not scanner.token_type in ["id", "string"]:
                self.show_error("expected short register name (id or string) but found: " + scanner.token_type)

            if len(short_register_name) == 0 or short_register_name[0] not in string.ascii_lowercase:
                self.show_error("short register name must start with a lowercase letter: " + short_register_name)

            scanner.scan()      # skip over short register name

            register_map[register_name] = short_register_name

            if scanner.token != "}":
                self.require_token(",", scanner)

        #scanner.stay_on_current_line = False
        scanner.scan()      # skip over ] 

        self.register_map = register_map
        return scanner.token

    def parse_system_statement(self, token, scanner):
        if self.psl_layers:
            self.show_error("system declaration must appear before the first statement in the program")

        #scanner.stay_on_current_line = True
        token = scanner.scan()      # skip over "system"
        self.require_token(":", scanner)
        self.require_token("{", scanner)
        system_map = {}
        system_names = ['symbol', 'position', 'output', 'parse', 'eop']

        # process all tokens on current line
        while scanner.token != "}":
            register_name = scanner.token
            if not scanner.token_type in ["id", "string"]:
                self.show_error("expected register name (id or string) but found: " + scanner.token_type)

            if register_name not in system_names:
                self.show_error("unknown system register name: " + register_name)
            
            scanner.scan()      # skip over register name
            
            user_register_name = self.require_token(":", scanner)
            if not scanner.token_type in ["id", "string"]:
                self.show_error("expected user register name (id or string) but found: " + scanner.token_type)

            if not user_register_name in self.register_map:
                self.show_error("unknown user register name: " + user_register_name)

            short_reg_name = self.register_map[user_register_name]

            scanner.scan()      # skip over short register name

            system_map[register_name] = short_reg_name

            if scanner.token != "}":
                self.require_token(",", scanner)

        #scanner.stay_on_current_line = False
        scanner.scan()      # skip over ] 

        # give errors if required system registers are missing
        for req_name in ["symbol", "output"]:
            if req_name not in system_map:
                self.show_error("missing map entry for system register: " + req_name)

        self.system_map = system_map
        return scanner.token

    def parse_constants_statement(self, token, scanner):
        if self.psl_layers:
            self.show_error("registers declaration must appear before the first statement in the program")

        #scanner.stay_on_current_line = True
        scanner.scan()      # skip over "constants"

        self.require_token(":", scanner)
        self.require_token("{", scanner)
        constant_map = {}

        # process all tokens on current line
        while scanner.token != "}":
            constant_name = scanner.token
            if not scanner.token_type in ["id", "string", "number"]:
                self.show_error("expected register name or value (id, string, or number) but found: " + scanner.token_type)

            scanner.scan()      # skip over constant name

            if scanner.token == ":":
                scanner.scan()
                if scanner.token_type not in ["string", "number"]:
                    self.show_error("expected constant string or number but found: " + scanner.token_type)
                constant_value = scanner.token
                constant_map[constant_name] = constant_value
                scanner.scan()

            else:
                constant_value = constant_name
                constant_map[constant_name] = constant_value

            if scanner.token != "}":
                self.require_token(",", scanner)

        #scanner.stay_on_current_line = False
        scanner.scan()      # skip over eol 

        self.constant_map = constant_map
        return scanner.token
    
    def parse_watch_statement(self, token, scanner):
        if self.psl_layers:
            self.show_error("watch declaration must appear before the first statement in the program")

        #scanner.stay_on_current_line = True
        scanner.scan()      # skip over "constants"

        self.require_token(":", scanner)
        self.require_token("[", scanner)
        watch_list = []

        # process all tokens on current line
        while scanner.token != "]":
            reg_name = scanner.token
            self.require_type("id", scanner)

            # watch name can be a register name or short name
            if reg_name in self.register_map:
                reg_name = self.register_map[reg_name]

            watch_list.append(reg_name)

            if scanner.token != "]":
                self.require_token(",", scanner)

        #scanner.stay_on_current_line = False
        scanner.scan()      # skip over eol 

        self.watch_list = watch_list
        return scanner.token

    def parse_causal_attn_statement(self, token, scanner):
        scanner.scan()      # skip over "causal_attn"
        value = self.require_token(":", scanner).lower()

        if value in ["true", "false", "reverse"]:
            self.causal_attn = value
        else:
            self.show_error("unknown value for causal-attn: " + value)
        
        scanner.scan()      # skip over value
        
    def parse_where_statement(self, token, scanner, where_type):
        layer = PslLayer(self.causal_attn, where_type=="where_rm")

        where_indent = scanner.indent
        self.bound_vars = {}

        # remember the where type
        where_verb = token     # where, where-lm, where-rm

        token = scanner.scan()      # skip over "where"
        token, conditions = self.parse_bool_exp(token, scanner)
        token = self.require_token(":", scanner)    

        layer.bound_vars = self.bound_vars  
        layer.where_conditions = conditions  

        # process statements belonging to this where clause
        while scanner.indent > where_indent and scanner.token:
            token, assignment = self.parse_assignment_statement(token, scanner, layer)
            layer.assignments.append(assignment)

        return token, layer

    def parse_repeat(self, token, scanner):
        repeat_layer = PslRepeatLayer(scanner.last_full_comment)
        token = scanner.scan()      # skip over "repeat"

        token = self.require_token(":", scanner)
        while token in ["where", "where_lm", "where_rm"]:
            where_type = token
            last_comment = scanner.last_full_comment
            token, layer = self.parse_where_statement(token, scanner, where_type)
            layer.comment = last_comment
            repeat_layer.layers.append(layer)
        token = self.require_token("until", scanner)

        if token == "NO_CHANGE":
            token = scanner.scan()    # skip over "NO_CHANGE"
            repeat_layer.until = {}
        else:
            def format_cond(cond):
                lhs = cond.lhs.__dict__
                rhs = cond.rhs.__dict__

                return {"lhs": lhs, "rhs": rhs}
            
            token, conditions = self.parse_bool_exp(token, scanner)
            cond_list = [format_cond(cond) for cond in conditions]
            repeat_layer.until = json.dumps(cond_list)
            #layer.until = "bool_exp"

        return token, repeat_layer

    def parse_bool_exp(self, token, scanner):
        token, conditions = self.parse_bool_term(token, scanner)

        if token == "and":
            token = scanner.scan()           # skip over "and"
            token, conditions2 = self.parse_bool_exp(token, scanner)
            conditions += conditions2

        return token, conditions
    
    def parse_bool_term(self, token, scanner):
        if token == "(":
            token = scanner.scan()        # skip over "("
            token, conditions = self.parse_bool_exp(token, scanner)
            token = self.require_token(")", scanner)        

        else:
            token, lhs = self.parse_lhs_register(token, scanner, allow_unfixed=True)

            found_not = False
            if token == "not":
                found_not = True
                token = scanner.scan()

            # parse comparison operator
            operator = token

            if token in ["in", "==", "!="]:
                token = scanner.scan()
            else:
                self.show_error("unknown operator: " + token)

            #lhs.operator = operator
            if operator == "in":
                token, rhs = self.parse_constant_list(found_not, scanner)
            else:
                token, rhs = self.parse_value(token, scanner)

            rhs.operator = operator

            condition = PslWhereCondition(lhs, rhs)
            conditions = [condition]

        return token, conditions
    
    def parse_constant_list(self, found_not, scanner):
        token = self.require_token("[", scanner)
        constants = []    

        while token != "]":
            constants.append(token)

            token, rhs = self.parse_value(token, scanner)

            if token != "]":
                token = self.require_token(",", scanner)

        token = scanner.scan()
        operator = "not_in" if found_not else "in"
        const_values = [operator] + [self.constant_map[const] for const in constants]

        rhs = PslRhs(None, None, const_values, self.register_map)
        
        return token, rhs

    def parse_weight_func(self, token, scanner):
        token = scanner.scan()      # skip over "@"

        if not token in self.weight_func_names:
            self.show_error("unknown weight function: " + token)

        weight_func = token
        token = scanner.scan()      # skip over weight function name

        return token, weight_func

    def parse_lhs_register(self, token, scanner, allow_unfixed):
        register_name = token
        token = self.parse_register_name(token, scanner)
        token = self.require_token("[", scanner)
        token, index_exp, found_star = self.parse_register_index(token, scanner, allow_unfixed, False)
        token = self.require_token("]", scanner)
        weight_func = None

        if token == "@":
            token, weight_func = self.parse_weight_func(token, scanner)

        lhs = PslLhs(register_name, index_exp, found_star, weight_func=weight_func, reg_map=self.register_map)
        return token, lhs
    
    def parse_rhs_register(self, token, scanner, allow_unfixed, allow_index_decrement):

        register_name = token
        token = self.parse_register_name(token, scanner)
        token = self.require_token("[", scanner)
        token, index_exp, found_star = self.parse_register_index(token, scanner, allow_unfixed, allow_index_decrement)
        token = self.require_token("]", scanner)
        weight_func = None

        if token == "@":
            token, weight_func = self.parse_weight_func(token, scanner)

        rhs = PslRhs(register_name, index_exp, weight_func=weight_func, reg_map=self.register_map)
        return token, rhs

    def parse_register_name(self, token, scanner):
        if token in self.register_map:
            token = scanner.scan()      # skip over register name
        else:
            self.show_error("unknown register name: " + token)

        return token

    def parse_register_index(self, token, scanner, allow_binding, allow_index_decrement): 
        found_star = False

        # if token == "n":   # "*":
        #     if allow_binding:
        #         # token = scanner.scan()      # skip over "*"
        #         # token = self.require_token(":", scanner)
        #         # if token != "n":
        #         #     self.show_error("binding index variable must be 'n'")
        #         found_star = True
        #     else:
        #         self.show_error("binding index variable not allowed here")
            
        register_exp = token

        # must be a register index
        if not token in ["n", "N"]:    # self.register_index_names:
            self.show_error("unknown register index name: " + token)
        
        if token == "n":
            self.bound_vars = token

        #known_var = token in self.bound_vars

        # if found_star:
        #     if known_var:
        #         self.show_error("register index already bound: " + token)
            
        #     self.bound_vars[token] = token
        # else:
        #     if not known_var and token == "n":
        #         self.show_error("register index not yet bound: " + token)
        
        token = scanner.scan()      # skip over register index

        return token, register_exp, found_star
    
    def parse_value(self, token, scanner):

        if token in self.constant_map:
            token_value = self.constant_map[token]

            rhs = PslRhs(None, None, token_value, reg_map=self.register_map)
            token = scanner.scan()      # skip over constant name

        elif token in self.register_map:
            token, rhs = self.parse_rhs_register(token, scanner, allow_unfixed=False, allow_index_decrement=True)

        else:
            self.show_error("unknown value (not a register or constant name): " + token)

        return token, rhs
    
    def parse_assignment_statement(self, token, scanner, layer):
        token, lhs = self.parse_lhs_register(token, scanner, allow_unfixed=False)
        token = self.require_token("=", scanner)
        token, rhs = self.parse_value(token, scanner)

        if lhs.index_var != "N":
            self.show_error("only index N can be used in left-hand side of assignments: " + lhs.index_var)
        
        if layer.bound_vars and rhs.index_exp == "N":
            self.show_error("index N cannot be used on right-hand of assignment with n is bound: " + rhs.index_exp)

        if not layer.bound_vars and rhs.index_exp == "n":
            self.show_error("index n cannot be used in an assigment when not used in the production conditions")

        assignment = PslAssignment(lhs, rhs)
        return token, assignment
    
    def require_token(self, expected_token, scanner):
        '''
        this function:
            - ensures the current token matches the expected
            - skips over the matching token
            - returns the token AFTER the expected token
        '''
        token = scanner.token
        if token != expected_token:
            self.show_error("expected token '" + expected_token + "' but found: " + token)

        token = scanner.scan()      # skip over expected token
        return token
    
    def require_type(self, expected_type, scanner):
        '''
        this function:
            - ensures the current token TYPE matches the expected
            - skips over the matching token
            - returns the matching token
        '''
        if scanner.token_type != expected_type:
            self.show_error("expected type '" + expected_type + "' but found: " + scanner.token_type)

        matching_token = scanner.token
        scanner.scan()      # skip over matching token

        return matching_token

    def print_production(self, layer, indent=False):
        prefix = "    " if indent else ""

        print("{}{}".format(prefix, layer.comment))
        condition_str = ""
        assignment_str = ""

        if isinstance(layer, PslRepeatLayer):
            print("repeat layers: ")
            for inner_layer in layer.layers:
                self.print_production(inner_layer, indent=True)

            print("until: {}\n".format(layer.until))

        else:
            for condition in layer.where_conditions:
                if condition_str != "":
                    condition_str += ", "
                condition_str += str(condition)

            for assignment in layer.assignments:
                if assignment_str != "":
                    assignment_str += ", "
                assignment_str += str(assignment)

            print("{}    {} \t {}\n".format(prefix, condition_str, assignment_str))

    def generate_weights(self, layers, print_them=False, indent=False):

        prefix = "    " if indent else ""
        if print_them:
            print("{}weights:".format(prefix))

        all_weights = []
        num_dat_layers = 0

        for i, layer in enumerate(layers):

            if print_them:
                print("{}{}".format(prefix, layer.comment))

            if isinstance(layer, PslRepeatLayer):
                inner_weights, num_inner_layers = self.generate_weights(layer.layers, print_them=print_them, indent=True)
                repeat_layer_weights = {"layer_comment": layer.comment, "until": layer.until, "weights": inner_weights}
                all_weights.append(repeat_layer_weights)

                if print_them:
                    print("until: {}".format(layer.until))

                num_dat_layers += num_inner_layers

            else:
                wd = self.generate_layer_weights(layer)
                num_dat_layers += 1

                layer_weights = {"layer_comment": layer.comment, "weights": wd, "causal_attn": layer.causal_attn, "right_match": layer.right_match}   
                all_weights.append(layer_weights)

                if print_them:
                    for key in wd:
                        print("{}  {}: {}".format(prefix, key, dict_str(wd[key])))
                    print()

        return all_weights, num_dat_layers

    def generate_layer_weights(self, layer):
        q_dict, k_dict = self.generate_q_k_dicts(layer)
        v_dict = self.generate_v_dict(layer.assignments)

        weights = {"q": q_dict, "k": k_dict, "v": v_dict}
        return weights
    
    def prime_name(self, name):
        return name + PRIME_CHAR
    
    def generate_q_k_dicts(self, layer):
        q_dict = {}
        k_dict = {}

        for condition in layer.where_conditions:
            lhs_key_str = condition.lhs.register_str(False)
            lhs_reg_name = lhs_key_str

            left_index = condition.lhs.index_var

            rhs_value = condition.rhs.register_str(True) if condition.rhs.register else condition.rhs.value_str()
            right_index = condition.rhs.index_exp if condition.rhs.register else None

            if right_index == "0":
                # special case to match position=0 column
                q_dict['p'] = '0'
                k_dict['p'] = 'p'

            # decide where to put the LHS and RHS (query vs. key)
            # if condition.rhs.weight_func:
            #     q_value = lhs_reg_name
            #     k_value = rhs_value
            #     swapped = True

            if left_index == "n":
                # this is when current condition is bound to n
                # in this case, we put constants on query
                if not condition.lhs.weight_func:      
                    alt_name = "alt_" + lhs_key_str
                    lhs_key_str = self.prime_name(lhs_key_str)

                    if not alt_name in self.register_map:
                        self.register_map[alt_name] = lhs_key_str

                q_value = rhs_value
                k_value = lhs_reg_name
                swapped = False

            elif layer.bound_vars:
                # where n is used in 1 or more of the conditions, and the current conditon is on $: put constants on key
                q_value = lhs_reg_name
                k_value = rhs_value
                swapped = True

            else:
                # for $-only based conditions: put constants on key (changed: Aug-29-2023)
                q_value = lhs_reg_name
                k_value = rhs_value
                swapped = True

            if condition.rhs.operator == "!=":
                # add the non-equal operator for use when building weights in TF_Explorer
                neq_target = rhs_value   # condition.rhs.constant if condition.rhs.constant is not None else condition.rhs.re
                if swapped:
                    k_value = ("!=", neq_target)
                else:
                    q_value = ("!=", neq_target)

            q_dict[lhs_key_str] = q_value
            k_dict[lhs_key_str] = k_value


        if not layer.bound_vars and "p" not in q_dict and "p" not in k_dict:
            # force current column to attend only to itself
            q_dict["p"] = "p"
            k_dict["p"] = "p"

        return q_dict, k_dict

    
    def generate_v_dict(self, assignments):
        v_dict = {}

        for assignment in assignments:
            reg_str = assignment.lhs.register_str(True)
            val_str = assignment.rhs.register_str(True) if assignment.rhs.register else assignment.rhs.value_str()
            v_dict[reg_str] = val_str

        return v_dict
    
    def print_productions(self, layers, indent=False):
        if not indent:
            print("productions:")

        for i, layer in enumerate(layers):
            self.print_production(layer, indent=indent)

    def save_weights(self, weights, fn_source):
        # write out weights to file
        data = {"register_map": self.register_map, "constant_map": self.constant_map, "watch_list": self.watch_list, "system_map": self.system_map, "weights": weights}
        text = json.dumps(data, indent=4)

        fn_base = os.path.basename(fn_source).split(".")[0]
        fn = "json_weights/{}.json".format(fn_base)

        # create dir, if needed
        dir = os.path.dirname(fn)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(fn, "wt") as f:
            f.write(text)

        return fn
    
def parse_args():
    fn_psl = "psl_programs/icl_parser_gen.yaml"

    parser = argparse.ArgumentParser(add_help=False, description="Compile the specified PSL program.")
    parser.add_argument("--psl", type=str, help="specify the PSL program to compile", default=fn_psl)
    parser.add_argument("--help", action="store_true", help="print information command line arguments", default=0)

    args = parser.parse_args()

    args.psl = dat_utils.fixup_psl(args.psl)
    return args, parser

def usage(parser):

    parser.print_help()
    # print("usage: python psl_compiler.py <psl_program_file>")

    print()
    print("examples: ")
    print("  > python psl_compiler.py   (compile deault psl program)")
    print("  > python psl_compiler.py   --psl icl_parser_gen.yaml")

def compile(fn_source, show_progress):
    
    fn_source = dat_utils.fixup_psl(fn_source)

    compiler = PslCompiler(show_progress)

    layers = compiler.compile(fn_source)

    if show_progress:
        compiler.print_productions(layers)

    weights, num_dat_layers = compiler.generate_weights(layers, print_them=show_progress)
    fn = compiler.save_weights(weights, fn_source)

    return fn, num_dat_layers

if __name__ == "__main__":  

    args, parser = parse_args()

    if args.help:
        usage(parser)
        sys.exit(0)

    started = time.time()
    fn = args.psl

    print("compiling psl program from: {}".format(fn))
    fn, num_dat_layers = compile(fn, show_progress=False)

    elapsed = time.time() - started
    print("compile completed, elapsed: {:.2f} secs, DAT layers: {}, weights written to: {}".format(elapsed, num_dat_layers, fn))

