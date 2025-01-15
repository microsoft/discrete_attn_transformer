# dat_explorer.py: GUI to view activations of DatTransformer while running a selected psl program on a selected or edited example prompt.
import os
import sys
import json
import math
import time
import yaml
import torch
import argparse
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from utils.tooltip import StaticToolTip, DynamicToolTip
from dat_common import *
from utils.dat_utils import *
from dat_transformer import DatTransformer
from dat_interpreter import DatInterpreter
from psl_compiler import PslCompiler
from weights_compiler import WeightsCompiler
       

#SUBLAYER_KEYS = ["input", "q", "k", "v", "output", "residual"]
SUBLAYER_KEYS = ["residual", "output", "value", "key", "query", "input"]             # from top to bottom visually on screen (and the order that we build them in)

def safe_key(dict, key, default=None):
    if key in dict:
        return dict[key]
    else:
        return default
    
class UIGridCell:
    '''
    Data structure for a single UI grid cell of the row/column grid.
    '''
    def __init__(self) -> None:
        self.cframe = None  
        self.wlabel = None
        self.controls = None
        self.values = None

        self.reset()

    def reset(self):
        self.values = {"input": "", "query": "", "key": "", "value": "", "output": "", "residual": ""}

    def update_ui(self):
        if self.controls:
            '''
            Update the i/q/k/v/o/r labels for this grid cell 
            '''
            for cname, gui in self.controls.items():

                # get watched register values
                values = self.values[cname]
                text = ":".join([str(v) for v in values])

                if text != gui.cget("text"):
                    gui.config(text=text)

       
class UIGridLayer:
    '''
    Data structure for a single UI layer of the row/column grid.
    '''
    def __init__(self, num_columns) -> None:
        self.columns = [UIGridCell() for i in range(num_columns)]
        self.k_weights = None
        self.q_weights = None
        self.v_weights = None
        self.lframe = None
        self.status_label = None
        self.status_label_frame = None

    def reset(self, update_ui=False):
        for col in self.columns:
            col.reset()

            if update_ui:
                col.update_ui()

    def update_ui(self):
        for col in self.columns:
            col.update_ui()

class DatExplorer():
    def __init__(self, program_name, example, fn_examples, d_register, n_heads, max_display_rows, 
        use_productions_vocab, interpreter=False) -> None:

        self.d_register = d_register
        self.num_heads = n_heads

        self.tk_root = None
        self.model = None            
        self.num_dat_layers = None
        self.model_needs_loading = True
        self.fn_examples = fn_examples
        self.tracing_enabled = False
        self.cb_tracing_var = None

        self.dummy = 0
        self.layer_steps = None
        self.raw_layer_steps = None
        self.col_step = 0       # tracks our stepping thru current leading column
        self.global_step = 0    # tracks our stepping thru all rows of leading to end column
        self.tokens = None
        self.col_step_label = None
        self.prompt_entry = None
        self.max_global_steps = None
        self.matrices_to_test = []
        self.leading_column = 0
        self.response_label = None
        self.response = None
        self.frame0 = None
        self.use_production_vocab = use_productions_vocab
        self.grid_canvas = None
        self.fn_save_outputs = None
        self.loaded_inputs = None
        self.grid_cell_name_labels = {}
        self.token_labels = []
        self.status_label_frame = None
        self.lframe = None
        self.system_map = None

        self.model_names = ["DAT Transformer", "DAT Interpreter"]
        if interpreter:
            self.model_name = self.model_names[1] 
        else:
            self.model_name = self.model_names[0] 

        self.max_displayed_layers = max_display_rows
        self.first_displayed_row = 0

        self.font_size = 10 # None 
        self.small_font_size = 8   # None 

        self.current_program_label = None
        self.program_labels = {}
        self.program_dropdown = None
        self.program_dropdown_variable = None
        self.example_dropdown = None
        self.example_dropdown_variable = None

        initial_name, initial_prompt, initial_gold = self.read_examples(fn_examples, example)

        self.prompt = initial_prompt
        self.gold_text = initial_gold
        self.gold_entry = None
        self.example = initial_name
        self.jit_decoding = True               # don't decode everything at once - wait until user references the row(s)

        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

        # process program_name
        program_name = program_name.replace("\\", "/")    # make slashes consistent
        if program_name.startswith("psl_programs/"):
            program_name = program_name.split("/")[1]

        if program_name.endswith(".yaml"):
            program_name = program_name.split(".")[0]

        self.on_psl_program_changed(program_name)
        self.init_ui(program_name)

        self.program_name = program_name

    def on_psl_program_changed(self, fn_psl):
        try:
            self.current_program_name = fn_psl
            self.model_needs_loading = True

            # compile PSL program
            if not os.path.exists(fn_psl):
                # qualify name with directory and extension
                fn_psl = "psl_programs/{}.yaml".format(fn_psl)

            # compile PSL program
            print("compiling program: {}".format(fn_psl))

            fn_weights, num_dat_layers = self.compile_psl_only(fn_psl)
            self.fn_weights = fn_weights
            self.num_dat_layers = num_dat_layers

            # open JSON weights file and load register/constant info

            weight_layers, register_map, constant_map, register_name_to_index, watch_list, system_map = load_json_weights(fn_weights)
            self.system_map = system_map

            self.weight_layers = weight_layers
            self.register_map = register_map
            self.constant_map = constant_map
            self.register_name_to_index = register_name_to_index

            n_registers = len(register_name_to_index)
            self.d_head = n_registers * self.d_register
            self.d_model = self.d_head * self.num_heads

            self.watch_registers = watch_list if watch_list else ["r", "t", "f", "d"]
            self.build_watch_register_numbers()
            self.update_window_title()

            self.build_fixed_vocab_and_embedding()

            self.update_program_related_stuff(fn_psl)
            
        except Exception as ex:
            if "Compile Error" not in str(ex):
                print("Error loading program: {}".format(ex))
                messagebox.showerror("Error loading program", "Error loading program: {}".format(ex))

            self.hide_message()

    def update_program_related_stuff(self, psl_path):

        prompt_count = len(self.prompt.split())
        gold_count = len(self.gold_text.split())
        self.max_displayed_cols = prompt_count + gold_count
        #print("loading program {}".format(psl_path))
        self.program = self.program_from_json_weights(psl_path)
        num_layers = len(self.program)

        self.num_layers = num_layers
        self.num_cols = self.max_displayed_cols
        self.max_col_steps = 1 + 5*num_layers
        self.initial_prompt = self.prompt

    def run_example(self, prompt, gold, example):

        started = time.time()

        d_register_needed = get_needed_d_register_size(prompt, gold, self.register_map, self.constant_map)
        
        if d_register_needed > self.d_register or not self.model:
            # resize embeddings/model for larger example/vocab
            self.d_register = d_register_needed
            self.model_needs_loading = True

        if self.model_needs_loading:
            self.build_fixed_vocab_and_embedding()
            self.compile_and_load_model(self.current_program_name, compile_psl=False, load_model=True)
            self.model_needs_loading = False

        vocab = build_vocab_from_example(prompt, gold, self.register_map, self.constant_map)
        self.model.load_vocab(vocab)

        prompt_tokens = prompt.split()
        stoi = {t: i for i, t in enumerate(vocab)}
        src_indexes = [stoi[t] for t in prompt_tokens]

        max_new_tokens = len(gold.split())

        self.show_message("Running prompt in model...")

        y_hat_text, last_steps = self.model.generate(prompt, self.device, max_new_tokens=max_new_tokens, 
            example=example, return_last_steps=True, tracing_enabled=self.tracing_enabled)
        correct = (y_hat_text == gold)

        #self.model = None

        print("  prompt: {}".format(prompt))
        print("  gold:   {}".format(gold))
        print("  y^:     {}".format(y_hat_text))

        feedback = "CORRECT" if correct else "WRONG"
        elapsed = time.time() - started
        print("  [{}]  ({:.2f} secs)".format(feedback, elapsed))

        return y_hat_text, last_steps

    def build_fixed_vocab_and_embedding(self):

        # get fixed part of vocab (and its stoi)
        self.fixed_vocab = build_vocab_from_example(prompt="", gold="", register_map=self.register_map, constant_map=self.constant_map)

        # make an embedding to match what is used in DatTransformer
        self.embedding = build_embedding(d_embedding=self.d_register, max_vocab_len=self.d_register, device=self.device)
        
    def compile_psl_only(self, fn_psl):
        show_progress = False
        compiler = PslCompiler(show_progress=show_progress)

        try:
            layers = compiler.compile(fn_psl)
        except Exception as ex:
            text = compiler.error_text
            # display in messagebox
            messagebox.showerror("Error compiling PSL program", text)
            raise Exception("Compile Error")

        weights, num_dat_layers = compiler.generate_weights(layers, print_them=show_progress)

        fn_weights = compiler.save_weights(weights, fn_psl)
        return fn_weights, num_dat_layers

    def compile_and_load_model(self, fn_source, compile_psl, load_model):
        started = time.time()

        if load_model and self.model_name == "DAT Interpreter":
            fn_psl = fn_source
            if not "." in fn_psl:
                fn_psl += ".yaml"
            fn_psl = fixup_psl(fn_psl)

            self.model = DatInterpreter(fn_psl)

        else:
            if compile_psl:
                # compile our TGT program
                compiler = PslCompiler()
                layers = compiler.compile(fn_source)
                weights, num_dat_layers = compiler.generate_weights(layers)
                fn_weights = compiler.save_weights(weights, fn_source)    

                print("compiled psl file: {}".format(fn_source))
                self.num_dat_layers = num_dat_layers
                self.fn_weights = fn_weights

            if load_model:

                self.show_message("Compiling program weights...")

                # compile generated JSON weights
                compiler = WeightsCompiler(self.fixed_vocab, self.embedding, self.d_register, self.register_name_to_index)
                matrix_layers = compiler.compile_json_weights(self.fn_weights, self.device)
                print("compiled weights file: {}".format(self.fn_weights))

                elapsed = time.time() - started
                print("total compilation time: {:.2f} sec".format(elapsed))

                self.show_message("Loading weights into model...")
                self.load_transformer_model(matrix_layers)
        
    def read_examples_from_file(self, fn_examples):
        # read fn_examples as a YAML file
        with open(fn_examples, "rt") as file:
            all_examples = yaml.safe_load(file)
            examples = all_examples["examples"]

        self.examples = examples
        return examples

    def read_examples(self, fn_examples, example):
        examples = self.read_examples_from_file(fn_examples)

        if example and example in examples:
            initial_name = example
        else:
            initial_name = list(examples)[0]

        ed = examples[initial_name]
        initial_prompt = ed["prompt"]
        initial_gold = ed["gold"]

        return initial_name, initial_prompt, initial_gold
    
    def load_transformer_model(self, matrix_layers):
        print("loading model...")
        load_started = time.time()
        self.model = None

        model = DatTransformer(self.fixed_vocab, self.embedding, d_register=self.d_register, num_encoder_layers=self.num_dat_layers, 
            register_name_to_index=self.register_name_to_index, log_progress=False, system_map=self.system_map)

        elapsed = time.time() - load_started
        #print("model created ({:.2f} secs)".format(elapsed))

        started = time.time()
        model.load_weights(self.d_register, self.embedding, weights=matrix_layers)
        #model.to(self.device)
        elapsed = time.time() - started
        #print("weights loaded/to device: ({:.2f} secs)".format(elapsed))

        load_elapsed = time.time() - load_started
        num_params = model.get_num_params()

        print("model loaded ({:.2f} secs), d_register: {}, NUM_REGS: {}, d_hidden: {:,}, params: {:,}".format(\
            load_elapsed, self.d_register, len(self.register_name_to_index), model.d_model, num_params))

        self.model = model

        if WEIGHT_TYPE == "direct":
            self.model = self.model.to(self.device)

    def init_ui(self, initial_program):

        # create root stuff once only
        self.create_root()

        #compile_elapsed = self.compile_psl_and_weights(path)

        # create UI GRID to match current program (will be updated later, when RUN is executed)
        self.show_message("Building the UI controls...")

        # load INITIAL PROGRAM
        self.load_new_program(initial_program, reset=False)
        self.on_prompt_or_gen_changed()

        # always compile at start of a new session
        # path = "psl_programs/{}.yaml".format(initial_program)
        # self.on_psl_program_changed(path)

        #self.reset()
        #self.create_ui(self.initial_prompt, self.gold_text)
        self.update_all_layers()
        self.hide_message()

    def build_watch_register_numbers(self):
        self.watch_register_numbers = []

        for key in self.watch_registers:
            if key in self.register_name_to_index:
                self.watch_register_numbers.append(self.register_name_to_index[key])

    def load_inputs(self, fn):
        with open(fn, "rt") as file:
            list_data = json.load(file)

        tensor_data = torch.tensor(list_data, dtype=torch.float32)
        self.log_outputs(tensor_data, "loaded inputs")

        return tensor_data
        
    def load_psl_program(self, psl_path):

        self.on_psl_program_changed(psl_path)

    def on_example_changed(self, prompt, gold_text):

        # must update the prompt in case user edited it
        self.prompt = prompt
        self.gold_text = gold_text

        prompt_count = len(prompt.split())
        gold_count = len(gold_text.split())

        cols_needed = prompt_count + gold_count
        if not self.max_displayed_cols or cols_needed > self.max_displayed_cols:
            self.max_displayed_cols = cols_needed
            # rebuild grid with new number of columns
            self.rebuild_grid()

        self.num_cols = self.max_displayed_cols
        self.initial_prompt = self.prompt

    def load_new_program(self, program_name, reset=True):

        self.show_message("Loading program: {}...".format(program_name))

        if reset:
            load_elapsed = self.load_psl_program(program_name)

        self.reset_ui()

        self.hide_message()

    def load_new_example(self, example):
        ed = self.examples[example]

        self.initial_prompt = ed["prompt"]
        self.gold_text = ed["gold"]

        self.example = example

        self.show_message("Loading example: " + example)

        self.on_example_changed(self.initial_prompt, self.gold_text)

        self.reset_ui()
        self.hide_message()
        
    def reset_ui(self):
        started = time.time()

        self.clear_all_ui()

        # pause and update UI
        self.tk_root.update()

        #self.reset()
        self.create_ui(self.initial_prompt, self.gold_text)
        self.update_all_layers()

        self.on_prompt_or_gen_changed()

        elapsed = time.time() - started
        return elapsed

    def build_program_layer_from_productions(self, layer_dict):
        '''
        build a text description of the program layer from the productions in the layer_dict.
        '''
        program_layer = {}
        program_layer["purpose"] = layer_dict["layer_comment"]
        weights = layer_dict["weights"]

        for key in ["q", "k", "v"]:
            value_dict = weights[key]

            def pretty(v):
                if isinstance(v, list):
                    v = "{} [{}] ".format(v[0], ", ".join(v[1:]))
                return v
            
            vd_pretty = {k:pretty(v) for k,v in value_dict.items()}
            program_layer[key] = vd_pretty

        program_layer["output"] = {}
        return program_layer

    def program_from_json_weights(self, psl_path):
        layers = []
        program_layers = []
        repeat_group = 1

        # automatically compile source file to fn (if needed)
        base_fn = os.path.basename(psl_path).split(".")[0]
        fn_weights = "json_weights/{}.json".format(base_fn)

        if not os.path.exists(fn_weights):
            print("WARNING: compiled weights file not found: {}".format(fn_weights))

        else:        
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

                    # inner_weights = []
                    # tf_layer = {"until": layer_dict["until"]}
                    # tf_layer["inner_weights"] = inner_weights
                    # layers.append(tf_layer)

                    for wd in weights:

                        program_layer = self.build_program_layer_from_productions(wd)
                        program_layer["repeat_group"] = repeat_group
                        program_layers.append(program_layer)

                    repeat_group += 1

                else:
                    program_layer = self.build_program_layer_from_productions(layer_dict)
                    program_layers.append(program_layer)

        return program_layers
    
    def clear_all_ui(self):
        # remove old UI holders
        self.current_program_label = None
        self.program_labels = None
        self.response_label = None

        if self.frame0:
            # destroy main frame under root
            self.frame0.destroy()

    def setup_fonts(self):
        if self.font_size:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family="Arial", size=self.font_size)

        self.font = tkfont.Font(root=self.tk_root, family='Arial', size=self.font_size)
        self.small_font = tkfont.Font(root=self.tk_root, family='Arial', size=self.small_font_size)

    def start_ui_loop(self):
        # after 100 ms, scroll to layer 0
        self.tk_root.after(100, self.scroll_to_layer(0))

        # run the GUI (wait for and process events)
        self.tk_root.mainloop()

    def reset_all(self, update_ui=False):

        self.layer = 0
        self.col_step = 0
        self.global_step = 0
        self.sub_layer = "input"
        self.layer_steps = None    # force latest prompt to be used as input 
        self.clear_column_borders("#eee")   # "gray")
        #self.show_columns(0, True)
        #self.set_program_step(0)

        self.token_labels = []

        self.response = ""
        self.update_response_text()

        for layer in self.layer_instances:
            layer.reset(update_ui)

        self.leading_column = None

        self.update_status_line()
        self.clear_col_weight_labels()

        # reset EVERYTHING
        self.load_new_program(self.current_program_name)

    def fill_grid_cell_header(self, cell_header_frame, cell_name, layer_index, col_index):

        name_label = tk.Label(cell_header_frame, text=cell_name, font=self.small_font)     # bg="gray", fg="white", 
        name_label.grid(row=0, column=0, sticky="w")    # , pady=(0, 5))

        wlabel = tk.Label(cell_header_frame, text="", font=self.small_font)   # bg="gray", fg="white",
        wlabel.grid(row=0, column=1, sticky="e")   #  pady=(0, 5))

        # give 2nd column some weight so it expands with parent width
        cell_header_frame.grid_columnconfigure(1, weight=1)

        # store labels in a dict for easy update later
        self.grid_cell_name_labels[cell_name] = name_label

        return name_label, wlabel, cell_header_frame

    def create_grid_cell(self, gui_row, cell_name, cell_inst, parent, layer_index, column_index):
        col_root = tk.Frame(parent)   # , bg="gray")
        col_root.grid(row=gui_row, column=column_index+1, padx=5, pady=5)
        col_root.configure(highlightbackground='#aaa', highlightthickness=1)    # add a border

        # row 0: cell header
        cell_header_frame = tk.Frame(col_root)
        cell_header_frame.grid(row=0, column=0, sticky="nwe")

        # hook double-click to update weights for the clicked-on column
        #self.bind_widget_to_pov_column_for_weights(cell_header_frame, layer_index, column_index)

        name_label, wlabel, cell_header_frame = self.fill_grid_cell_header(cell_header_frame, cell_name, layer_index, column_index)

        # save # of UI elements created on last layer 
        last_layer = False    # layer_index == len(self.layers)-1
        if last_layer:
            sub_layers = ["i"]
        else:
            sub_layers = ["i", "q", "k", "v", "o", "r"]
        
        sub_count = len(sub_layers)
        col_labels_frame = tk.Frame(col_root, highlightthickness=8)
        col_labels_frame.grid(row=1, column=0)

        cell_inst.cframe = col_labels_frame
        cell_inst.wlabel = wlabel
        cell_inst.controls = {}

        tt_children = []      # allow for tooltip on cell header (to reflect residual)

        for i in range(sub_count):
            # each column in a layer has 5 rows (input, q, k, v, output)
            mylabel = tk.Label(col_labels_frame, text=sub_layers[(sub_count-1)-i])
            mylabel.grid(row=i+1, column=1)

            text = ""  # 'col-' + str(column_index) + "-" + str(i)
            cname = SUBLAYER_KEYS[i]

            # for scaling UI, just use a single label here (that will display all registers on our watch list)
            label_control = tk.Label(col_labels_frame, text=text, width=10, padx=5, pady=2, borderwidth=1, relief="sunken")
            label_control.grid(row=i+1, column=2)
            cell_inst.controls[cname] = label_control
            tt_children.append(label_control)

            # add some context to the label
            label_control.layer = layer_index
            label_control.column = column_index
            label_control.sub_layer = i

        # hide_frame = tk.Frame(cframe, bg="gray", width=150, height=150)
        # hide_frame.grid(row=0, column=0, columnspan=99, rowspan=99)

        return tt_children

    def run_program(self):

        self.show_message("Running program...")

        # self.response = ""
        # self.update_response_text()

        # # pause to update controls
        # self.tk_root.update()

        # self.reset_all(False)

        # # RESET UI
        # reset_elapsed1 = self.reset_ui()

        # expand self.max_displayed_cols to fit the new program, if needed
        self.prompt = self.prompt_entry.get()
        self.gold_text = self.gold_entry.get()

        self.on_example_changed(self.prompt, self.gold_text)

        prompt_tokens = self.prompt.split()
        gold_tokens = self.gold_text.split()
        prompt_len = len(prompt_tokens)

        #self.on_prompt_or_gen_changed()

        self.layer_steps = {}   # for JIT decoding

        try:
            predicted_text, last_steps = self.run_example(self.prompt, self.gold_text, self.example)
        except Exception as e:
            predicted_text = "<error encountered>"
            last_steps = None

            # show stack trace
            print("error running program: {}".format(e))
            import traceback
            traceback.print_exc()

            self.hide_message()
            messagebox.showerror("Fatal Error", "Error running program: {}".format(e))
            exit(1)

        predicted_tokens = predicted_text.split()
        last_col_index = prompt_len + len(predicted_tokens) - 1
        self.response = predicted_text

        for col in range(prompt_len-1, last_col_index):
            self.set_column_borders(col, "green")

        self.raw_layer_steps = last_steps
        self.leading_column = last_col_index

        self.update_response_text()
        self.on_prompt_or_gen_changed()

        self.show_rows_at(self.first_displayed_row)

        self.hide_message()

    def update_response_text(self):
        if "." in self.response:
            # truncate all text AFTER the first "."
            index = self.response.index(".")
            self.response = self.response[:index+1]

        if self.response_label:
            self.response_label.config(text=self.response)

        self.gold_text = self.gold_entry.get()

        if self.response == self.gold_text:
            self.response_label.config(fg="green")
        elif self.response:
            self.response_label.config(fg="red")
        else:
            self.response_label.config(bg="gray")

    def update_leading_column_marker(self):

        #self.clear_column_borders("gray")

        if self.leading_column is not None:
            self.set_column_borders(self.leading_column, "green")

    def fill_ui_with_label(self, label):
        label.grid(row=0, column=0, columnspan=4, sticky="nsew")
        label.configure(background='#fff')
        self.tk_root.grid_columnconfigure(0, weight=1)
        self.tk_root.grid_rowconfigure(0, weight=1)

    def show_message(self, msg, msg_title="Please wait"):

        # nesting of messages is not supported
        if self.status_label_frame is None:

            # fill window to hide blocky UI update
            message_frame = tk.Frame(self.tk_root, bg="white", bd=1, relief="solid")
            self.fill_ui_with_label(message_frame)

            # inner frame to hold our 2 labels
            inner_frame = tk.Frame(message_frame, bg="white")
            inner_frame.place(relx=0.5, rely=0.5, anchor="center")

            # wait label: bold, larger font, white bg
            wait_label = tk.Label(inner_frame, text=msg_title)
            wait_label.configure(font=("Arial", 12, "bold"), bg="white")

            # message label: normal font
            message_label = tk.Label(inner_frame, text=msg)
            message_label.configure(font=("Arial", 10), bg="white")

            # stack labels vertically
            wait_label.grid(row=0, column=0, pady=2)
            message_label.grid(row=1, column=0, pady=0)

            # left justify the text
            #message_label.configure(width=30)  # , wraplength=300, justify="left")

            self.status_label_frame = message_frame
            self.status_label = message_label

        else:
            # just update the existing message label
            self.status_label.config(text=msg)

        # force UI update
        self.tk_root.update_idletasks()  # Update the window and process events

    def lift_message(self):
        if self.status_label_frame is not None:
            # bring message to front
            self.status_label_frame.lift()
            self.tk_root.update_idletasks()  # Update the window and process events

    def hide_message(self):

        # finish updating UI while it is hidden
        self.tk_root.update_idletasks()  # Update the window and process events

        if self.status_label_frame is not None:
            self.status_label_frame.destroy()
        self.status_label_frame = None
        self.status_label = None

        #print("message now hidden")

    def on_prompt_or_gen_changed(self):
        tokens = []

        if self.prompt_entry:
            prompt = self.prompt_entry.get()
            if prompt:
                tokens += prompt.split()

        self.tokens = tokens

        if self.response:
                tokens += self.response.split()

        self.update_tokens_line(tokens)

    def ensure_row_is_decoded(self, row_num):
        layer_index = row_num - 1

        # # debug decoding of IN and NOT IN for layer_index=6, step=keys
        # raw_steps = self.raw_layer_steps[5]   # layer_index=6
        # k_steps = raw_steps["key"]
        # k_step = k_steps[0]  # col=0
        # self.model.decoder.decode_step(k_step, 5)

        if layer_index not in self.layer_steps:
            self.show_message("Decoding rows JIT...")
            print("  starting decode of row: {}".format(row_num))

            # decode steps of specified layer
            raw_steps = self.raw_layer_steps[layer_index]
            decoded_steps = self.model.decoder.decode_steps(raw_steps, layer_index)
            self.layer_steps[layer_index] = decoded_steps

            self.hide_message()

    def save_outputs(self, output):
        fn = self.fn_save_outputs

        if fn:
            # create dir, if needed
            dir = os.path.dirname(fn)
            if not os.path.exists(dir):
                os.makedirs(dir)

            with open(fn, "w") as f:
                logits = output.detach().numpy()

                self.log_outputs(logits, "saved outputs")

                safe_logits = logits.tolist()
                json.dump(safe_logits, f, indent=4)

    def log_outputs(self, logits, name):
        print(name)

        logits = logits.squeeze(0)
        for c, col in enumerate(logits):
            tensor_col = torch.tensor(col).unsqueeze(0)
            decoded = self.model.decoder.decode_step(tensor_col)
            print("col {}: {}".format(c, decoded))

    def update_grid_cell(self, layer_index, key_index, layer_inst):
        '''
        NOTE: layer_inst refers to one of the displayable layers that layer_index has been temporarily mapped to. 
        '''
        key = None

        if self.layer_steps is not None:
            key = SUBLAYER_KEYS[key_index]

            #layer = self.layers[layer_index]
            #if layer_index < len(self.layer_steps):
            if layer_index in self.layer_steps:
                sub_layer_values_by_col = self.layer_steps[layer_index][key]

                for c, text in enumerate(sub_layer_values_by_col):
                    # allow for smaller number of columns, for faster debugging
                    #print("testing column for layer: {}, col: {}".format(layer_index, c))
                    if c >= len(layer_inst.columns):
                        break

                    if c > self.leading_column:
                        break

                    grid_inst = layer_inst.columns[c]
                    if not grid_inst.controls:
                        # skip columns that have not been initialized (tkinter scaling issue)
                        break
                    #values = [token, "", "", "p"+str(c)]

                    #print("updating column for layer: {}, col: {}".format(layer_index, c))

                    # build the cell text for this cell
                    ctext = []
                    for num in self.watch_register_numbers:
                        symbol = text[num]
                        ctext.append(symbol)

                    grid_inst.values[key] = ctext
                    grid_inst.update_ui()

            else:
                key = None

        return key

    def set_pov_column_for_weights(self, event):
        col_index = event.widget.column_index 
        relative_layer_index = event.widget.layer_index 
        absolute_layer_index = event.widget.layer_index + self.first_displayed_row - 1

        #print("col_index: {}, layer_index: {}".format(col_index, layer_index))
        layer_inst = self.layer_instances[relative_layer_index]
        all_col_weights = self.layer_steps[absolute_layer_index]["attn_weights"]
        weights = all_col_weights[col_index]

        self.update_col_weight_labels(layer_inst, weights, col_index)

    def update_col_weight_labels(self, layer_inst, weights, pov_col_index):
        for w, weight in enumerate(weights):
            col_inst = layer_inst.columns[w]
            if col_inst.wlabel:
                text = "wt(C{}): {:.2f}".format(1+pov_col_index, weight)

                if weight < .1:
                    col_inst.wlabel.config(text=text, fg="black", bg="#eee")
                else:
                    col_inst.wlabel.config(text=text, fg="white", bg="blue")

    def clear_col_weight_labels(self):
        for layer_inst in self.layer_instances:
            for col_inst in layer_inst.columns:
                if col_inst.wlabel:
                    col_inst.wlabel.config(text="", fg="black", bg="#eee")  

    def update_status_line(self):
        if self.col_step_label:
            self.col_step_label.config(text=str(self.col_step))

            layer_number = 1 + (self.col_step-1)//5
            self.layer_label.config(text=str(layer_number))

    def update_all_layers(self):
        for layer_inst in self.layer_instances:
            layer_inst.update_ui()

    def clear_column_borders(self, color):
        for layer in self.layer_instances:
            if layer.lframe is not None:
                for c in range(0, self.max_displayed_cols):
                    col_frame = layer.columns[c]
                    if col_frame.cframe:
                        col_frame.cframe.configure(highlightbackground=color)

    def set_column_borders(self, start_index, color):
        for layer in self.layer_instances:
            if layer.lframe is not None:
                for c in range(start_index, self.max_displayed_cols):
                    col_frame = layer.columns[c]
                    if col_frame.cframe:
                        col_frame.cframe.configure(highlightbackground=color)
                    break     # for now, only highlight first specified column

    def build_program_frame(self, pframe):

        scrollable_frame, canvas = self.create_scrollable_frame(pframe, vertical=False)
        self.program_labels = {}
        start_step = 0

        for p, player in enumerate(self.program):
            self.build_pl_frame(scrollable_frame, player, p, start_step)
            start_step += 5

            # UI debug
            # if p == 10:
            #     break

        # tkinter WORKAROUND: adjust the canvas height to the that of its contents
        scrollable_frame.update_idletasks()            
        bbox = scrollable_frame.bbox(tk.ALL)
        canvas.configure(height=bbox[3] - bbox[1])            

    def build_pl_frame(self, pframe, player, row, pl_step):
        '''
        Build a program layer frame consisting of:
            title (title)
            indented, executable steps:
                - input, q, k, v, output
        '''
        pl_frame = tk.Frame(pframe, padx=5, pady=5)    # , background="blue")
        pl_frame.configure(highlightbackground='#aaa', highlightthickness=1)    # add a border
        pl_frame.grid(row=0, column=row, padx=(0,15), pady=(15,10), sticky="nsew")

        # allow column to expand as per its children
        pframe.grid_columnconfigure(row, weight=1)

        # double-click on program frame navigates to that row
        pframe.bind("<Double-Button-1>", lambda a: self.show_rows_at(row+1, center=True))

        # title at top
        width = 40

        if "repeat_group" in player:
            text = "Repeat group: {}".format(player["repeat_group"])
            label = tk.Label(pl_frame, text=text, anchor="w", width=width, fg="blue")
            label.grid(row=0, column=0)        

        comment = player["purpose"]
        if comment.startswith("//"):
            comment = comment[2:].strip()
            
        text = "L{}: {}".format(1+row, comment)
        label = tk.Label(pl_frame, text=text, anchor="w", width=width)
        label.grid(row=1, column=0)        

        # create tooltip for title of program frame
        tooltip = StaticToolTip(label, text)

        # second frame on indented steps
        step_frame = tk.Frame(pl_frame, padx=5, pady=5, background="white")   # , background="orange")
        step_frame.configure(highlightbackground='#aaa', highlightthickness=1)    # add a border
        step_frame.grid(row=2)

        # double-click on program frame navigates to that row
        step_frame.bind("<Double-Button-1>", lambda a: self.show_rows_at(row+1, center=True))

        for key in ["input", "q", "k", "v", "output"]:
            if key == "input":
                if row == 0:
                    value = "<from token/position embeddings>"
                else:
                    value = "<from residual stream>"
            else:
                value = self.remove_quotes_from_dict_entries(player[key])

            text = "{}: {}".format(key, value)
            label = tk.Label(step_frame, text=text, anchor="w", width=width-8, bg="white", fg="black")
            label.grid()  

            # double-click on program frame navigates to that row
            label.bind("<Double-Button-1>", lambda a: self.show_rows_at(row+1, center=True))

            self.program_labels[pl_step] = label  
            pl_step += 1

    def remove_quotes_from_dict_entries(self, dict):
        '''
        Remove quotes from dict entries, for display purposes
        '''
        result = "{"
        for key, value in dict.items():

            if len(result) > 1:
                result += ", "

            result += "{}: {}".format(key, value)

        result += "}"
        return result
    
    def set_program_step(self, step):

        if self.current_program_label:
            self.current_program_label.config(bg="white", fg="black")
            self.current_program_label = None

        if self.program_labels and step in self.program_labels:
            self.current_program_label = self.program_labels[step]
            self.current_program_label.config(bg="green", fg="white")

    def create_root(self):
        # root window
        root = tk.Tk()

        # change window icon to something not so dark that it can't be seen on Windows tray
        root.iconbitmap('python_icon.ico')

        root.geometry('1100x800')
        self.tk_root = root

        # size of root TK Window
        # if self.font_size == 16:
        #     #root.geometry('2620x750')
        #     root.geometry('3220x780')
        # else:
        #     root.geometry('2030x600')

        #root.resizable(False, False)

        self.update_window_title()
        self.setup_fonts()

    def update_window_title(self):
        if self.tk_root:
            title = "DatExplorer (watching registers: {})".format(", ".join(self.watch_registers))
            self.tk_root.title(title) 

    def create_scrollable_frame(self, parent,  horizontal=True, vertical=True):
        # Create a Canvas and Scrollbars
        canvas = tk.Canvas(parent)    # , background="red")  
        canvas.grid(row=0, column=0, sticky="new")

        if horizontal:
            x_scrollbar = tk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
            x_scrollbar.grid(row=1, column=0, sticky="ew")
            canvas.configure(xscrollcommand=x_scrollbar.set)

        if vertical:
            y_scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
            y_scrollbar.grid(row=0, column=1, sticky="ns")
            canvas.configure(yscrollcommand=y_scrollbar.set)

        # Configure Grid Weights
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Create a Frame inside the Canvas
        frame = tk.Frame(canvas)   # , background="pink")
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Configure Scroll Functionality
        def scroll_function(*args):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", scroll_function)

        return frame, canvas

    def rebuild_grid(self):
        # self.layer_instances = [UIGridLayer(self.max_displayed_cols) for i in range(self.max_displayed_layers)]
        # self.create_layers_ui(self.lframe, row=0)
        self.create_ui(self.prompt, self.gold_text)

    def create_ui(self, prompt_text, gold_text):
        '''
        UI design:
            root holds a single child: frame0
            frame0 holds 3 children:
                1. canvas
                    - holds layers_frame (layers x cols grid of registers)
                2. control_frame
                    - holds controls (prompt textbox, next button, etc.)
                3. program_frame
                    - holds programs for each layer (layed out horizontally)

        '''

        self.layer_instances = [UIGridLayer(self.max_displayed_cols) for i in range(self.max_displayed_layers)]

        # create frame0 (holds canvas, control_frame, pld_frame)
        frame0 = tk.Frame(self.tk_root)   # , bg="green")
        frame0.grid(row=0, sticky="new", padx=25, pady=25)
        self.frame0 = frame0

        self.lift_message()

        # fill window/root with frame0
        self.tk_root.grid_rowconfigure(0, weight=1)
        self.tk_root.grid_columnconfigure(0, weight=1)

        # row 0: add LAYERS_FRAME 
        lframe = tk.Frame(frame0, padx=5, pady=5)   # , bg="blue")
        lframe.grid(row=0, column=0, columnspan=1, sticky="nswe")
        self.layers_frame = lframe
        self.create_layers_ui(lframe, row=0)

        # row 1: add CONTROL_FRAME 
        control_frame = tk.Frame(frame0, padx=10, pady=10)   # , bg="gray")
        control_frame.configure(highlightbackground='#aaa', highlightthickness=1)    # add a border
        control_frame.grid(row=1)

        self.build_control_panel(control_frame, prompt_text, gold_text)

        # row 2: add PROGRAM_FRAME 
        pframe = tk.Frame(frame0, border=0, padx=5, pady=5)   # bg="green"
        pframe.grid(row=2, column=0, sticky="nwe")
        self.build_program_frame(pframe)

        #  add EXAMPLE_DROPDOWN_FRAME (row 0)
        self.fill_example_dropdown_frame(control_frame, row=0, prompt_width=8)

        # add PROGRAM_DROPDOWN_FRAME (row 1)
        self.fill_program_dropdown_frame(control_frame, row=1, prompt_width=8)

        # add PROGRAM_DROPDOWN_FRAME (row 2)
        self.fill_model_dropdown_frame(control_frame, row=2, prompt_width=8)

        # finish layout stuff
        frame0.grid_rowconfigure(0, weight=1)          # expand vertically to fill any remaining space
        frame0.grid_columnconfigure(0, weight=1)

    def create_tk_label(self, frame, row, column, text):
        label = tk.Label(frame, text=text, borderwidth=0, relief="solid", width=4, anchor="w")
        label.grid(row=row, column=column, sticky="nsew", padx=0, pady=0)
        return label

    def create_tk_option_menu(self, frame, row, column, menu_variable, options, command):
        # NOTE: we cannot set the width; it is determined by the longest option by tkinter
        menu = tk.OptionMenu(frame, menu_variable, *options, command=command)
        menu.grid(row=row, column=column, sticky="we")
        menu.config(anchor="w")

        return menu

    def fill_program_dropdown_frame(self, control_frame, row, prompt_width):

        # add program label/dropdown
        text = "Program:"
        # label = tk.Label(control_frame, text=text, fg="black", anchor="e", width=prompt_width)
        # label.grid(row=row, column=2, sticky="nsew", padx=(0, 0))
        label = self.create_tk_label(control_frame, row, 2, text)

        options = self.find_avail_programs()
        program_dropdown_variable = tk.StringVar(control_frame)  # pframe)
        program_dropdown_variable.set(self.current_program_name) # default value

        program_dropdown = self.create_tk_option_menu(control_frame, row, 3, program_dropdown_variable, options, self.load_new_program)

        # Bind the mouse click event to the dropdown
        program_dropdown.bind("<Button-1>", self.refresh_program_list)

        self.program_dropdown = program_dropdown
        self.program_dropdown_variable = program_dropdown_variable

    def refresh_program_list(self, tk_event):
        files = self.find_avail_programs()

        menu = self.program_dropdown['menu']
        menu.delete(0, 'end')

        # Add the new options to the menu
        for fn in files:
            menu.add_command(label=fn, command=lambda value=fn: self.load_new_program(value))

    def fill_example_dropdown_frame(self, control_frame, row, prompt_width):

        # add program label/dropdown
        text = "Example:"
        # label = tk.Label(control_frame, text=text, fg="black", anchor="e", width=prompt_width)
        # label.grid(row=row, column=2, sticky="nsew", padx=(0, 0))
        label = self.create_tk_label(control_frame, row, 2, text)

        options = list(self.examples)
        example_dropdown_variable = tk.StringVar(control_frame)  # pframe)
        example_dropdown_variable.set(self.example) # default value

        # example_dropdown = tk.OptionMenu(control_frame, example_dropdown_variable, *options, command=self.load_new_example)
        # example_dropdown.grid(row=row, column=3, sticky="we")
        # example_dropdown.config(anchor="w")
        example_dropdown = self.create_tk_option_menu(control_frame, row, 3, example_dropdown_variable, options, self.load_new_example)

        # Bind the mouse click event to the dropdown
        example_dropdown.bind("<Button-1>", self.refresh_example_list)

        self.example_dropdown = example_dropdown
        self.example_dropdown_variable = example_dropdown_variable

    def refresh_example_list(self, tk_event):
        examples = self.read_examples_from_file(self.fn_examples)

        menu = self.example_dropdown['menu']
        menu.delete(0, 'end')

        # Add the new options to the menu
        for ex_name in examples:
            menu.add_command(label=ex_name, command=lambda value=ex_name: self.load_new_example(value))

    def fill_model_dropdown_frame(self, control_frame, row, prompt_width):

        # add model label/dropdown
        text = "Model:"
        # label = tk.Label(control_frame, text=text, fg="black", anchor="e", width=prompt_width)
        # label.grid(row=row, column=2, sticky="nsew", padx=(0, 0))
        label = self.create_tk_label(control_frame, row, 2, text)

        model_dropdown_variable = tk.StringVar(control_frame)  # pframe)
        model_dropdown_variable.set(self.model_name) # default value

        # model_dropdown = tk.OptionMenu(control_frame, model_dropdown_variable, *self.model_names, command=self.load_new_model)
        # model_dropdown.grid(row=row, column=3, sticky="we")
        # model_dropdown.config(anchor="w")
        model_dropdown = self.create_tk_option_menu(control_frame, row, 3, model_dropdown_variable, self.model_names, self.load_new_model)

    def load_new_model(self, model_name):
        self.model_name = model_name
        self.model = None     # needs loading
        #self.on_psl_program_changed(self.program_name)

    def set_tracing(self):
        self.tracing_enabled = self.cb_tracing_var.get()

    def build_control_panel(self, control_frame, prompt_text, gold_text):
        # prompt input
        for i in range(6):
            control_frame.columnconfigure(1+i, pad=30)

        # row 0: Prompt
        prompt = tk.Label(control_frame, text="Prompt: ")
        #prompt.bind('<KeyRelease>', self.on_example_changed)
        
        prompt.grid(row=0, column=0, sticky="w")

        prompt_width = 80
        prompt_entry = tk.Entry(control_frame, width=prompt_width)
        #prompt_entry.bind('<KeyRelease>', self.on_example_changed)
        
        # seems to need its own font
        entry_font = tkfont.Font(family="Arial", size=self.font_size)
        prompt_entry.config(font=entry_font)

        prompt_entry.grid(row=0, column=1, sticky="w")
        prompt_entry.insert(0, prompt_text)
        self.prompt_entry = prompt_entry

        next = tk.Button(control_frame, text="Run", command=lambda: self.run_program(), width=6)
        next.grid(row=1, column=4, sticky="wsne", padx=(10,0))

        cb_tracing_var = tk.BooleanVar()
        cb_tracing_var.set(False)  # Initialize checkbutton as unchecked
        self.cb_tracing_var = cb_tracing_var

        btTrace = tk.Checkbutton(control_frame, text="Trace steps", variable=cb_tracing_var, command=self.set_tracing, width=7)
        btTrace.grid(row=0, column=4, sticky="w", padx=(10,0))

        self.tk_root.bind("<F5>", lambda e: self.run_program())
        self.tk_root.bind("<Home>", lambda e: self.show_rows_at(1))
        self.tk_root.bind("<End>", lambda e: self.show_rows_at(self.num_layers))
        self.tk_root.bind("<Up>", lambda e: self.show_rows_at(self.first_displayed_row+1))
        self.tk_root.bind("<Down>", lambda e: self.show_rows_at(self.first_displayed_row-1))

        # reset = tk.Button(control_frame, text="Reset", command=lambda: self.reset_all(True), width=10)
        # reset.grid(row=0, column=3, sticky="w")

        # GOLD frame
        # row 1: "Gold:" label
        gold_prompt = tk.Label(control_frame, text="Gold: ")  # , width=10)   # , bg="gray", fg="white")
        gold_prompt.grid(row=1, column=0, sticky="w")
        
        # make approx the same width as prompt_entry (but will vary by font size, screen DPI, etc.)
        prompt_width = 80
        gold_entry = tk.Entry(control_frame, width=prompt_width)
        gold_entry.grid(row=1, column=1, sticky="w")
        gold_entry.config(font=entry_font)
        gold_entry.insert(0, gold_text)
        self.gold_entry = gold_entry

        # RESPONSE frame
        # row 2: Response
        response_label = tk.Label(control_frame, text="Response: ")  # , width=10)   # , bg="gray", fg="white")
        response_label.grid(row=2, column=0, sticky="w")
        
        # make approx the same width as prompt_entry (but will vary by font size, screen DPI, etc.)
        response_value = tk.Label(control_frame, text="", anchor="w", bg="#cccccc", width=prompt_width-10)
        response_value.grid(row=2, column=1, sticky="w")
        response_value.config(font=entry_font)
        self.response_label = response_value

    def show_rows_at(self, row_number, center=False):
        if not self.raw_layer_steps:
            return

        # row_number is 1-based
        max_first_row = self.num_layers - self.max_displayed_layers + 1

        if center:
            row_number = row_number - self.max_displayed_layers // 2

        if row_number < 1:
            row_number = 1
        elif row_number > max_first_row:
            row_number = max_first_row

        self.first_displayed_row = row_number

        # update grid cell label for each of our N rows
        row_count = self.max_displayed_layers
        for r in range(row_count):
            for c in range(self.max_displayed_cols):
                cell_name = "L{}-C{}".format(self.first_displayed_row + r, 1+c)
                key = "L{}-C{}".format(1+r, 1+c)
                label = self.grid_cell_name_labels[key]
                label.config(text=cell_name)

        # update grid cell CONTENTS
        for i, row_num in enumerate(range(self.first_displayed_row, row_count + self.first_displayed_row)):

            if self.jit_decoding:
                self.ensure_row_is_decoded(row_num)

            for key_index in range(6):
                layer_inst = self.layer_instances[i]
                self.update_grid_cell(row_num-1, key_index, layer_inst)

            # set weight label to show the column we attended to
            sub_layer_values_by_col = self.layer_steps[row_num-1]["attn_weights"]
            for c, values in enumerate(sub_layer_values_by_col):                
                col_inst = layer_inst.columns[c]
                if col_inst.wlabel:
                    if values.sum() == 0:
                        # no column attended to
                        text = " "
                        col_inst.wlabel.config(text=text)
                    else:
                        col_index = np.argmax(values)
                        text = "attn: C{}".format(1+col_index)

                        if col_index == c:
                            # attended to self
                            col_inst.wlabel.config(text=text, fg="blue", bg="#eee")
                        else:
                            # attended to another column
                            col_inst.wlabel.config(text=text, fg="green", bg="#eee")

    def create_tokens_line(self, frame0, row):
        # tok_frame = tk.Frame(frame0)  
        # tok_frame.grid(row=row, column=1, sticky="nsew")  # , padx=25, pady=25)

        for c in range(self.num_cols):
            label = tk.Label(frame0, text="", fg="black")  # , height=5)
            label.grid(row=row, column=1+c, padx=5, pady=10, sticky="ew")
            self.token_labels.append(label)

    def update_tokens_line(self, tokens):

        count = len(self.token_labels) // 2

        for c, token in enumerate(tokens):
            if c >= count:
                break

            # first line
            self.token_labels[c].config(text=token)

            # second line
            self.token_labels[c+count].config(text=token)
            

    def create_layers_ui(self, frame0, row):
        outer_layers_frame, canvas = self.create_scrollable_frame(frame0)

        # tkinter bug workaround: need a 2nd layer to keep scrollbars working correctly
        layers_frame = tk.Frame(outer_layers_frame)  # , bg="blue")
        layers_frame.grid(row=0, column=0, sticky="new")

        self.token_labels = []

        # upper tokens line
        self.create_tokens_line(layers_frame, row=0)

        # build UI for each LAYER
        num_layers = self.num_layers
        num_cols = self.num_cols 
        tt_children = []

        build_layer_count = self.max_displayed_layers   #  min(num_layers, self.max_displayed_layers)

        for r in range(build_layer_count):

            # 1-relative row number, in reverse order
            row = 1 + (build_layer_count - r - 1)
            #lframe = tk.Frame(layers_frame)      # , bg="gray")
            #layers_frame.grid(row=gui_row, column=1, pady=(5, 15))

            layer_inst = self.layer_instances[row-1]
            layer_inst.lframe = layers_frame

            # build UI for each COLUMN
            for c in range(num_cols):

                cell_name = "L{}-C{}".format(row, 1+c)
                cell_inst = layer_inst.columns[c]
                tt_kids = self.create_grid_cell(r+1, cell_name, cell_inst, layers_frame, row-1, c)
                tt_children += tt_kids
                #layers_frame.columnconfigure(1+c, pad=15)
                #print("created column for layer: {}, col: {}".format(r, c))
 
        for c in range(self.num_cols):
            layers_frame.columnconfigure(1+c, pad=10)

        # bottom tokens line
        self.create_tokens_line(layers_frame, row=1+build_layer_count)

        self.tooltip = DynamicToolTip(layers_frame, tt_children, self.fill_tooltip_for_label)

        # tkinter WORKAROUND: adjust the canvas height to the that of its contents
        canvas.update_idletasks()
        bbox = layers_frame.bbox(tk.ALL)
        canvas.configure(height=bbox[3] - bbox[1])      

        self.grid_canvas = canvas

    def scroll_to_layer(self, layer_num):
        # for now, hardcode to scroll to the first layer
        self.grid_canvas.update_idletasks()
        self.grid_canvas.yview_moveto(1.0)

    def make_table(self, parent, data, table_title):
        # add title
        title = tk.Label(parent, text=table_title)
        title.grid(row=0, column=0, sticky='ew')

        # Add labels for column headers

        # register column
        header1 = tk.Label(parent, text='Reg', bg='grey', fg='white')
        header1.grid(row=1, column=0, sticky='ew')

        # value column
        header2 = tk.Label(parent, text='Val', bg='grey', fg='white')
        header2.grid(row=1, column=1, sticky='ew')

        # table style
        relief = "raised"    # solid, groove, ridge, sunken, raised, flat
        borderwidth = .5

        # Add labels for table data
        for i, values in enumerate(data, start=1):

            # add register name
            label1 = tk.Label(parent, text=values[0], borderwidth=borderwidth, relief=relief)
            label1.grid(row=1+i, column=0, sticky='ew')

            # add register value
            label2 = tk.Label(parent, text=values[1], borderwidth=borderwidth, relief=relief)
            label2.grid(row=1+i, column=1, sticky='ew')

        #parent.configure(highlightthickness=1, highlightbackground="red")
        parent.configure(borderwidth=1, relief="solid")

    def fill_tooltip_for_label(self, event, parent):
        label = event.widget

        layer_index = label.layer + self.first_displayed_row - 1
        column_index = label.column
        sub_layer_index = label.sub_layer

        if self.layer_steps is None:
           return "<program not loaded>"
        
        # these keys are top-down order
        key = SUBLAYER_KEYS[sub_layer_index]

        key_steps = self.layer_steps[layer_index]
        col_steps = key_steps[key]
        if column_index >= len(col_steps):
            return "<column not yet set>"

        steps = col_steps[column_index]
        reg_value_pairs = [(key, steps[index]) for key, index in self.register_name_to_index.items()]

        table_title = "L{}-C{}\n{}".format(layer_index+1, column_index+1, key)
        self.make_table(parent, reg_value_pairs, table_title)

        return True

    def scroll_to_widget(self, canvas, widget):
        # Get the bounding box of the widget
        bbox = canvas.bbox(widget)

        # Calculate the fractions to move in x and y directions
        x_fraction = bbox[0] / canvas.winfo_width()
        y_fraction = bbox[1] / canvas.winfo_height()

        # Scroll to the widget
        canvas.xview_moveto(x_fraction)
        canvas.yview_moveto(y_fraction)              

    def find_avail_programs(self):
        from glob import glob

        path = "psl_programs/*.yaml"
        files = glob(path)
        files = [os.path.basename(fn).split(".", 1)[0] for fn in files]

        return files

def parse_args():
    program_name = "icl_parser_gen"
    fn_examples = "examples.yaml"

    # Create the parser
    parser = argparse.ArgumentParser(add_help=False, description="A GUI program to explore the inner workings of the DAT transformer/interpreter")

    # Add the arguments
    parser.add_argument("--example", type=str, nargs="*", help="name of example to initially select")

    parser.add_argument("--psl", 
                        type=str, 
                        default=program_name,
                        help="the name of the PSL program to initially load")

    parser.add_argument("--examples",
                        type=str,
                        default=fn_examples,
                        help="The path to the file containing examples prompt and gold pairs")

    parser.add_argument("--max_display_rows",
                        type=int,
                        default="3",
                        help="The maximum number of layers/rows to display")

    parser.add_argument("--interpreter", action="store_true", help="initially select the DAT interpreter as the model", default=0)
    parser.add_argument("--help", action="store_true", help="print information command line arguments", default=0)

    # Execute the parse_args() method
    args = parser.parse_args()

    # don't fixup args.psl for explorer (will only be removed later)

    if not args.example:
        # passing None will select the first example of fn_examples
        args.example = None
    elif isinstance(args.example, list):
        args.example = args.example[0]

    return args, parser

def usage(parser):

        parser.print_help()
        # print("Usage: python dat_transformer.py [ <options> ]")
        # print("   <options>: <example> | all | --tgt <task_name> | --prompt <prompt> --gold <gold> | --fn_psl <psl_file_path> | --trace")

        print()
        print("examples:")
        print("  > python dat_explorer.py                         (run with default PSL program as initially selected program)")
        print("  > python dat_explorer.py --psl test_prog.yaml    (initially select psl_programs/test_prog.yaml)")
        print("  > python dat_explorer.py --help                  (show this help text)")

def main():
    args, parser = parse_args() 

    if args.help:
        usage(parser)
        exit(0)

    d_register = DEFAULT_D_REG

    ts = DatExplorer(args.psl, args.example, args.examples, d_register=d_register, n_heads=1,  
        max_display_rows=args.max_display_rows, use_productions_vocab=True, interpreter=args.interpreter)

    ts.start_ui_loop()

if __name__ == "__main__":
    main()  