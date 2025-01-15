# trace_generator.py: helper class to generate step traces from transformer or interpreter
import os

class TraceGenerator():
    def __init__(self, example_name, decoder, all_gens=False):
        self.decoder = decoder
        self.trace_layers = "all"  
        self.trace_cols = "all"  
        self.trace_gens = "all"  
        self.trace_step_name = "residual"     # actual output of layer
        self.only_trace_last_col_of_new_gens = (not all_gens)

        self.fn_trace = "example_traces/{}.yaml".format(example_name)
        os.makedirs("example_traces", exist_ok=True)
        self.trace_file = open(self.fn_trace, "w")

        print("tracing layer output for each generation pass")

    def trace(self, gen_num, curr_gen_steps):
        if self.trace_gens == "all" or gen_num in self.trace_gens:
            if self.trace_layers == "all":
                self.trace_layers = range(len(curr_gen_steps))

            if self.trace_cols == "all":
                col_count = len(curr_gen_steps[0]["input"])
                if self.only_trace_last_col_of_new_gens and gen_num > 0:
                    trace_cols_actual = [col_count-1]    # only trace the last column of each generation pass
                else:   
                    trace_cols_actual = range(col_count)
            else:
                trace_cols_actual = self.trace_cols

            self.trace_steps(gen_num, curr_gen_steps, trace_cols_actual, self.trace_step_name)

    def trace_steps(self, gen_num, curr_gen_steps, trace_cols, step_name):

        step_names = ["input", "query", "key", "value", "output", "residual"] if step_name == "all" else [step_name]
        layer_text = ""

        for li in self.trace_layers:
            layer = curr_gen_steps[li]
            col_count = len(layer["input"])

            for col_num in trace_cols:  
                for step_name in step_names:
                    layer_step = layer[step_name]
                    col_values = layer_step[col_num]

                    if step_name == "keys" and li == 6:
                        pass  # debug
                    
                    decoded_col = self.decoder.decode_step(col_values, return_pairs=True)

                    text = "{" + "g: {}, 'L{}.C{}', {}: {}".format(gen_num, 1+li, 1+col_num, step_name, str(decoded_col)) + "}\n"
                    layer_text += text

                if len(step_names) > 1:
                    layer_text += "#----------------\n"    # end of steps within a col

            if len(trace_cols) > 1:
                layer_text += "#================\n"        # end of cols within a layer

            self.trace_file.write(layer_text)
            layer_text = ""

        layer_text += "#++++++++++++++++\n"            # end of layers within a gen pass
        self.trace_file.write(layer_text)

    def close(self):
        self.trace_file.close()
        print("tracing output saved to: {}".format(self.fn_trace))

