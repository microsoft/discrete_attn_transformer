# llm_tester.py: test our nc_tgt dataset against GPT-3 and GPT-4 turbo models.
import os
import json
import sys
import time
import argparse
import random

from utils.dat_utils import get_dataroot, get_time_str, load_dataset_task_split

model_names = {
    "o1": "o1-2024-12-17",             
    "o1-preview": "o1-preview",             
    "o1-mini": "o1-mini",             
    "gpt-4": "gpt-4",             
    "gpt-4-turbo": "gpt-4-turbo", 
    "gpt-4o-prev": "gpt-4o-2024-05-13",  #"GPT-4o", 
    "gpt-4o": "gpt-4o-2024-08-06",  # quietly updated to GPT-4o-2024-08-06
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",  
    "gpt-3.5": "gpt-3.5-turbo",     # OpenAI recommends using GPT-4o instead of GPT-3.5

    "phi-3": "microsoft/Phi-3-medium-128k-instruct",     # doesn't even guess; just comments on what the task appears to be

    "claude-3.5-sonnet-new": "claude-3-5-sonnet-20241022",  
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    
    "claude-3-opus": "claude-3-opus-20240229", 

    # together API    
    "toget-llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "toget-llama-3.1-70b": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "toget-llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",

    # groq API
    "groq-llama-3.1-405b": "llama-3.1-405b-reasoning",
    "groq-llama-3.1-70b": "llama-3.1-70b-versatile",
    "groq-llama-3.1-8b": "llama-3.1-8b-instant",

    # XAI API
    "grok2": "grok-beta",

    # google API
    "google-llama-3.1-405b": "meta/llama3-405b-instruct-maas",

    # gemini API (simplier google API)
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
    "gemini-1.5-pro-0827": "gemini-1.5-pro-exp-0827",
    "gemini-1.5-pro-0801": "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",

    # hyperbolic.xyz API
    "hyperbolic-llama-3-70B": "hyper$meta-llama/Meta-Llama-3-70B-Instruct",
    "hyperbolic-llama-3.1-405b": "hyper$meta-llama/Meta-Llama-3.1-405B-Instruct",
    "hyperbolic-reflection-70b": "hyper$mattshumer/Reflection-Llama-3.1-70B",
    
}

use_enhanced_sys_prompt = True

if use_enhanced_sys_prompt:
    sys_prompt = "You are a helpful assistant; please complete the following abstract pattern exactly once. " \
        + " The pattern contains an example question/answer pair, followed by a second question and a missing answer. " \
        + " Do not output anything except the final answer. Pay close attention to all special characters." 
else:
    sys_prompt = "You are a helpful assistant; please complete the following abstract pattern exactly once. " \
        + " Do not output anything except the final answer. Pay close attention to all special characters." 

# sys_prompt_x = '''
# Take a deep breath and focus.  You are an expert in learning and completing in-context, template-driven abstract patterns.
# You will be presented with one or more input/output pairs that follow a specific template pattern. Your job is to generate the correct output for a new input based on the patterns seen in the examples.

# Instructions:
# 1. Each input starts with a "Q" ends with an "A"
# 2. Each output ends with a period.
# 3. Inputs and outputs consist of abstract symbols formed by sequences of one or more 2-letter combinations.
# 4. Inputs and outputs may contain 1 or more delimiters (special characters) that follow a consistent pattern.

# # Your task: Generate the correct output for the last input, based on the preceeding examples.
# '''

grammar_prompt = "The grammar for these patterns can be described as follows: " + \
"""
<tgt>               ::= <example> <cue>
<example>           ::= Q <question> A <answer>
<cue>               ::= Q <question> A 
<question>          ::= <dc sequence>
<answer>            ::= <dc sequence>

<dc sequence>       ::= <constituent list> [ <delimiter> ]
<constituent list>  ::= <constituent> [ <delimiter> <constituent list> ]

<delimiter>         ::= <symbol> [ <delimiter> ] 
<constituent>       ::= <symbol> [ <constituent> ]
"""        

class LlmTester():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.use_grammar = None
        self.sys_prompt = sys_prompt    
        self.log_file = None

    def set_use_grammar(self, value):
        self.use_grammar = value

        if self.use_grammar:
            self.sys_prompt = sys_prompt + "\n" + grammar_prompt
        else:
            self.sys_prompt = sys_prompt

    def create_phi3_model(self, model_id):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        print("loading model: {}...".format(model_id))

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def process_phi3_request(self, user_prompt, model_id):
        
        if self.model is None:
            self.create_phi3_model(model_id)

        generation_args = {"max_new_tokens": 50, "return_full_text": False, "temperature": 0.0, "do_sample": False}

        system_message = "You are a expert in learning and completing in-context, template-drivien abstract patterns.  " 
        if self.use_grammar:
            system_message += grammar_prompt

        system_message += "Do not output additonal examples." \
            + "Please complete the following abstract pattern exactly once, outputting only the missing symbols:"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        output = self.pipe(messages, **generation_args)
        answer = output[0]['generated_text']

        return answer
            
    def get_together_completion(self, user_prompt, model_id):
        from together import Together

        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_prompt},
            ]

        # create() uses the following defaults: temperature=None, top_p=None, top_k=None, 
        result = client.chat.completions.create(model=model_id, messages=messages, stream=False)
        text = result.choices[0].message.content

        return text

    def get_groq_completion(self, user_prompt, model_id):

        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_prompt},
            ]

        chat_completion = client.chat.completions.create(messages=messages, model=model_id)

        text = chat_completion.choices[0].message.content
        return text

    def get_anthropic_completion(self, user_prompt, model_id):
        import anthropic

        # caller does retries
        client = anthropic.Anthropic()

        message = client.messages.create(
            model=model_id,
            max_tokens=1000,
            temperature=0,
            system=self.sys_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt  
                        }
                    ]
                }
            ]
        )

        text = message.content[0].text    

        return text

    def get_xai_completion(self, user_prompt, model_id):
        import openai

        XAI_API_KEY = os.getenv("XAI_API_KEY")
        client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

        completion = client.chat.completions.create(model=model_id,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt},
            ])

        msg = completion.choices[0].message
        text = msg.content

        return text

    def get_google_completion(self, user_prompt, model_id):
        import vertexai
        import openai
        from google.auth import default, transport

        PROJECT_ID = "zinc-citron-430721-v1"
        LOCATION = "us-central1"  
        MODEL_LOCATION = "us-central1"

        credentials, _ = default()
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        vertexai.init(project=PROJECT_ID, location=LOCATION)   

        client = openai.OpenAI(
            base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
            api_key=credentials.token)

        #MODEL_ID = "meta/llama3-405b-instruct-maas" 

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_prompt},
            ]

        response = client.chat.completions.create(model=model_id, messages=messages, temperature=0)

        text = response.choices[0].message.content
        return text

    def get_gemini_completion(self, user_prompt, model_id):
        # use the VERTEXAI API so we can specify system prompt, etc.
        # NOTE: this requires you to login thru the browser to ...  
        # no API key used
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = "zinc-citron-430721-v1"

        vertexai.init(project=project_id, location="us-central1")

        model = GenerativeModel(model_id, system_instruction=[self.sys_prompt])  # , temperature=0)

        response = model.generate_content(user_prompt)    # , temperature=0)
        text = response.text.strip()

        # import google.generativeai as genai
        # import os
        # genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # model = genai.GenerativeModel('gemini-1.5-flash')

        return text


    def get_hyperbolic_completion(self, user_prompt, model_id):
        from openai import OpenAI
        HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
        client = OpenAI(api_key=HYPERBOLIC_API_KEY, base_url="https://api.hyperbolic.xyz/v1",)

        # recommended:  temperature of .7 and a top_p of .95.
        # and add "think carefully" to end of messages/system prompt

        sys_prompt = self.sys_prompt
        max_tokens = 4096
        temperature = 0
        top_p = 1
        think_carefully = False

        if think_carefully:
            sys_prompt += " Think carefully."
            max_tokens = 4096
            temperature = .7
            top_p = .95

        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}]

        #response = client.chat.completions.create(model=model_id, temperature=0, messages=messages, max_tokens=255)
        response = client.chat.completions.create(model=model_id, temperature=temperature, top_p=top_p, messages=messages, max_tokens=max_tokens)

        choice = response.choices[0]
        reply = choice.message.content

        if "<thinking>" in reply:
            pass

        if "<output>" in reply:
            # extract text between <output> and </output> tags
            reply = reply.split("<output>")[1]
            reply = reply.split("</output>")[0]
            reply = reply.strip()

        return reply
        
    def get_openai_completion(self, user_prompt, model_id):
        from openai import OpenAI
        client = OpenAI()

        # GPT-X models are processed here
        if model_id.startswith("o1-"):
            # currently doesn't support system prompt
            messages=[
                {"role": "user", "content": self.sys_prompt + "\n\n" + user_prompt}]
            
            # currently doesn't support max_tokens or temperature
            response = client.chat.completions.create(model=model_id, messages=messages)
        else:
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt}]

            response = client.chat.completions.create(model=model_id, temperature=0, messages=messages, max_tokens=255)

        choice = response.choices[0]
        reply = choice.message.content
        return reply

    def get_completion(self, user_prompt, model_id):
        text = None
        ex = None

        for i in range(10):
            try:
                if "phi-3" in model_id.lower():
                    answer = self.process_phi3_request(user_prompt, model_id)
                    text = answer.strip()

                elif model_id.startswith("meta-"):
                    text = self.get_together_completion(user_prompt, model_id)

                elif model_id.startswith("llama-"):
                    text = self.get_groq_completion(user_prompt, model_id)

                elif model_id.startswith("claude-"):
                    text = self.get_anthropic_completion(user_prompt, model_id)

                elif model_id.startswith("meta/"):
                    text = self.get_google_completion(user_prompt, model_id)

                elif model_id.startswith("gemini-"):
                    text = self.get_gemini_completion(user_prompt, model_id)

                elif model_id.startswith("grok-"):
                    text = self.get_xai_completion(user_prompt, model_id)

                elif model_id.startswith("hyper$"):
                    model_id = model_id[6:]
                    text = self.get_hyperbolic_completion(user_prompt, model_id)

                else:
                    text = self.get_openai_completion(user_prompt, model_id)

                break       # important!

            except Exception as ex:
                text = str(ex)
                print("RETRYING exception: " + text)
                rand_secs = 3*random.random()
                # if "limit exceeded" in str(e) or "rate limited" in str(e):
                #     print("rate limited; waiting...")
                #     time.sleep(1.2)
                # elif "likely blocked by the safety filters" in str(e):
                #     print("safety filters blocked response")
                #     text = "<safety filters blocked response>"
                #     break
                time.sleep(rand_secs)

        if not text:
            raise ex

        return text

    def log_print(self, str):
        print(str)
        if self.log_file:
            self.log_file.write(str + "\n")

    def limit_shot(self, example, shot_count):
        parts = example.split("\t")
        x, y, info = parts

        x_parts = x.split(" Q ")
        x_shots = x_parts[0:-1]
        new_shots = x_shots[:shot_count]
        new_x = " Q ".join(new_shots + [x_parts[-1]])

        return "{}\t{}\t{}".format(new_x, y, info)

    def get_regular_examples(self, dataset_name, version, task_name, split_name, split_filter=None):
        # v11 doesn't have a 7_shot, we can construct it dynamically
        shot_filter_count = None

        if task_name == "7_shot_rlw":
            task_name = "10_shot_rlw"
            shot_filter_count = 7
        elif task_name == "4_shot_rlw":
            task_name = "5_shot_rlw"
            shot_filter_count = 4

        examples = load_dataset_task_split(dataset_name, version, task_name, split_name)

        if split_filter:
            examples = [ex for ex in examples if split_filter in ex]  

        if shot_filter_count:
            examples = [self.limit_shot(ex, shot_filter_count) for ex in examples]

        return examples

    def get_table_lookup_examples(self, dataset, version, task_name, split_name):

        # form the N-shot examples from the train split
        train_examples = self.get_regular_examples(dataset, version, task_name, "train")
        n_shot = ""
        for example in train_examples:
            x, y, info = example.split("\t")
            xx = x.replace(".", "=>")
            yy = y.split(" ")[-1]
            shot = "{} {} ; ".format(xx, yy)
            n_shot += shot

        test_examples = self.get_regular_examples(dataset, version, task_name, split_name)
        new_test_examples = []
        for example in test_examples:
            x, y, info = example.split("\t")
            xx = x.replace(".", "=>")
            xxx = n_shot + xx
            yy = y.split(" ")[-1]
            new_example = "{}\t{}\tn_shot".format(xxx, yy)
            new_test_examples.append(new_example)

        return new_test_examples

    def get_ood_lexical_examples(self, dataset, version, task_name, split_name):
        train_examples = self.get_regular_examples(dataset, version, task_name, "train")
        new_examples = []

        # carefully uppercase all cue/gold text
        for train in train_examples:
            train_x, train_y, train_info = train.split("\t")  
            if "echo" in train_info:
                # omit the echo examples
                continue

            train_x_qs = train_x.split(" Q ")
            train_x_qs[-1] = train_x_qs[-1].upper()
            new_train_x = " Q ".join(train_x_qs)
            new_train_y = train_y.upper()

            example = "{}\t{}\t{}".format(new_train_x, new_train_y, train_info)
            new_examples.append(example)

        return new_examples

    def extract_template(self, cue, gold):

        # from tpx-datasets/our_examples/nc_tgt/data_gen.py
        delimiters = [":", "!", "#", "%", "^", "&",  "*", "(", ")", "_", "-", "+", "=", "!=", 
            "{", "}", "[", "]", "|", "/", "?", ",", ";", ":", "<", ">", "~", "`", "@", "$"]

        cue_tokens = cue.split(" ")
        gold_tokens = gold.split(" ")

        cue_template = []
        gold_template = []

        next_cue_arg = 1
        last_token_was_cons = False
        con_dict = {}

        # build cue template
        for ct in cue_tokens:
            if ct in delimiters or ct in ["Q", "A"]:
                cue_template.append(ct)
                last_token_was_cons = False
            else:
                if not last_token_was_cons:
                    arg_name = "arg{}".format(next_cue_arg)
                    cue_template.append(arg_name)
                    next_cue_arg += 1
                    last_token_was_cons = True
                    con_dict[ct] = arg_name

        # build gold template
        last_token_was_cons = False
        for gt in gold_tokens:
            if gt in delimiters or gt in ["."]:
                gold_template.append(gt)
                last_token_was_cons = False
            else:
                if not last_token_was_cons:
                    arg = con_dict[gt]
                    gold_template.append(arg)
                    last_token_was_cons = True

        return cue_template, gold_template, con_dict

    def convert_cons_len(self, cue, gold, new_lengths_list):
        cue_parts = cue.split(" ")
        gold_parts = gold.split(" ")

        x_template, y_template, con_dict = self.extract_template(cue, gold)
        arg_dict = {}

        # build arg dict (the actual constituent tokens for each arg)
        arg_count = len(con_dict)
        
        # pick the const lengths
        cons_lens = random.choices(new_lengths_list, k=arg_count)
        total_cons = sum(cons_lens)

        # build a vocab of all 2 letter combinations
        rl_words = [a+b for a in "abcdefghijklmnopqrstuvwxyz" for b in "abcdefghijklmnopqrstuvwxyz"]   

        # pick random constituent parts with no overlaps
        cons_parts = random.sample(rl_words, total_cons)

        for i, arg_name in enumerate(con_dict.values()):
            cons_len = cons_lens[i]
            arg_parts = [cons_parts.pop() for _ in range(cons_len)]
            arg_dict[arg_name] = " ".join(arg_parts)

        # build new cue from x_template and arg_dict
        new_cue_parts = []
        for ct in x_template:
            if ct in arg_dict:
                constituent = arg_dict[ct]
                new_cue_parts.append(constituent)
            else:
                new_cue_parts.append(ct)

        # build new gold from y_template and arg_dict
        new_gold_parts = []
        for gt in y_template:
            if gt in arg_dict:
                constituent = arg_dict[gt]
                new_gold_parts.append(constituent)
            else:
                new_gold_parts.append(gt)

        new_cue = " ".join(new_cue_parts)
        new_gold = " ".join(new_gold_parts)

        return new_cue, new_gold, cons_lens

    def get_ood_cons_len_examples(self, dataset, version, task_name, split_name, max_examples):
        train_examples = self.get_regular_examples(dataset, version, task_name, "train")
        new_examples = []

        # carefully convert all cue/gold text constituents from len of [1, 2, 4] to len of [7]
        for train in train_examples:
            train_x, train_y, train_info = train.split("\t")  

            if "echo" in train_info:
                # omit the echo examples
                continue

            train_x_qs = train_x.split(" Q ")
            train_cue = train_x_qs[-1]

            new_train_cue, new_train_y, cons_lens = self.convert_cons_len(train_cue, train_y, new_lengths_list=[7])
            
            train_x_qs[-1] = new_train_cue
            new_train_x = " Q ".join(train_x_qs)

            # update the train_info to reflect the new cue/gold lengths
            ti_data = json.loads(train_info)
            #print("ti_data: " + str(ti_data))

            ti_len_parts = ti_data["cons_len"].split(".")
            ti_len_parts[-1] = "Q" + "".join([str(c) for c in cons_lens])

            new_ti_data = ".".join(ti_len_parts)
            ti_data["cons_len_7"] = new_ti_data
            train_info = json.dumps(ti_data)

            example = "{}\t{}\t{}".format(new_train_x, new_train_y, train_info)
            new_examples.append(example)

            if len(new_examples) >= max_examples:
                break   

        return new_examples

    def get_rev_cons_len_examples(self, dataset, version, task_name, split_name, max_examples):
        train_examples = self.get_regular_examples(dataset, version, task_name, "train")
        new_examples = []

        # carefully convert all N-shot examples to use len=7 constituents 
        for train in train_examples:
            train_x, train_y, train_info = train.split("\t")  

            if "echo" in train_info:
                # omit the echo examples
                continue

            train_x_space = " " + train_x           # easier to work with if all parts start with " Q "
            train_x_qs = train_x_space.split(" Q ")
            train_cue = train_x_qs[-1]

            new_train_qs = []
            actual_train_x_qs = train_x_qs[1:-1]

            for shot in actual_train_x_qs:
                shot_x, shot_y = shot.split(" A ")
                shot_x += " A"
                new_x, new_y, cons_lens = self.convert_cons_len(shot_x, shot_y, new_lengths_list=[7])
                new_shot = new_x + " " + new_y
                new_train_qs.append(new_shot)
            
            new_train_qs.append(train_cue)
            new_train_x = "Q " + " Q ".join(new_train_qs)

            # update the train_info to reflect the new cue/gold lengths
            ti_data = json.loads(train_info)
            #print("ti_data: " + str(ti_data))

            ti_len_parts = ti_data["cons_len"].split(".")
            new_q_part = "Q" + "".join([str(c) for c in cons_lens])
            new_ti_len_parts = [new_q_part] + [ti_len_parts[-1]]

            new_ti_data = ".".join(new_ti_len_parts)
            ti_data["cons_len"] = new_ti_data
            train_info = json.dumps(ti_data)

            example = "{}\t{}\t{}".format(new_train_x, train_y, train_info)
            new_examples.append(example)

            if len(new_examples) >= max_examples:
                break   

        return new_examples

    def get_dyn_stutter_examples(self, dataset, version, task_name, split_name, max_examples, stutter_count):
        train_examples = self.get_regular_examples(dataset, version, task_name, "train")
        new_examples = []

        # replace N-shot examples with first example repeated N times
        for train in train_examples:
            train_x, train_y, train_info = train.split("\t")  

            if "echo" in train_info:
                # omit the echo examples
                continue

            train_x_space = " " + train_x           # easier to work with if all parts start with " Q "
            train_x_qs = train_x_space.split(" Q ")

            train_shots = train_x_qs[1:-1]      # skip empty first and cue at end
            train_cue = train_x_qs[-1]

            first_shot = train_shots[0]
            new_train_shots = [first_shot] * stutter_count

            new_train_x_qs = new_train_shots + [train_cue]
            new_train_x = "Q " + " Q ".join(new_train_x_qs)

            # update the train_info to reflect the new cue/gold lengths
            ti_data = json.loads(train_info)
            ti_len_parts = ti_data["cons_len"].split(".")
            first_part = ti_len_parts[0]
            new_ti_parts = [first_part] * stutter_count + [ti_len_parts[-1]]

            new_ti_data = ".".join(new_ti_parts)
            ti_data["cons_len"] = new_ti_data
            train_info = json.dumps(ti_data)

            example = "{}\t{}\t{}".format(new_train_x, train_y, train_info)
            new_examples.append(example)

            if len(new_examples) >= max_examples:
                break   

        return new_examples

    def test_task(self, args):
        task_started = time.time()

        model_name = args.model
        dataset = args.dataset
        task_name = args.task
        split_name = args.split
        max_examples = args.max_examples
        stutter_count = args.stutter
        split_filter = args.filter
        version = args.version
        xt_run = args.xt_log

        model = model_names[model_name]

        if split_name == "dyn_ood_lexical":
            examples = self.get_ood_lexical_examples(dataset, version, task_name, split_name)

        elif split_name == "dyn_ood_cons_len":
            examples = self.get_ood_cons_len_examples(dataset, version, task_name, split_name, max_examples)

        elif split_name == "dyn_rev_cons_len":
            examples = self.get_rev_cons_len_examples(dataset, version, task_name, split_name, max_examples)

        elif stutter_count > 1:
            examples = self.get_dyn_stutter_examples(dataset, version, task_name, split_name, max_examples, stutter_count)

        elif dataset == "table_lookup":
            self.sys_prompt = "This task concerns a series of transformations on a 3 bit binary string.  The string is passed thru a series of tables, each of which maps the " + \
                "string to a unique output (also a 3 bit binary string).  The tables are named t1-t8.  So, each table has 8 possible inputs, with 8 corresponding outputs.  You can also "+\
                "think of this as 8 functions that memorize 8 input/output mappings.  You will first be given 64 examples of single table transformations of the form: <input> <table> => output.  "+ \
                "These will fully define the mappings for all 8 tables.  This will be followed by over 100 examples of 2 table transformations, of the form: <input> <table1> <table2> => output.  "+\
                "At the end of the prompt, a final example will be presented and your job is to predict the output string.  Just output the final answer, nothing else. " 

            examples = self.get_table_lookup_examples(dataset, version, task_name, split_name)

        else:
            examples = self.get_regular_examples(dataset, version, task_name, split_name, split_filter)

        correct = 0
        total = 0
        if not max_examples:
            max_examples = len(examples)

        self.log_print("\nsystem prompt:\n{}\n".format(self.sys_prompt))

        for e, ex in enumerate(examples):

            if max_examples and e >= max_examples:
                break

            parts = ex.split("\t")
            prompt, y, info = parts
            prompt = prompt.strip()

            y_hat = self.get_completion(prompt, model_id=model)
            if not y_hat:
                y_hat = "<no response>"

            # adjust output for models that include the prompt
            if y_hat.startswith(prompt):
                y_hat = y_hat[len(prompt):].strip()

            is_correct = (y==y_hat)

            print("example #{}, info: {}, model: {}, task: {}, version: {}, split: {}".format(1+e, info, model_name, task_name, version, split_name))
            self.log_print("prompt: {}".format(prompt))
            self.log_print("gold:   {}".format(y))
            self.log_print("y^:     {}".format(y_hat))

            if is_correct:
                correct += 1
                self.log_print("[CORRECT]\n")
            else:
                self.log_print("[WRONG]\n")

            total += 1

            # show status of testing

            if e and e % 20 == 0 and xt_run:
                xt_run.log_metrics({"correct": correct, "example_num": 1+e, "accuracy": correct/e}, step_name="example_num")

            self.log_print("({:,}/{:,} correct, {:.4f})\n".format(correct, e+1, correct/(e+1)))

        msg = "TOTAL: correct={}/{} (accuracy: {:.4f})".format(correct, total, correct/total)
        self.log_print(msg)

        task_elapsed = time.time() - task_started
        elapsed_text = get_time_str(task_elapsed)
    
        self.log_print("task time: {} ({:.2f} secs/example)".format(elapsed_text, task_elapsed/max_examples)) 

        return msg, correct, total

    def test(self, args):
        import datetime
        import socket

        model_name = args.model
        dataset = args.dataset
        task_name = args.task
        split_name = args.split
        xt_log = args.xt_log

        # print date/time and machine name
        print("time: \t{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("host: \t{}".format(socket.gethostname()))
        print("model: \t{}".format(model_name))

        # print task, split
        print("dataset: \t{}".format(dataset))
        print("task: \t{}".format(task_name))
        print("split: \t{}".format(split_name))
        print()        

        xt_run = None

        if xt_log:
            from xtlib.run import Run
            xt_run = Run()

            # log hyperparameters
            safe_hp_dict = dict(args.__dict__)
            safe_hp_dict["model"] = model_name
            xt_run.log_hparams(safe_hp_dict)

        msg, correct, total = self.test_task(args)

        if xt_run:
            xt_run.log_metrics({"correct": correct, "example_num": total, "accuracy": correct/total}, step_name="example_num")
            xt_run.close()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Test a pretrained LLM using the nc_tgt dataset.")
        parser.add_argument("--model", type=str, help="name of the model to be tested", default="gpt-4")
        parser.add_argument("--dataset", type=str, help="name and version of dataset to test", default=None)
        parser.add_argument("--task", type=str, help="name of the dataset task to test", default=None)
        parser.add_argument("--split", type=str, help="name of the task split to test", default=None)
        parser.add_argument("--filter", type=str, help="string to match for inculded examples", default=None)
        parser.add_argument("--max_examples", type=int, help="max examples to test", default=100)
        parser.add_argument("--stutter", type=int, help="number of times to repeat first example", default=1)
        parser.add_argument("--grammar", type=int, help="include TGT grammar in system prompt", default=0)
        parser.add_argument("--xt_log", type=int, help="set =1 to log hparams and metrics to XT", default=0)

        # # legacy args
        # parser.add_argument("--tgt_task", type=str, help="name of the tgt task to test", default="3_shot_rlw")
        # parser.add_argument("--tgt_split", type=str, help="tgt dataset split name", default="ood_cons_len")
        # parser.add_argument("--tgt_version", type=str, help="version of tgt dataset", default="v11")
        #parser.add_argument("--models", type=str, help="list of models to test (or use 'best' to test predefined set)", default=0)

        args = parser.parse_args()

        # apply dataset/task/split to args
        if not args.dataset:
            args.dataset = "nc_tgt/" + args.tgt_version
            args.task = args.tgt_task
            args.split = args.tgt_split

        if not args.task:
            raise Exception("task name is required")

        if not args.split:
            raise Exception("split name is required")

        # split dataset into dataset and version
        if "/" in args.dataset:
            dataset, version = args.dataset.split("/")
            args.dataset = dataset
            args.version = version
        else:
            args.version = None

        return args

    def usage(self):
        print("usage: python llm_tester.py [ --model <model_name> ] [ --tgt_task <name> ] [ --tgt_split <name> ] [ --max_examples <number> ] ")
        print("examples: ")
        print("  > python llm_tester.py                      (tests gpt-4o model)")
        print("  > python llm_tester.py  help                (print this help message)")
        print("  > python llm_tester.py  --model sonnet-3.5  (tests sonnet 3.5 model")
        print("  > python llm_tester.py  --tgt_task 2_shot   (test default gpt-4 model with 2_shot task")
        print("  > python llm_tester.py  --max_examples 1000 (test default gpt-4 model with 1000 examples")
        print()

        print("available models: {}".format(", ".join(model_names.keys())))

if __name__ == "__main__":  
    tester = LlmTester()

    args = tester.parse_args()

    if args.model not in model_names.keys():
        print("unknown model: {}".format(args.model))
        tester.usage()
        sys.exit(1)

    if args.task.endswith("_grammar"):
        args.grammar = 1
        args.tgt_task = args.task.replace("_grammar", "")

    tester.set_use_grammar(args.grammar)
    
    tester.test(args)

