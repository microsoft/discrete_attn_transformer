# Repo: discrete_attn_transformer
Implementation of Discrete Attn Transformer.  Includes from scratch training of transformer for
Templatic Generation Tasks (dataset available at: https://huggingface.co/datasets/rfernand/templatic_generation_tasks), 
along with code to evaluate LLMs on same dataset.

See our paper: "Mechanisms of Symbol Processing for In-Context Learning in Transformer Networks" (https://arxiv.org/pdf/2410.17498)

# Available tools/apps:

    - psl_compiler.py
        compiles a PSL program (production system language) into JSON weights (output to json_weights directory; under 1 sec, typically)

    - weights_compiler.py
        compiles JSON weights to DAT Transformer matrices (output to the dat_weights directory as a *.pt file, under 1 sec, typically)

    - dat_interpreter.py: 
        interprets a DAT Transformer's processing of a TGT prompt to predict its completion (under 1 sec, typically)

    - dat_transformer.py
        - the DAT transformer model (each example runs under 5 secs, typically) - can we do this??

    - dat_explorer.py
        - the GUI program to view DatTransformer activations after running a tpp program with a input prompt
        
    - llm_tester.py
        test one of available pre-trained LLM models on one of the TGT dataset tasks

# Examples:
    - to run the DAT transformer on the john_loves_mary task:
        > python dat_transformer.py --example=john_loves_mary

    - to run the DAT interpreter on all examples:
        > python dat_interpreter.py --example=all

    - to run the DAT explorer app:
        > python dat_explorer.py 

    - to test the GPT-4o LLM on the 1_shot_rlw task of the TGT dataset:
        - ensure you have the environment variable "OPENAI_API_KEY" defined to your API key
        > python llm_tester.py --model=gpt-4o --dataset=nc_tgt/v11 --task=1_shot_rlw --split=test --max_examples=100

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.