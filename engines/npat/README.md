Reimplementation of [Alighning Large Language Models with Recommendation Knowledge](https://arxiv.org/pdf/2404.00245)


Step 1: Model selection: [Gemma 3 4b-it](https://huggingface.co/google/gemma-3-4b-it). 
**NOTE**: login HuggingFace and request access to the model. 

Step 2: Download weights of the model by "litgpt download google/gemma-3-4b-it". Otherwise, follow this [tutorial](https://lightning.ai/lightning-ai/studios/litgpt-quick-start?section=featured).

Step 3: Follow this [notebook](https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning?section=featured) to fine-tune model.