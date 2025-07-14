# Decodar as Retriever and Reranker (DaRR)

In a de facto RecSys, two stages, retriver and reranker, require manual engineering effort and heavy cost to build and maintain the system, especially when transfering to different domains or adapting to new features. Hence, we're proposing Decoder as Retriever and Reranker to address these issues. 

## Project structure:
- data: collection of data and prompts for training
- engines: collection of models
- endpoints: connectors fron engines to outside