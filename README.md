# RAG-JOURNEY

## Overview Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique designed to enhance the performance of Large Language Models (LLMs) by providing them with additional information from external knowledge sources. This approach allows for the generation of more accurate and contextually relevant responses while minimizing the occurrence of hallucinations.

## Problem Statement

Traditionally, adapting neural networks to specific domains or proprietary information involves fine-tuning the model. While effective, this method is often:
- Compute-intensive
- Expensive
- Requires technical expertise 

This makes it less agile in adapting to evolving information.

## Solution

In 2020, Lewis et al. proposed RAG in their paper *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. This approach combines a generative model with a retriever module, allowing for:
- Easier updates to external knowledge sources
- Flexibility in handling diverse information

## RAG Pipeline
![image](https://github.com/user-attachments/assets/8c7fd4fc-4d7e-4801-a883-8b07d5a48470)

The RAG pipeline consists of three main steps:

1. **Retrieve**: 
   - The user query is used to fetch relevant context from an external knowledge source.
   - The query is embedded into a vector space that corresponds to the context in a vector database, enabling a similarity search to return the top k closest data objects.

2. **Augment**: 
   - The retrieved context is combined with the user query using a prompt template.

3. **Generate**: 
   - The augmented prompt is fed into the LLM to produce the final response.

### Example

Here’s a visualization of the RAG pipeline applied to a specific example:
![image](https://github.com/user-attachments/assets/e1651b98-fcc1-440a-badc-0e49f86a4ac2)

1. **User Query**: "What did the president say about Justice Breyer?"
2. **Retrieve**: The context is retrieved from the vector database.
   - Example Context: "The president thanked Justice Breyer for his service and acknowledged his dedication to serving the country."
3. **Augment**: Combine the query and context into a prompt.
4. **Generate**: The LLM processes this prompt to generate a comprehensive response.
