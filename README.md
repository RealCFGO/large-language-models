# LLM Tokenization Visualizer

This repository contains a small, focused Streamlit application designed to examine how Large Language Models tokenize text and perform next-token prediction. The project is intended as a technical and educational prototype rather than a full production system, with emphasis on transparency and inspectability of core mechanisms.

## Scope and Design Choices

The application uses a **fixed model**, GPT-2 (small), to ensure reproducibility and to avoid confounding effects from comparing different model architectures. GPT-2 is used exclusively for next-token prediction, as it is a causal language model and therefore suitable for sequential generation.

Two tokenization strategies are supported for comparison:

- **Byte Pair Encoding (BPE)** via `GPT2TokenizerFast`
- **WordPiece** via `BertTokenizerFast`

WordPiece is used strictly for analytical and visual comparison. All probability calculations and predictions are performed using GPT-2’s BPE tokenizer to maintain model compatibility.

## Core Functionality

The application exposes two tightly coupled components:

### Tokenization Inspection

Input text is tokenized and rendered directly in the interface. Each token is displayed inline with visual differentiation based on its role, including word starts, subword continuations, punctuation, and special tokens. Token indices and token IDs are available in tabular form, along with a mapping that shows how many tokens each word is split into. This makes differences in tokenization granularity immediately observable, particularly for longer or morphologically complex words.

### Next-Token Prediction

For a given input sequence, the model computes logits for the next token position. These logits are transformed into probabilities using softmax, and the top-N most likely tokens are displayed. The distribution is shown both numerically and graphically, making it clear that generation is probabilistic rather than deterministic.

Two sampling-related parameters are exposed:

- **Temperature**, which scales the logits and controls how concentrated or diffuse the probability distribution is.
- **Repetition penalty**, which reduces the likelihood of tokens that already appear in the input.

Adjusting these parameters updates the predictions in real time, allowing direct inspection of their effect on model behavior.

## Technical Structure

The codebase is structured around clear separation of concerns:

- Model and tokenizer loading is isolated and cached to avoid redundant computation.
- Tokenization logic, token classification, and visualization are encapsulated in dedicated functions.
- Probability computation and sampling logic are kept separate from the UI layer.
- Visualization is handled via Plotly, while Streamlit is used solely as a lightweight interface layer.

## Intended Use

The project is intended for analytical and educational purposes, particularly in contexts where understanding parsing, subword tokenization, and next-token prediction is more important than raw generation quality. It is suitable for demonstrations, coursework, and exploratory analysis of LLM behavior at the token level.
