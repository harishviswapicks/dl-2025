# Beyond Keywords: An Attention-Based Architecture for Semantic and Credibility-Aware Ranking

**Authors:** Ria Patel, Harish Viswanathan, Rhea Garg, Girish Jayakumar

## Project Summary

Large language models (LLMs) are typically known to surface and reference sources in response to user queries. However, the method to which these sources are ranked remain largely unknown and tend to vary based on the model, leading to a new type of search optimization compared to traditional Search Engine Optimization (SEO): Generative Engine Optimization (GEO). While traditional SEO relies heavily on keywords, LLMs rely on deep semantic embeddings and contextualized information to determine source relevancy. However, this creates a new challenge where sources that are semantically similar to the user query but not credible may be ranked higher, while authoritative content without SEO optimization may be underweighted and consequently ranked lower. Understanding how to bridge this gap between semantic similarity and credibility is the primary motivation.

**Hypothesis:** An attention-based ranking model that jointly integrates semantic embeddings from LLMs with credibility-based factors (domain authority, publication recency, etc.) will outperform models that rely on either alone, such as traditional SEO and GEO.

## Approach

Our approach combines two key components:

1. **Semantic Branch**: Encodes query-document embeddings through a pretrained transformer encoder (e.g., SBERT)
2. **Credibility Branch**: Processes metadata through a multilayer perceptron (domain authority, keyword density, publication recency, etc.)

These representations are combined through a **gated attention mechanism** to enable the model to adaptively weigh semantic and credibility features when producing a ranking score.

## Datasets

- **MS MARCO**: Large-scale dataset with millions of query-document pairs for training
- **TREC-COVID**: Benchmark dataset focused on scenarios where credibility and recency are important
- **CommonCrawl**: Source for metadata including publication timestamps, keyword density, and other features

## Technical Stack

- **Framework**: PyTorch with GPU acceleration (PACE clusters or AWS)
- **Models**: Sentence-BERT (SBERT), Mini-LM from Hugging Face Transformers
- **Loss Functions**: RankNet (pairwise), LambdaRank (listwise)
- **Baselines**: BM25, semantic-only transformer models

## Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain (primary metric)
- **Precision@k**: Fraction of top-k documents that are relevant
- **MAP**: Mean Average Precision across all ranks

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook ranking_model.ipynb
```

## Project Structure

```
dl-2025/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── ranking_model.ipynb      # Main implementation notebook
```

## Expected Outcomes

We hypothesize that the credibility-aware attention fusion model will output rankings that better align with LLM-cited sources than keyword-only or semantic-only methods, demonstrating improved performance on NDCG@10, Precision@k, and MAP metrics.
