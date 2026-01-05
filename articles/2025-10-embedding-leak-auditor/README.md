# Embedding Leak Auditor (ELA)

This folder contains the code and Colab notebooks for **Embedding Leak Auditor (ELA)**,
a framework for empirically measuring **privacy leakage in embedding representations**.

Modern AI systems rely heavily on embeddings to represent text, power semantic search,
enable clustering, and support retrieval-augmented generation (RAG).  
While embeddings are often treated as anonymised or privacy-safe abstractions, they
retain **latent memory of their training data**.

ELA provides a structured way to **measure, visualise, and reason about that memory**.

---

## 🔗 Article Link

Medium article:  
**Embedding Leak Auditor: Measuring Privacy Leakage in Vector Representations**  
[Embeddings, Privacy, and the Leak Auditor: Auditing the Hidden Memory of AI](https://medium.com/@yassien/understanding-is-the-work-why-machine-learning-cannot-be-reduced-to-optimisation-a12d9b78cef2](https://medium.com/@yassien/embeddings-privacy-and-the-leak-auditor-auditing-the-hidden-memory-of-ai-2e7c78339ad9))

---

## What This Is About

Embeddings are not neutral.  
They encode meaning  and in doing so, they encode information about the data used
to train them.

This raises concrete questions that are often discussed abstractly:

- Can an attacker infer whether a specific text was part of the training data?
- Can embeddings be inverted back into readable or semantically equivalent text?
- How much noise or perturbation is required to meaningfully reduce leakage?
- At what point does privacy protection destroy downstream utility?

ELA turns these questions into **measurable experiments**, producing actionable
signals rather than theoretical arguments.

---
## The code
The code is [here](https://github.com/yassienshaalan/embedding_leak_auditor)


---

## Contents

