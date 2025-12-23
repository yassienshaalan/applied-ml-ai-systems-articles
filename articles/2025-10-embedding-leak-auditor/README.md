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


---

## What This Is About

Embeddings are not neutral.  
They encode meaning — and in doing so, they encode information about the data used
to train them.

This raises concrete questions that are often discussed abstractly:

- Can an attacker infer whether a specific text was part of the training data?
- Can embeddings be inverted back into readable or semantically equivalent text?
- How much noise or perturbation is required to meaningfully reduce leakage?
- At what point does privacy protection destroy downstream utility?

ELA turns these questions into **measurable experiments**, producing actionable
signals rather than theoretical arguments.

---

## 🔗 Article Link

Medium article:  
**Understanding Is the Work: Why Machine Learning Cannot Be Reduced to Optimisation**  
 *link to be added once published*
https://medium.com/@yassien/understanding-is-the-work-why-machine-learning-cannot-be-reduced-to-optimisation-a12d9b78cef2
---

##  What This Article Is About

As ML tooling becomes more powerful, it is increasingly tempting to treat modelling
as a procedural task:
- choose a model
- optimise a metric
- deploy the result

This article explores why that framing breaks down in practice.

In particular, it focuses on:
- The gap between strong metrics and genuine understanding
- How Goodhart’s Law manifests in ML systems
- Why the most dangerous failures come from *high-performing* models
- The distinction between procedural optimisation and interpretive judgement
- What changes when LLMs are introduced into the modelling loop

The goal is not to argue against automation or LLMs, but to clarify **where optimisation
ends and responsibility begins**.

---

## Contents

