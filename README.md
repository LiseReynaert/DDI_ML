Can the performance of the LLM + GCN framework on OGB-ddi be improved by
(1) substituting biomedical-specific LLM embeddings (PubMedBERT, ChemBERTa)
for general-purpose ones, and (2) replacing random negative sampling with clinically
verified non-interacting pairs from CRESCENDDI — and if so, by how much compared
to the state-of-the-art result of AUC = 0.900 reported by LLM-DDI [1 ]? Additionally,
how do these configurations compare against classical classifiers (logistic regression,
random forest, MLP)?

[1] X. Li et al., “LLM-DDI: Leveraging Large Language Models for Drug-Drug Interaction
Prediction on Biomedical Knowledge Graph,” IEEE Journal of Biomedical and Health
Informatics, Vol. 30, No. 1, pp. 778–789, 2026.
