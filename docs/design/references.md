# 21. References

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

Retrieval-augmented generation, graphs, and topics

- [R1] Edge et al., 2024. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* https://arxiv.org/abs/2404.16130
- [R13] Girvan & Newman, 2002. *Community structure in social and biological networks.* PNAS 99(12). https://doi.org/10.1073/pnas.122653799
- [R14] Traag, Waltman & van Eck, 2019. *From Louvain to Leiden: guaranteeing well-connected communities.* https://arxiv.org/abs/1810.08473
- [R15] Pham et al., 2024. *TopicGPT: A Prompt-based Topic Modeling Framework.* NAACL. https://arxiv.org/abs/2311.01449
- [R12] Liu et al., 2023. *Lost in the Middle: How Language Models Use Long Contexts.* https://arxiv.org/abs/2307.03172

Evaluation of RAG and factuality

- [R2] Es et al., 2023. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* https://arxiv.org/abs/2309.15217
- [R3] Min et al., 2023. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* https://arxiv.org/abs/2305.14251
- [R4] Ru et al., 2024. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation.* https://arxiv.org/abs/2408.08067
- [R16] Wei et al., 2024. *Long-form factuality in large language models (SAFE).* https://arxiv.org/abs/2403.18802
- [R20] Dhuliawala et al., 2023. *Chain-of-Verification Reduces Hallucination in Large Language Models.* https://arxiv.org/abs/2309.11495
- [R18] Roberts, Raffel & Shazeer, 2020. *How Much Knowledge Can You Pack Into the Parameters of a Language Model?* https://arxiv.org/abs/2002.08910

Synthetic question generation and question types

- [R5] Alberti et al., 2019. *Synthetic QA Corpora Generation with Roundtrip Consistency.* ACL. https://arxiv.org/abs/1906.05416
- [R6] Rajpurkar, Jia & Liang, 2018. *Know What You Don't Know: Unanswerable Questions for SQuAD.* https://arxiv.org/abs/1806.03822
- [R7] Yu et al., 2023. *CREPE: Open-Domain Question Answering with False Presuppositions.* ACL. https://arxiv.org/abs/2211.17257
- [R8] Min et al., 2020. *AmbigQA: Answering Ambiguous Open-domain Questions.* https://arxiv.org/abs/2004.10645
- [R8b] Aliannejadi et al., 2019. *Asking Clarifying Questions in Open-Domain Information-Seeking Conversations.* SIGIR. https://arxiv.org/abs/1907.06554
- [R19] Wen et al., 2024. *Know Your Limits: A Survey of Abstention in Large Language Models.* TACL. https://arxiv.org/abs/2407.18418
- [R21] Ribeiro et al., 2020. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* https://arxiv.org/abs/2005.04118
- [R32] Yang et al., 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering* (background for integration and cross-document questions). https://arxiv.org/abs/1809.09600
- Tooling with similar goals: [Ragas test-set generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/), [DeepEval synthesizer](https://deepeval.com/docs/synthetic-data-generation-introduction).

LLM judges

- [R28] Zheng et al., 2023. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* https://arxiv.org/abs/2306.05685
- [R31] Thakur et al., 2024. *Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.* https://arxiv.org/abs/2406.12624

Retrieval and text processing

- [R9] Robertson & Zaragoza, 2009. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR. https://doi.org/10.1561/1500000019
- [R10] Cormack, Clarke & Büttcher, 2009. *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR. https://doi.org/10.1145/1571941.1572114
- [R26] Karpukhin et al., 2020. *Dense Passage Retrieval for Open-Domain Question Answering.* https://arxiv.org/abs/2004.04906
- [R11] Kusupati et al., 2022. *Matryoshka Representation Learning.* https://arxiv.org/abs/2205.13147
- [R27] Broder, 1997. *On the resemblance and containment of documents.* https://doi.org/10.1109/SEQUEN.1997.666900
- [R17] Lee et al., 2022. *Deduplicating Training Data Makes Language Models Better.* https://arxiv.org/abs/2107.06499
- [R24] Tsarfaty et al., 2019. *What's Wrong with Hebrew NLP? And How to Make it Right.* EMNLP demo. https://aclanthology.org/D19-3044/
- [R25] Larkey, Ballesteros & Connell, 2002. *Improving Stemming for Arabic Information Retrieval: Light Stemming and Co-occurrence Analysis.* SIGIR. https://doi.org/10.1145/564376.564425

Security

- [R22] Greshake et al., 2023. *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* https://arxiv.org/abs/2302.12173
- [R23] Hines et al., 2024. *Defending Against Indirect Prompt Injection Attacks With Spotlighting.* https://arxiv.org/abs/2403.14720

Statistics

- [R29] Wilson, 1927. *Probable Inference, the Law of Succession, and Statistical Inference.* JASA 22(158). https://doi.org/10.1080/01621459.1927.10502953
- [R30] Brown, Cai & DasGupta, 2001. *Interval Estimation for a Binomial Proportion.* Statistical Science 16(2). https://doi.org/10.1214/ss/1009213286

Document conversion

- [R33] Unicode Standard Annex #9, *Unicode Bidirectional Algorithm.* https://www.unicode.org/reports/tr9/
- [R34] Tesseract OCR engine. https://github.com/tesseract-ocr/tesseract
- [R35] Docling. https://github.com/docling-project/docling
- pdfplumber. https://github.com/jsvine/pdfplumber
- Gemini document understanding: https://ai.google.dev/gemini-api/docs/document-processing

Platform

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google GenAI Python SDK: https://github.com/googleapis/python-genai
