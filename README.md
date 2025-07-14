<p align="center">
  <img src="https://github.com/giosullutrone/COVER/blob/main/resources/COVER_Title.png" style="width:60%;min-width:240px;display:block;margin:auto;">
</p>

<p align="center">
  📑 <a href="<!-- PAPER_LINK -->">Paper (coming soon)</a> &nbsp;|&nbsp;
  🏷️ <a href="LICENSE">MIT&nbsp;License</a> &nbsp;|&nbsp;
  📝 <a href="TABLES.md">Full Results</a> &nbsp;|&nbsp;
  🖼️ <a href="FRAMEWORK.md">Framework</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
</p>

> **Context-Driven Over-Refusal Benchmark (COVER)**  
> A test-bed for measuring _if_ and _how often_ Large Language Models refuse harmless requests once additional context is provided.

---

## ✨ Overview

Language Models are increasingly used beyond direct Q&A: retrieval-augmented generation, tool-use chains, and agentic reasoning all feed **external documents into the prompt**.

But this additional context may trigger **over-refusals**, cases where models reject otherwise safe queries because of information embedded in the context.

<p align="center"> <strong> <i> COVER benchmarks this Context-Driven Over-Refusal in real-world conditions </strong> </i> </p>

---

## 🧪 What's Included

**🧠 13 Models Tested**
- *Open-weight models*: Mistral-7B-Instruct-v0.3, Llama-2-7B, Meta-Llama-3.0/3.1/3.2 (various sizes), Phi-4, Qwen2.5-7B, gemma-2-9b-it, DeepSeek-R1-Distill, Llama-3.1-8B-Abliterated
- *Closed-source models*: GPT-4o-mini, Gemini 1.5 Flash

**🧾 8 Tasks**
- Question Answering (QA)  
- QA with Chain-of-Thought (CoT)  
- Summarization  
- Keyword Extraction  
- Metadata Generation  
- Topic Generation  
- Translation  
- Named Entity Recognition (NER)

**⚙️ 3 System Prompts**
- NS: No System Prompt  
- HS: Helpful System  
- ES: Ethical System (most restrictive)  
→ See `Appendix A.2` in the paper for exact wording.

**📚 2 Corpora**
- **Hadith** and **Sefaria** religious document corpora  
- Carefully balanced in sensitivity and document structure

**📈 5 Document Retrieval Settings (k)**
- `k = 1, 3, 5, 10, 20`  
- Samples per `k`:  
  - *Hadith*: 60, 320, 362, 1183, 1556  
  - *Sefaria*: 71, 175, 456, 1361, 2438

---
## 📈 Snapshot Results

<p align="center">
  <img src="https://github.com/giosullutrone/COVER/blob/main/resources/SumTableAVGRefusal.png" alt="Average refusal with one sensitive doc" style="background:white;padding:10px;width:40%;min-width:200px;">
</p>

<p align="center">
  <img src="https://github.com/giosullutrone/COVER/blob/main/resources/SumTableAVGRefusalByNDocs.png"     alt="Average refusal vs k docs" style="background:white;padding:10px;width:40%;min-width:200px;">
</p>

<p align="center">
  🔍  These heat-maps are just <strong>summaries</strong>.
  See the full-size tables in <a href="TABLES.md"><code>Full Results</code></a>.
</p>

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/giosullutrone/COVER.git
cd COVER
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

### 2. Generate Question–Context Pairs

```bash
python -u dataset_to_questioned.py \
  --top_ks 1 3 5 10 20 \
  --unsafe_threshold 0.5 \
  --skip_if_exists \
  --model_folder /path/to/models \
  --output_folder /path/to/collections \
  --unsafe_only \
  --hadith_folder /path/to/hadith_corpus \
  --sefaria_folder /path/to/sefaria_corpus
```

### 3. Probe the Models

```bash
python -u questioned_to_results.py \
  --top_ks 1 3 5 10 20 \
  --skip_if_exists \
  --n_generations 5 \
  --datasets_names hadith sefaria \
  --model_folder /path/to/models \
  --input_folder /path/to/collections \
  --output_folder /path/to/results
```

---

## 🔖 Citation

Paper, DOI and BibTeX will be added once the ACL 2025 anthology page goes live. Stay tuned!

```bibtex
@inproceedings{Sullutrone2025cover,
  title     = {COVER: Context‑Driven Over‑Refusal Verification in LLMs},
  author    = {Giovanni Sullutrone and Riccardo Amerigo Vigliermo and Sonia Bergamaschi and Luca Sala},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  url       = {TBD}
}
```

---

## 🤝 Contributing

We welcome issues and contributions!

If you find bugs, inconsistencies, or unclear behaviors, please open an issue or submit a PR.
Community feedback is crucial to improving COVER.

---

## 📝 License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

*If you use COVER in your research, please cite us and let us know, we would love to hear about your work!*
