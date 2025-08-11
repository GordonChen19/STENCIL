<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/cs.CV-Computer%20Vision-4b8bbe.svg" alt="cs.CV"></a>
  <a href="#paper"><img src="https://img.shields.io/badge/Paper-Coming%20soon-informational.svg" alt="Paper"></a>
  <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-Benchmark-success.svg" alt="Evaluation"></a>
  <a href="https://gordonchen19.github.io/STENCIL.github.io/"><img src="https://img.shields.io/badge/Website-Project-ff69b4.svg" alt="Website"></a>
  <a href="https://hits.seeyoufarm.com">
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgordonchen19%2FSTENCIL&count_bg=%23007EC6&title_bg=%23555555&icon=github.svg&icon_color=%23FFFFFF&title=Visitors&edge_flat=false"/>
  </a>
</p>

<h1 align="center">STENCIL: Subject-Driven Generation with Context Guidance</h1>

<p align="center">
  <a href="https://gordonchen19.github.io">Gordon Chen</a><sup>*</sup>,
  <a href="https://ziqihuangg.github.io">Ziqi Huang</a>,
  <a href="https://www.a-star.edu.sg/cfar/about-cfar/our-team/dr-cheston-tan">Cheston Tan</a>,
  <a href="https://liuziwei7.github.io/team.html">Ziwei Liu</a>
</p>

<p align="center">
  <sup>*</sup>lead
</p>

<p align="center">
  IEEE ICIP 2025
</p>
<p align="center">
  Oral Spotlight (36 of 491)
</p>

Recent text-to-image diffusion models can produce impressive visuals from textual prompts, but they struggle to reproduce the same subject consistently across multiple generations or contexts. Existing fine-tuning based methods for subject-driven generation face a trade-off between quality and efficiency. Fine-tuning larger models yield higher-quality images but is computationally expensive, while fine-tuning smaller models is more efficient but compromises image quality. To this end, we present Stencil. Stencil resolves this trade-off by leveraging the superior contextual priors of large models and efficient fine-tuning of small models. Stencil uses a small model for fine-tuning while a large pre-trained model provides contextual guidance during inference, injecting rich priors into the generation process with minimal overhead. Stencil excels at generating high-fidelity, novel renditions of the subject in less than a minute, delivering state-of-the-art performance and setting a new benchmark in subject-driven generation.

> **Note:** This codebase is a work in progress.  
> Core components (prompt generation, reference image captioning) are functional,  
> while full model integration and end-to-end generation are still under development.

## Pipeline Overview

The diagram below shows the STENCIL pipeline, combining a small fine-tuned model for subject fidelity with a large frozen model for rich contextual priors.

![STENCIL Pipeline Diagram](static/Diagram.png)

## Repository Structure

```
STENCIL/
├── main.py
├── vlm/
│   ├── extraction_chain.py
│   ├── data_models.py            # AugmentedPrompt, Image schemas
│   └── prompt_template.py        # augmented_prompt_template, caption_template
├── models/
│   ├── base_model.py
│   └── support_model.py
├── references/                   # your reference images (png/jpg)
├── static/
│   ├── Diagram.pdf
│   └── Diagram.png               # for README embedding
├── .env                          # environment variables (gitignored)
└── requirements.txt
```

## Guide

Install: `pip install -r requirements.txt`

---

## Environment Variables

Create a `.env` at the project root (same folder as `main.py`):

OPENAI_API_KEY="your api key"

> No spaces around `=` and no quotes.

---

## Prepare Reference Images

Place one or more images into:

Put a few images into `STENCIL/references/`

---

## Run

```bash
python main.py