# STENCIL — Subject-Driven Generation with Context Guidance

Recent text-to-image diffusion models can produce impressive visuals from textual prompts, but they struggle to reproduce the same subject consistently across multiple generations or contexts. Existing fine-tuning based methods for subject-driven generation face a trade-off between quality and efficiency. Fine-tuning larger models yield higher-quality images but is computationally expensive, while fine-tuning smaller models is more efficient but compromises image quality. To this end, we present Stencil. Stencil resolves this trade-off by leveraging the superior contextual priors of large models and efficient fine-tuning of small models. Stencil uses a small model for fine-tuning while a large pre-trained model provides contextual guidance during inference, injecting rich priors into the generation process with minimal overhead. Stencil excels at generating high-fidelity, novel renditions of the subject in less than a minute, delivering state-of-the-art performance and setting a new benchmark in subject-driven generation.

## Pipeline Overview

The diagram below shows the STENCIL pipeline, combining a small fine-tuned model for subject fidelity with a large frozen model for rich contextual priors.

![STENCIL Pipeline Diagram](static/Diagram.png)

## Repository Structure

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

## Guide

Install: `pip install -r requirements.txt`

---

## Environment Variables

Create a `.env` at the project root (same folder as `main.py`):

OPENAI_API_KEY= "your api key"

> No spaces around `=` and no quotes.

---

## Prepare Reference Images

Place one or more images into:

Put a few images into `STENCIL/references/`

---

## Run

From the `STENCIL/` folder:

```bash
python main.py