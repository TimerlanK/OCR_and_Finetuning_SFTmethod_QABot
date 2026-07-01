# OCR and Fine-tuning QA Bot

Fine-tuning a language model to answer questions about the history of Kazakhstan
(11th-grade school textbook), in Russian.

## Overview

The project implements a full pipeline: OCR of the textbook, generation of a
question–answer training set, QLoRA fine-tuning, and quality evaluation.

**Pipeline:**

1. **OCR** — recognize pages 3–23 of `book.pdf` ("История Казахстана, 11 класс")
   with the **Qwen2.5-VL-7B** vision model (4-bit). The model reads each page with
   context, so Roman numerals (centuries), dates, and Kazakh/English words are
   recognized correctly → `history_text.txt`.
2. **Dataset generation** — from the recognized text, a set of question–answer
   pairs is built (283 pairs, generated with `gpt-5-mini`) →
   `history_sft_dataset.json`.
3. **Fine-tuning (SFT / QLoRA)** — `Meta-Llama-3.1-8B-Instruct` (4-bit) is
   fine-tuned with Unsloth and the TRL `SFTTrainer`.
4. **Evaluation** — ROUGE / BLEU / BERTScore metrics on a held-out split and a
   comparison of the base and fine-tuned models.

## Files

| File | Description |
| --- | --- |
| `history_finetuning.ipynb` | Notebook with all cell outputs (OCR → dataset → training → evaluation). |
| `history_text.txt` | Textbook text after OCR. |
| `history_sft_dataset.json` | Question–answer dataset. |
| `evaluation_answers.csv` | Held-out answers: base vs. fine-tuned. |
| `book.pdf` | Source textbook scan ("История Казахстана, 11 класс"). |
| `requirements.txt` | Dependencies. |

## Run (Google Colab, GPU T4)

1. Open `history_finetuning.ipynb` in Google Colab.
2. Upload `book.pdf` (included in this repo) to the Colab session.
3. Set secrets via Colab Secrets or a local `.env`: `OPENAI_API_KEY`, `HF_TOKEN`
   (optionally `HF_USERNAME` to enable the Hub push).
4. **Runtime → Run all**.
5. Save the notebook together with its cell outputs.

## Publishing

After training, the artifacts are published to the Hugging Face Hub (public):
- Model (LoRA adapter): `timkaiyr/kz-history-qa-llama3.1-8b`
- Dataset + OCR text (`history_text.txt`): `timkaiyr/kz-history-qa-dataset`

The loss curve is plotted in the notebook with matplotlib (`loss_curve.png`) and
shows two lines — training loss and eval loss — on the same axes.

## Secrets

`OPENAI_API_KEY` and `HF_TOKEN` are stored in a local `.env` (excluded from git)
or in Colab Secrets. Do not add them to version control.

## Stack

Python · Unsloth · TRL · Transformers · PyTorch · Hugging Face Datasets ·
Qwen2.5-VL (OCR) · OpenAI API · `evaluate` (ROUGE / BLEU / BERTScore) · matplotlib.
