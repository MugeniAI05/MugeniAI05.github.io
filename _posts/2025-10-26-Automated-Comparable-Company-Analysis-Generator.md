---
layout: post
title: Automated Comparable Company Analysis Generator
image: "/posts/comps-title-img.png"
tags: [Comparable Companies, Equity Research, NLP, Embeddings, Python]
---

Building a comps list is one of those “simple in theory, painful in practice” analyst tasks: you need peers that *actually* match the business model, are *actively traded*, and are *comparable in scale* (or at least clearly flagged when they aren’t). This project automates that funnel for **Ralph Lauren Corporation** using an LLM + deterministic validation + embedding-based similarity scoring.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Notebook Overview](#data-overview)
- [02. Methodology Overview](#modelling-overview)
- [03. Broad Screen (LLM Reasoning)](#linreg-title)
- [04. Verification (Deterministic Validation)](#regtree-title)
- [05. Semantic Validation (Embeddings + Similarity)](#rf-title)
- [06. End-to-End Workflow (Pipeline Orchestration)](#modelling-summary)
- [07. Results & Findings](#modelling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

In a traditional comps workflow, an analyst must filter thousands of public companies to find peers that match:

1. **Business model similarity** (e.g., brand-led “apparel & luxury” vs. mass retail)
2. **Comparable scale** (often proxied by market cap / revenue)
3. **Operational validity** (public, active, and actually traded)

This notebook implements an “AI Analyst” pipeline that mimics that decision funnel and produces a **validated comps list** for **Ralph Lauren Corporation**.

<br>
<br>

### Actions <a name="overview-actions"></a>

The notebook is structured as a cognitive pipeline with two layers:

**1) Reasoning Layer (LLM):**
- Uses an LLM to brainstorm a broad list of publicly traded candidates based on the target company’s business description.

**2) Validation Layer (Deterministic + Mathematical):**
- **Financial check:** Uses `yfinance` to confirm tickers are valid and actively traded, and to pull market cap / industry metadata.
- **Semantic check:** Uses OpenAI embeddings to score similarity between the target’s business description and each candidate’s business summary.
- Applies a **similarity threshold of 0.30** to remove structurally unrelated firms.

<br>
<br>

### Results <a name="overview-results"></a>

In the example run shown in the notebook output:

- The pipeline selected **10 publicly traded comparables** after semantic and financial validation.
- Similarity scoring behaved sensibly:
  - **Ralph Lauren vs. itself** scored **0.88** (a sanity check that the embedding + similarity logic is working).
  - Core peers clustered roughly in the **0.45–0.55** similarity range, consistent with “same space, different positioning.”
- Top strategic matches in that run included **PVH**, **Burberry**, **Capri**, **V.F.**, and **Columbia Sportswear**, each passing the similarity threshold.

<br>
<br>

### Growth/Next Steps <a name="overview-growth"></a>

A few high-impact upgrades (without changing the spirit of the notebook):

- Add a second scale metric beyond market cap (e.g., revenue bands) to separate “operating comps” from “aspirational comps.”
- Replace free metadata proxies (Yahoo Finance industry/sector) with richer classifications (e.g., CapIQ/Bloomberg SIC/GICS) in production.
- Calibrate the similarity threshold using labeled historical comps (human-curated “true comps” sets).
- Add an analyst review UI: accept/reject comps and store decisions for future retrieval (human-in-the-loop, but scalable).

<br>
<br>

### Key Definition  <a name="overview-definition"></a>

A **comparable company** (comp) is a publicly traded firm used as a benchmark for valuation or performance because it shares meaningful economic characteristics with the target — typically business model, customer, product mix, and operating context.  

In practice, comps are often split into:
- **Strategic peers** (compete in a similar market space)
- **Financial peers** (similar in size/valuation profile)

___

# Notebook Overview  <a name="data-overview"></a>

This project is implemented in a single notebook: **Automated Comparable Company Analysis Generator**.

Key components used:

- `openai` for LLM reasoning and embeddings  
- `yfinance` for market data + company metadata (free, compliance-friendly)  
- `tenacity` for retry logic (API resilience)  
- `pandas` for table assembly and CSV export  

The target company is defined as a structured object (name, URL, business description, industry), and the pipeline produces a ranked dataframe of validated comps.

```python
# Target defined in-notebook
ralph_lauren_data = {
    "name": "Ralph Lauren Corporation",
    "url": "https://corporate.ralphlauren.com/",
    "business_description": (
        "Ralph Lauren Corporation designs, markets, and distributes premium lifestyle products, "
        "including apparel, accessories, footwear, home furnishings, and fragrances. "
        "The company operates through a combination of wholesale, retail, and digital channels "
        "and sells products globally under the Ralph Lauren brand."
    ),
    "primary_industry_classification": "Apparel & Luxury Goods"
}

<br>

## Methodology Overview <a name="modelling-overview"></a>

The notebook mirrors how a human analyst would do comps screening:

1. **Broad screen:** generate candidate tickers from a semantic understanding of the target business.
2. **Verify:** confirm candidates are valid public tickers and actively traded (and pull market cap).
3. **Validate:** compute semantic similarity using embeddings and discard weak matches below a threshold.
4. **Rank + export:** return a comps table sorted by similarity score and write to CSV.

---

## Broad Screen (LLM Reasoning) <a name="linreg-title"></a>

This step is intentionally *broad*. The notebook prompts the LLM as if it were a senior investment analyst and requests **15–20 publicly traded tickers** that match brand positioning, product categories, and customer demographics — returning **only a JSON list of ticker symbols**.
```
prompt = f"""
Target Company: {target_company['name']}
URL: {target_company['url']}
Description: {target_company['business_description']}
Industry: {target_company['primary_industry_classification']}

Please identify 15–20 PUBLICLY TRADED companies that are strong comparables.
Focus on companies with similar brand positioning, product categories, and customer demographics.

Return ONLY a JSON list of ticker symbols.
"""
response_text = agent.reason(prompt)
candidates = json.loads(clean_text)

```
