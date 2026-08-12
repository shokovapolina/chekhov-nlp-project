# Computational Analysis of Anton Chekhov's Collected Works

Corpus linguistics, computational semantics, and machine learning analysis of Anton Chekhov's literary prose.

> Academic and research project in Digital Linguistics, Southern Federal University.
>
> The source corpus was created from Volume 4 of Anton Chekhov's collected works provided by the [Digital Chekhov project](https://chekhov-digital.sfedu.ru/corpus), converted into plain text (.txt) format, and analyzed using Python and specialized natural language processing tools.

## Project overview

This project applies modern Natural Language Processing (NLP) and Digital Humanities methods to a literary corpus of Anton Chekhov's prose.

The research pipeline includes:

- corpus collection, conversion to plain text, and text preprocessing;
- morphological and syntactic analysis comparing multiple POS taggers and parsers;
- named entity recognition (NER) and character network mapping exported to Gephi;
- training a sentiment classification model on Russian tweets and applying it to Chekhov's prose;
- topic modeling with MALLET LDA, parameter tuning, and model evaluation;
- distributional semantics and custom Word2Vec embedding training;
- semantic analysis using external lexical databases (RuWordNet, BabelNet);
- stylometric analysis and exploratory text analytics.

## Main research results

### 1. Frequency analysis and lexical diversity

Analysis of the lemmatized corpus in Voyant Tools showed:

- 98 documents;
- 59,765 words after stop-word removal;
- 11,651 unique lemmas;
- the most frequent lemmas were `говорить` (361), `свой` (357), `глаз` (295), `человек` (276), and `сказать` (238).

Document-level lexical density ranged from 0.528 to 0.949. The highest values were found in short works, including *Vrachebnye sovety*, *Delets*, *Reklama*, and *Zapiska*.

### 2. Contexts, links, and collocations

- `говорить` was strongly associated with `сказать` and `ответить`, which is consistent with the prominent role of dialogue in the corpus;
- `глаз` was associated with `смотреть` and `лицо`, reflecting visual description;
- `человек` was associated with `жизнь` and `рука`, pointing to anthropocentric and bodily vocabulary;
- notable correlations included `говорить`–`глаз`, `человек`–`говорить`, `сказать`–`свой`, and `свой`–`человек`.

### 3. Work-length statistics

According to the Voyant Tools analysis, the longest documents were *Vedma* (2,116 words), *Moi zheny* (1,239 words), *Perepolokh* (1,181 words), *Tryapka* (1,131 words), and *Akterskaya gibel* (1,021 words). The shortest texts included *Vrachebnye sovety* (59 words), *Nadul* (64 words), and *Reklama* (76 words).

### 4. Topic modeling

The MALLET model run through Gensim in Google Colab was trained on the lemmatized corpus. Models with 8–10 topics were compared using `c_v` coherence; the notebook saved an eight-topic model with 1,000 iterations.

Based on representative words, the model produced thematic areas related to:

- communication, thinking, and social relations;
- the body, movement, and perception;
- everyday actions and characters;
- family relationships, marriage, and money;
- sleep, illness, and medical contexts;
- everyday communication and family ties;
- money, purchases, and material relations;
- writing, speech, and movement.

An additional Voyant Tools interpretation examined topics related to life, women, and God; family roles such as `отец`, `муж`, and `жена`; and the social-economic vocabulary of rubles, money, and the head. These interpretations belong to the corresponding topic outputs and should not be conflated with the eight-topic MALLET model.

### 5. Morphology, NER, sentiment, and semantics

- compared UDPipe, spaCy, pymorphy, and Mystem on Russian 19th-century prose;
- extracted named entities, including character names and geographical locations, and exported network data to Gephi;
- trained a sentiment classifier on Russian tweets and applied it to Chekhov's works. This result should be interpreted cautiously because the model is transferred across domains;
- trained Word2Vec models on the lemmatized corpus and examined semantic fields for selected lemmas.

### Interpretation note

The frequency and topic-modeling results describe this corpus and depend on stop-word removal, lemmatization, the selected number of topics, and model parameters. Topic labels are research interpretations of representative words, not predefined categories.

## Data and corpus statistics

- **Corpus source:** Volume 4 of Anton Chekhov's collected works from the [Digital Chekhov platform](https://chekhov-digital.sfedu.ru/corpus).
- **Composition:** 98 literary works in 98 plain-text (.txt) files.

### Summary statistics

| Metric | Original corpus (`raw_corpus`) | Lemmatized corpus (`lemmatized_corpus`) |
|---|---:|---:|
| **Number of works (files)** | 98 | 98 |
| **Total character count** | 633,099 | 576,500 |
| **Estimated sentence count** | 12,817 | — |
| **Total token / lemma count** | 95,695 | 94,497 |
| **Unique vocabulary (wordforms / lemmas)** | 22,814 | 11,790 |
| **Type-Token Ratio (TTR)** | 0.2384 (23.8%) | 0.1248 (12.5%) |
| **Average length per work** | ~977 tokens | ~964 lemmas |
| **Median length per work** | 1,061 tokens | 1,052 lemmas |
| **Shortest work** | "Vrachebnye sovety" (78 tokens) | "Vrachebnye sovety" (76 lemmas) |
| **Longest work** | "Ved'ma" (3,378 tokens) | "Ved'ma" (3,362 lemmas) |

## Corpus versions

The project uses two versions of the corpus:

- `chekhov_original/` — 98 original literary texts in plain-text format. This version preserves word forms, punctuation, capitalization, sentence boundaries, and named entities (used for morphological parsing, NER + Gephi network extraction, stylometry, and sentiment analysis).
- `chekhov_lemmatized/` — 98 texts after preprocessing and lemmatization without punctuation. This version combines grammatical word forms (used for frequency analysis, LDA in Gensim/MALLET, Word2Vec, TF-IDF, and clustering).

## Methods and tools

| Analysis stage | Tools and libraries | Purpose |
|---|---|---|
| Data preparation | Python, regex, os, io | Conversion to TXT, cleaning, text normalization |
| Morphology and syntax | UDPipe, spaCy, pymorphy (pymorphy2/pymorphy3), Mystem | Comparative POS tagging, lemmatization, and parsing |
| Entity extraction & graphs | spaCy, Natasha, Gephi | NER (names, locations) and lexical-semantic network visualization |
| Topic modeling | Gensim LDA, MALLET LDA | Latent topic modeling, hyperparameter tuning, and coherence evaluation |
| Distributional semantics | Gensim (Word2Vec) | Word embedding training and semantic neighborhood analysis |
| Semantic analysis | RuWordNet, BabelNet | Measuring semantic distances, hypernym relations, and lexical networks |
| Clustering and sentiment | scikit-learn, sentence-transformers, Naive Bayes | Training a sentiment classifier on tweets, applying it to Chekhov's prose, TF-IDF, Sentence-BERT, K-Means |
| Stylometry and visualization | AntConc, Voyant Tools, matplotlib, seaborn | Frequency lists, N-grams, stylometric charts, and context exploration |

## Project notebooks

Notebook workflow in Google Colab / Jupyter:

1. `preprocessing+lemmatization.ipynb` — raw corpus cleaning and lemmatized corpus generation;
2. `descriptive statistics.ipynb` — computing statistical metrics for both original and lemmatized corpora;
3. `morphological analysis.ipynb` — comparing performance across UDPipe, spaCy, pymorphy, and Mystem;
4. `NER + Gephi code.ipynb` — named entity extraction and network graph export for Gephi;
5. `model training on russian tweets.ipynb` — training a sentiment classifier on a Russian tweet dataset;
6. `sentiment analysis of whole corpus.ipynb` — applying the trained sentiment model to all 98 works in Volume 4;
7. `6_2_1_Topic_modelling_with_malletmine.ipynb` — topic modeling using MALLET LDA;
8. `7_W2V_training_mine.ipynb` — training Word2Vec models and analyzing semantic fields.

## Repository structure

```text
chekhov-nlp-project/
├── README.md
├── README.ru.md
├── requirements.txt
├── data/
│   ├── chekhov_original/           # 98 raw text files (.txt)
│   ├── chekhov_lemmatized/         # 98 lemmatized text files (.txt)
│   └── chekhov_corpus_catalog.csv  # Detailed statistics catalog
├── notebooks/
│   ├── 1_preprocessing+lemmatization.ipynb
│   ├── 2_descriptive_statistics.ipynb
│   ├── 3_morphological_analysis.ipynb
│   ├── 4_NER_+_Gephi_code.ipynb
│   ├── 5_model_training_on_russian_tweets.ipynb
│   ├── 6_sentiment_analysis_of_whole_corpus.ipynb
│   ├── 7_Topic_modelling_with_malletmine.ipynb
│   └── 8_W2V_training_mine.ipynb
└── results/
    ├── corpus_statistics/
    ├── frequency_analysis/
    ├── morphological_analysis/
    ├── ner_and_gephi/
    ├── sentiment_analysis/
    ├── topic_modeling/
    │   ├── mallet_cli/
    │   ├── mallet_gensim/
    │   └── interpretation/
    ├── word2vec/
    └── voyant_tools/
        └── volume_4_analysis.docx
```

## Getting started

1. Clone the repository:

```bash
git clone https://github.com/shokovapolina/chekhov-nlp-project.git
cd chekhov-nlp-project
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

3. Launch Jupyter Notebook or run in Google Colab:

```bash
jupyter notebook
```

Note: some components (such as MALLET LDA or specific spaCy/UDPipe language models) require a Java Runtime Environment (JRE) and downloaded model weights.

## Voyant Tools report

A detailed textual report of the Voyant Tools analysis of the lemmatized corpus is available at [`results/voyant_tools/volume_4_analysis.docx`](results/voyant_tools/volume_4_analysis.docx).

## References and attribution

- Corpus source: [Digital Chekhov Corpus (SFedU)](https://chekhov-digital.sfedu.ru/corpus)
- Conducted as part of the Digital Linguistics program at Southern Federal University.

## Author

**Polina Shokova**  
Digital Linguistics, Southern Federal University  
Email: shokovaapolina@gmail.com  
GitHub: https://github.com/shokovapolina  
LinkedIn: https://www.linkedin.com/in/polina-shokova-71026a389/
