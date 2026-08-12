# Topic modeling results

This directory contains outputs from two MALLET-based pipelines:

- `mallet_cli/` contains files produced by running MALLET directly from the command line on macOS.
- `mallet_gensim/` contains files produced in Google Colab through Gensim's `LdaMallet` wrapper. The final saved model has eight topics and 1,000 iterations.
- `models/` contains saved model files. Files with the same prefix belong to one model and must be kept together.
- `interpretation/` contains the formatted manual interpretation of topic outputs.
