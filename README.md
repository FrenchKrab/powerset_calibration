# powerset_calibration

This package contains the code used to perform the experiments conducted in the paper '*On the calibration of powerset speaker diarization models*' published at Interspeech 2024.

- [ 📄 Browse the original paper repository ](https://github.com/FrenchKrab/IS2024-powerset-calibration?tab=readme-ov-file)
- [ 📘 Read the docs ](https://frenchkrab.github.io/powerset_calibration/)

This package builds upon the pyannote suite, and thus heavily depends on `pyannote.audio`, `pyannote.core` and `pyannote.database` functionalities.

# Installing

### 🐍 Using pip

```bash
pip install powerset_calibration
```

### 🏠 Using a local installation

```bash
git clone https://github.com/FrenchKrab/powerset_calibration
pip install -e powerset_calibration
```

# Usage

Most of the functionalities of the library are easy to access and just require you to plug in the right parameters.

## ⏯️ Notebook

To learn how to use this library, please refer to the notebooks which should give you 90% of the informations you need.

- Essential features
  - [A1_model_inference](notebooks/A1_model_inference.ipynb): Generate and evaluate an 'inference file' from your segmentation model
  - [A2_active_learning_protocol](notebooks/A2_active_learning_protocol.ipynb): Create subsets from an existing protocol using active learning-like criterions (e.g. select the 10% least confident data)
- Advanced usage
  - [B1_subset_one_file](notebooks/B1_subset_one_file.ipynb): Manually do all the steps to find the regions of interest in one file (instead of relying on `ActiveLearningProtocol`).

## 📘 Documentation

If you want more detail about function/method arguments, please refer to the documentation: https://frenchkrab.github.io/powerset_calibration