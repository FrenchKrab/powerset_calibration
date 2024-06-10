"""
This package goes with the paper "On the calibration of powerset speaker diarization models" published at Interspeech 2024.
More details about the paper (pdf, citation, supplementary material) can be found at https://github.com/FrenchKrab/IS2024-powerset-calibration

Unlike the name suggests, this package contains mostly code to select regions of interest in the data using the model output probabilities.

Please take a look at the notebooks available in the repository to get started.

The two most important part are first obtaining the model probabilities file. Relevant functions are in the `powerset_calibration.inference` file:
- `powerset_calibration.inference.model_inference_to_file`

And second obtaining the regions of interest using the model probabilities. Relevant functions are in the `powerset_calibration.protocol` file:
- `powerset_calibration.protocol.ActiveLearningProtocol`
- `powerset_calibration.protocol.ActiveLearningProtocolSettings`

"""

__version__ = "0.1.2"
