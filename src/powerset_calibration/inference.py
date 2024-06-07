"""Contains code to generate 'Inference Files' (aka 'Inference DataFrames') from local segmentation models.
Each inference will generate the inference file (.inf.csv or .inf.parquet) and an associated metadata file (.meta.yaml).
"""

import itertools
import os
import pathlib
import re
from logging import warn
from typing import Iterable, Literal, Optional, TypedDict, Union

import numpy as np
import pandas as pd
import torch
import yaml
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.utils import get_devices
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
)
from pyannote.database import FileFinder, ProtocolFile, registry
import tqdm

METADATA_FORMAT_VERSION = 4
SEP = "@"

FileId = Union[str, pathlib.Path]


class SlidingWindowDict(TypedDict):
    duration: float
    step: float
    start: Optional[float]


class InferenceModelMetadata(TypedDict):
    fps: float
    duration: float
    num_frames: int
    specifications: dict
    frames: SlidingWindowDict
    receptive_field: SlidingWindowDict


class InferenceMetadata(TypedDict):
    """Structure of the data saved in inference metadata files"""

    experiment_name: str
    protocol: str
    subset: str
    activation_disabled: str
    file_format: str
    batch_size: int
    model_path: str
    classes: list[str]
    date: dict
    model: InferenceModelMetadata
    format_version: int
    step_duration: float
    protocol_files: list[str]
    last_inference_shape: list[int]
    model_output_cols: list[str]


def get_inference_fid(experiment_name: str, protocol_name: str, subset_name: str) -> str:
    """Get 'unique' file identifier for an experiment inference"""
    return f"{experiment_name}{SEP}{protocol_name}{SEP}{subset_name}"


def get_inf_metadata_filename(fid: FileId) -> str:
    return f"{fid}.meta.yaml"


def is_inf_metadata(filepath: FileId) -> bool:
    """Returns true if the filepath is a valid inference metadata file (whether it exists or not)"""
    return str(filepath).endswith(".meta.yaml")


def is_inf_file(filepath: FileId) -> bool:
    """Returns true if the filepath is a valid inference file (whether it exists or not)"""
    return re.match(r".*\.inf\.([a-zA-Z]+)", str(filepath)) is not None


def load_inference_file(fid: FileId) -> pd.DataFrame:
    """Load the inference file with the given id (do not include the .inf.csv extension)"""
    path_parquet = f"{fid}.inf.parquet"
    if os.path.isfile(path_parquet):
        return pd.read_parquet(path_parquet)

    path_csv = f"{fid}.inf.csv"
    if os.path.isfile(path_csv):
        return pd.read_csv(path_csv)

    raise FileNotFoundError(f"There's no inference file with id {fid}")


def load_inference_metadata(filepath) -> InferenceMetadata:
    """Load the inference metadata file"""
    if not is_inf_metadata(filepath):
        filepath = get_inf_metadata_filename(filepath)
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def _get_aggregated_posteriors_dataframe(
    output: SlidingWindowFeature, file: ProtocolFile, metadata: InferenceMetadata
):
    reference: Annotation = file["annotation"]
    uem: Timeline = file["annotated"]

    output.labels = metadata["classes"]

    # add relevant classes if we predict confidence as separate output
    if output.data.shape[-1] == 2 * len(output.labels):
        output.labels += [f"{c}_conf" for c in output.labels]
    elif output.data.shape[-1] == len(output.labels) + 1:
        output.labels.append("conf")
    elif output.data.shape[-1] != len(output.labels):
        raise Exception(
            f"ERROR : model has classes {metadata['classes']} ({len(metadata['classes'])}), but output has dims {output.data.shape}"
        )

    # Posterior saving
    extent = Segment(0, uem.extent().end)
    ref_t: np.ndarray = reference.discretize(
        support=extent,
        resolution=output.sliding_window,
        labels=file["classes"],
    ).data  # .crop(uem, mode="strict")

    uem_t: np.ndarray = (
        uem.support()
        .to_annotation()
        .rename_labels(generator=itertools.cycle(["uem"]))
        .discretize(
            support=extent,
            resolution=output.sliding_window,
            labels=["uem"],
        )
        .data
        > 0.5
    )  # .crop(uem, mode="strict")

    df_tmp = pd.DataFrame()

    # TODO: get rid of cutting the final part ?
    out_t: np.ndarray = output.data[: ref_t.shape[0], :]
    # print(f'{out_t.shape=}; {ref_t.shape=}; {uem_t.shape=}')
    ref_t = ref_t[: out_t.shape[0]]
    uem_t = uem_t[: out_t.shape[0]]
    assert out_t.shape[0] == ref_t.shape[0], f"expected {out_t.shape }, got {ref_t.shape}"
    assert out_t.shape[0] == uem_t.shape[0], f"expected {out_t.shape }, got {uem_t.shape}"

    df_out = pd.DataFrame(out_t, columns=output.labels)
    df_tmp = df_tmp.join(df_out, how="outer")

    # add the dataset, URI and priors to the dataframe list
    ref_columns: list[str] = [f"{c}_ref" for c in file["classes"]]
    df_ref = pd.DataFrame(ref_t, columns=ref_columns)
    df_uem = pd.DataFrame(uem_t, columns=["uem"])
    df_tmp = df_tmp.join(df_ref, how="outer")
    df_tmp = df_tmp.join(df_uem, how="outer")

    # add file info to every row of the dataframe
    df_file_info = pd.DataFrame([{"dataset": file["database"], "uri": file["uri"]}])
    df_tmp = df_tmp.join(df_file_info, how="cross")
    return df_tmp


def _get_non_aggregated_posteriors_dataframe(
    output: SlidingWindowFeature, file: ProtocolFile, metadata: InferenceMetadata
):
    model_info: InferenceModelMetadata = metadata["model"]

    reference: Annotation = file["annotation"]
    uem: Timeline = file["annotated"]

    num_chunks, num_frames, num_classes = output.data.shape
    num_classes_ref = len(reference.labels())

    # Posterior saving
    ref_t: np.ndarray = np.ones((num_chunks, num_frames, num_classes_ref)) * -2
    uem_t: np.ndarray = np.zeros((num_chunks, num_frames, 1), dtype=bool)
    output_sw: SlidingWindow = output.sliding_window
    for i in range(num_chunks):
        extent = output_sw[i]
        ref_t[i] = reference.discretize(
            support=extent,
            resolution=model_info["duration"] / model_info["num_frames"],
            duration=model_info["duration"],
            labels=reference.labels(),
        ).data

        uem_t[i] = (
            uem.support()
            .to_annotation()
            .rename_labels(generator=itertools.cycle(["uem"]))
            .discretize(
                support=extent,
                resolution=model_info["duration"] / model_info["num_frames"],
                duration=model_info["duration"],
                labels=["uem"],
            )
            .data
            > 0.5
        )

    data = {
        "uem": uem_t.flatten(),
    }
    data.update({f"out_{i}": output.data[:, :, i].flatten() for i in range(num_classes)})
    data.update({f"ref_{i}": ref_t[:, :, i].flatten() for i in range(num_classes_ref)})
    df = pd.DataFrame(data)
    df_file_info = pd.DataFrame([{"dataset": file["database"], "uri": file["uri"]}])
    df = df.join(df_file_info, how="cross")
    df.astype({f"ref_{i}": "int8" for i in range(num_classes_ref)})
    df.astype({f"out_{i}": "float32" for i in range(num_classes)})
    df.astype({"uem": "int8"})

    # print(
    #     f"{uem_t.flatten().shape=} ;; {output.data[:, :, 0].flatten().shape=} ;; {ref_t[:, :, 0].flatten().shape=} ;; {num_classes=} ;; {len(df)=}"
    # )

    return df


def model_inference_to_file(
    model: Union[str, Model],
    protocol: Union[str, Iterable[ProtocolFile]],
    output_identifier: str,
    outputs_folder: str,
    disable_activation: bool = False,
    batch_size: int = 32,
    step_ratio: Optional[float] = None,
    step_duration: Optional[float] = None,
    file_format: Literal["csv", "parquet"] = "parquet",
    metadata: Optional[dict] = None,
    preprocessors: Optional[dict] = None,
):
    """Do segmentation model inference and save the result to a file.
    (Do NOT contain the embedding extraction + clustering process, only the local EEND inference)

    Parameters
    ----------
    model : Union[str, Model]
        Path to the model or Model instance
    files: Union[str, Iterable[ProtocolFile]]
        Files to use for the inference, or protocol name
    output_identifier : str
        Name of the experiment (=> name of the file)
    outputs_folder : str
        Where to save the inference files
    disable_activation : bool, optional
        Gets rid of the activation layer (to output logits directly), by default False
    batch_size : int, optional
        Batch size for the inference, by default 32
    step_ratio : float, optional
        Set step_duration = step_ratio * model_duration, defaults to 0.25
    step_duration : float, optional,
        Step duration for the inference,
    file_format : Literal["csv", "parquet"] = "parquet",
        File format for the inference file, by default "parquet"
    metadata : dict, optional,
        Additional metadata to save in the inference metadata file, by default None
    preprocessors : Optional[dict], optional
        Preprocessors for the protocol, defaults to a FileFinder if left None, by default None
    """
    # fix/check the parameters
    metadata = metadata.copy() if metadata is not None else {}
    if preprocessors is None:
        preprocessors = {"audio": FileFinder()}

    # load the model and prepare the inference
    model_path = None
    if not isinstance(model, Model):
        model_path = model
        model = Model.from_pretrained(model)

    if disable_activation:
        model.activation = torch.nn.Identity()
    else:
        # special behaviours
        if isinstance(model.activation, torch.nn.LogSoftmax):
            model.activation = torch.nn.Softmax(dim=model.activation.dim)

    # fix/check the parameters that need the model to be initialized
    if step_ratio is not None and step_duration is None:
        if step_duration is not None:
            raise ValueError("step_ratio and step_duration are both set")
        step_duration = step_ratio * model.specifications.duration

    # prepare objects for inference
    (device,) = get_devices(needs=1)
    model = model.to(device)
    inference = Inference(
        model,
        skip_conversion=True,
        batch_size=batch_size,
        step=step_duration,
        device=device,
    )

    # fill the metadata
    metadata["model_path"] = model_path
    metadata["experiment_name"] = output_identifier
    metadata["activation_disabled"] = disable_activation
    metadata["batch_size"] = batch_size
    metadata["step_duration"] = step_duration
    metadata["date"] = {
        "inference_start": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata["file_format"] = file_format
    metadata["classes"] = model.specifications.classes
    num_frames = int(model.num_frames(16000 * model.specifications.duration))
    metadata["model"] = {
        "duration": model.specifications.duration,
        "num_frames": num_frames,
        "fps": num_frames / model.specifications.duration,
        "frames": {
            "duration": model.specifications.duration / num_frames,
            "step": model.specifications.duration / num_frames,
        },
        "receptive_field": {
            "start": model.receptive_field.start,
            "duration": model.receptive_field.duration,
            "step": model.receptive_field.step,
        },
        "specifications": {
            "classes": list(model.specifications.classes),
            "duration": model.specifications.duration,
            "min_duration": model.specifications.min_duration,
            "num_powerset_classes": model.specifications.num_powerset_classes,
            "permutation_invariant": model.specifications.permutation_invariant,
            "powerset": model.specifications.powerset,
            "powerset_max_classes": model.specifications.powerset_max_classes,
            "problem": model.specifications.problem.name,
            "resolution": model.specifications.resolution.name,
            "warm_up": list(model.specifications.warm_up),
        },
    }
    metadata["format_version"] = METADATA_FORMAT_VERSION

    # prepare the files / protocol, and store them in metadata
    if isinstance(protocol, str):
        if len(protocol.split(".")) != 4:
            raise ValueError(
                f"Invalid protocol name {protocol}. Expected 'database.task.protocol.subset' format."
            )
        protocol_name, subset_name = protocol.rsplit(".", 1)
        metadata["protocol"] = protocol_name
        metadata["subset"] = subset_name
        real_protocol = registry.get_protocol(protocol_name, preprocessors=preprocessors)
        print(f"{metadata=}")
        files = list(getattr(real_protocol, subset_name)())
    elif isinstance(protocol, Iterable):
        files = list(protocol)
    metadata["protocol_files"] = [f"{f['database']} {f['uri']}" for f in files]

    # compute the final posteriors DataFrame
    print(f"Inference started at {metadata['date']['inference_start']}")

    # accumulate the output of all files into 'posteriors'
    posteriors: pd.DataFrame = None  # type: ignore
    for file_idx, file in (pbar := tqdm.tqdm(enumerate(files), total=len(files))):
        pbar.set_postfix_str(f"({pd.Timestamp.now().strftime('%H:%M:%S')}) {file['uri']}")
        output = inference(file)
        metadata["last_inference_shape"] = list(output.data.shape)

        if output.data.ndim == 2:
            df_tmp = _get_aggregated_posteriors_dataframe(output, file)
        elif output.data.ndim == 3:
            df_tmp = _get_non_aggregated_posteriors_dataframe(output, file, metadata)

        posteriors = pd.concat((posteriors, df_tmp))

    # postprocess
    posteriors.fillna(-1, inplace=True)

    # fill the metadata
    metadata["date"]["inference_end"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Inference done at {metadata['date']['inference_end']}. Saving ...")
    # save the inference result
    os.makedirs(outputs_folder, exist_ok=True)

    out_fid = output_identifier
    out_path = os.path.join(outputs_folder, f"{out_fid}.inf.{file_format}")
    outmeta_path = os.path.join(outputs_folder, get_inf_metadata_filename(out_fid))

    if file_format == "csv":
        posteriors.to_csv(out_path)
    else:
        if file_format != "parquet":
            warn("Unrecognized file format, using parquet instead")
        posteriors.to_parquet(out_path, compression="brotli", index=False)

    # save the metadata
    with open(outmeta_path, "w") as f:
        yaml.dump(metadata, f)
