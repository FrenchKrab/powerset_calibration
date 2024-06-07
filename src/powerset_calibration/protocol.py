"""Contains the Active learning protocol. The advised way to interact with the library :)

"""

import sys
from enum import Flag, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
import copy
import torch
from pandas import DataFrame
from pyannote.audio.utils.powerset import Powerset
from pyannote.core import SlidingWindow, Timeline
from pyannote.database import FileFinder, ProtocolFile, registry
from pyannote.database.protocol import Protocol, SpeakerDiarizationProtocol
from pyannote.database.registry import Registry

from powerset_calibration.inference import (
    InferenceMetadata,
    load_inference_file,
    load_inference_metadata,
)
from powerset_calibration.utils.inference import (
    get_correctshaped_filetensor,
    get_output_columns,
    get_reference_columns,
    aggregate_sliding_window_tensor,
    apply_aggregation_strategy,
    get_heuristic_tensor_segmentation,
)
from powerset_calibration.region_selector import (
    generate_windows,
    generate_windows_bins,
    generate_windows_multiplefiles,
    generate_windows_multiplefiles_bins,
    tensor_to_timeline,
)


def fix_uem_tensor(uem_t: torch.Tensor, max_consecutive_wrongs: int = 1) -> torch.Tensor:
    """Fixes UEM tensors that have been aggregated and have "intermediate" values instead
    of pure 0.0 and 1.0. The point of the function is to only fix tensors that are NOT
    too wrong.

    Parameters
    ----------
    uem_t : torch.Tensor
        Input UEM tensor
    max_consecutive_wrongs : int, optional
        Number of consecutive wrong values allowed, by default 1

    Returns
    -------
    torch.Tensor
        Fixed up (if needed) tensor
    """

    unique_vals, unique_counts = uem_t.unique_consecutive(return_counts=True)
    for v, c in zip(unique_vals, unique_counts):
        if v == 0 or v == 1:
            continue
        elif c > max_consecutive_wrongs:
            raise ValueError(
                f"Invalid uem_t ! Too many consecutive wrong values ! : {v} has {c} consecutive values !"
            )
    result = uem_t.clone()
    result[(uem_t != 0) & (uem_t != 1)] = 0
    return result


class ActiveLearningProtocolSettings(TypedDict):
    """Settings to parametrize the active learning protocol."""

    inference: Union[str, Path, None]
    """Path to the inference file to use."""

    scope: Literal["file", "dataset"]
    """Scope of the active learning duration constraint :
    - `file` to select x% of each file
    - `dataset` to select x% of data in the whole dataset
    """

    window_duration: float
    """Duration of the active learning windows in seconds (ie minimum time of an annotated segment)."""

    sliding_window_step: float
    """Size of the step to move the sliding window in seconds."""

    annotated_ratio: Optional[float]
    """Ratio of the data that should be annotated (eg 0.1 for 10% of the data)."""

    annotated_duration: Optional[float]
    """Amount of data to annotate (eg 20.0 for 20 seconds)."""

    heuristic: Literal["confidence", "bce", "ce", "random", "entropy"]
    """Heuristic to use to compute the confidence of the model in each window."""

    heuristic_direction: Literal["worst", "best", "bins"]
    """Direction of the heuristic, 'worst' is the canonical way to do active learning (ie select the worst windows).
    'best' is the opposite, it selects windows where the model is the best. 'bins' split possible windows into
    heuristic bins and select one example in each bin."""

    heuristic_bins: Optional[int]
    """Number of bins to use when the heuristic is 'bins'."""

    aggregation: Literal["lazy", "soft_segmentation"]
    """How to aggregate the model outputs to obtain the heuristic (when the data
    in the inference file is not yet aggregated).
    - `lazy` = compute the heuristic and then aggregate it.
    - `soft_segmentation` = aggregate the model outputs with SoftSpeakerSegmentation-like pipeline and then compute the heuristic."""

    selections_per_iteration: int
    """Number of selections to make per iteration. Use math.inf if you dont know (see the `generate_windows` fn)"""


def get_default_active_learning_protocol_settings() -> ActiveLearningProtocolSettings:
    return {
        "inference": None,
        "scope": "dataset",
        "window_duration": 10.0,
        "sliding_window_step": 1.0,
        "annotated_ratio": None,
        "annotated_duration": None,
        "heuristic": "confidence",
        "heuristic_direction": "worst",
        "heuristic_bins": None,
        "aggregation": "soft_segmentation",
        "selections_per_iteration": sys.maxsize,
    }


class ActiveLearningProtocolDebugFlags(Flag):
    SAVE_OUT = auto()
    SAVE_REF = auto()
    SAVE_UEM = auto()
    SAVE_HEUR = auto()


class ActiveLearningProtocol(SpeakerDiarizationProtocol):
    """Active learning protocol wrapper.
    Computes the active learning windows for each file in the protocol.
    """

    active_learning_uems: Dict[str, Timeline]
    """Maps each URI to a timeline representing the (active learning) annotated segments."""

    settings: Dict[Union[str, Tuple[str]], ActiveLearningProtocolSettings]
    """Maps subset(s) to one settings dict. See constructor for details"""

    def __init__(
        self,
        protocol: Union[Protocol, str],
        settings: Dict[Union[str, Tuple[str]], ActiveLearningProtocolSettings],
        protocol_preprocessors: Optional[Dict[str, Callable]] = None,
        protocol_registry: Optional[Registry] = None,
        uem_save_path: Optional[Path] = None,
        debug_flags: Optional[ActiveLearningProtocolDebugFlags] = None,
    ):
        """Active learning protocol wrapper.

        Parameters
        ----------
        protocol : Union[Protocol, str]
            Protocol to use (if Protocol) or load (if str)
        settings : Dict[Union[str, Tuple[str]], ActiveLearningProtocolSettings]
            Maps a subset or tuple of subset to an active learning settings dict.
            A subset is 'train', 'development' or 'test'.
            If a tuple of subset is provided (e.g. `settings={('train','val',): {...}}`),
            the active learning windows will be computed for all subsets in the
            tuple (meaning they share the total time to annotate !).
        protocol_preprocessors : Dict[str, Callable], optional
            Preprocessors to use if the provided `protocol` is not
            yet loaded (= is a str). Default to just a FileFinder if not specified.
            By default None
        protocol_registry : Registry, optional
            Registry to use if the provided `protocol` is not
            yet loaded (= is a str). If left to None, will use pyannote.database's
            default registry, by default None
        uem_save_path : Path, optional
            If specified, will save the active learning windows in a UEM file at the given path.
        """
        super().__init__(preprocessors=None)
        self.settings = settings

        # load protocol if not already loaded
        self.protocol = protocol
        if isinstance(self.protocol, str):
            if protocol_registry is None:
                protocol_registry = registry
            if protocol_preprocessors is None:
                protocol_preprocessors = {"audio": FileFinder()}
            self.protocol = protocol_registry.get_protocol(
                self.protocol, preprocessors=protocol_preprocessors
            )

        self.uem_save_path = uem_save_path
        self.debug_flags = (
            debug_flags if debug_flags is not None else ActiveLearningProtocolDebugFlags(0)
        )
        self.debug_data = {"out": {}, "ref": {}, "uem": {}, "raw_out": {}, "heur": {}}

        # compute active learning windows
        self.active_learning_uems: Dict[str, Timeline] = {}
        self.create_active_learning_uems()

    @property
    def name(self) -> str:
        return f"AL..{getattr(self.protocol, 'name', 'unknown')}"

    @staticmethod
    def iterate_df_files(
        df: DataFrame, metadata: InferenceMetadata
    ) -> Generator[dict[str, Any], Any, None]:
        """Iterate on all files in the inference DataFrame.
        The tensor contained in the dictionaries are reshaped.
        2D tensors mean that the data was aggregated, 3D tensors mean it was not.

        Parameters
        ----------
        df : DataFrame
            The inference Dataframe
        metadata : InferenceMetadata
            The inference metadata

        Yields
        ------
        Generator[dict[str, Any], Any, None]
            Generate dictionaries containing keys
            - 'uri' : str
            - 'uem', 'ref' and S'out' : torch.Tensor
        """
        output_cols = get_output_columns(df.columns)
        ref_cols = get_reference_columns(df.columns)

        for uri in df["uri"].unique():
            filedf = df[df["uri"] == uri]
            uem_t = get_correctshaped_filetensor(torch.from_numpy(filedf["uem"].values), metadata)

            ref_t = get_correctshaped_filetensor(
                torch.stack([torch.from_numpy(filedf[col].values) for col in ref_cols], dim=-1),
                metadata,
            )
            valid_speakers_in_ref = (ref_t >= 0).all(dim=0).all(dim=0)
            ref_t = ref_t[..., valid_speakers_in_ref]

            out_t = get_correctshaped_filetensor(
                torch.stack(
                    [torch.from_numpy(filedf[col].values) for col in output_cols],
                    dim=-1,
                ),
                metadata,
            )
            data = {
                "uri": uri,
                "uem": uem_t,
                "ref": ref_t,
                "out": out_t,
            }
            # check all tensor have the same number of frames
            tensor_values = [v for v in data.values() if isinstance(v, torch.Tensor)]
            for i, val in enumerate(tensor_values[:-1]):
                if val.shape[0] != tensor_values[i + 1].shape[0]:
                    raise ValueError(
                        f"Invalid shape for {list(data.keys())[i]} : {val.shape} != {tensor_values[i+1].shape}. Different # of frames !"
                    )

            yield data

    def _generate_uems(
        self,
        settings: ActiveLearningProtocolSettings,
        scope: Literal["file", "dataset"],
        uris: Optional[List[str]] = None,
    ):
        if scope not in ["file", "dataset"]:
            raise ValueError(f"Unknown scope {scope}")

        # --- gather useful data

        infdf = load_inference_file(str(settings["inference"]))
        metadata: InferenceMetadata = load_inference_metadata(settings["inference"])

        if uris is not None:
            infdf = infdf[infdf["uri"].isin(uris)]
        print(f"Generating active learning windows from {len(infdf)} samples")

        is_aggregated = len(metadata["last_inference_shape"]) == 2
        model_fps = metadata["model"]["fps"]
        model_sw = SlidingWindow(metadata["model"]["duration"], metadata["step_duration"])
        model_frames = SlidingWindow(
            metadata["model"]["frames"]["duration"],
            metadata["model"]["frames"]["step"],
        )
        powerset = Powerset(
            len(metadata["model"]["specifications"]["classes"]),
            metadata["model"]["specifications"]["powerset_max_classes"],
        )
        argminmax = torch.argmin if settings["heuristic_direction"] == "worst" else torch.argmax

        # ---- read the data, compute the heuristic(, generate active learning windows if scope == 'file')

        if scope == "dataset":
            cat_uris = []
            cat_heur = []
            cat_uem = []
        else:
            result: Dict[str, Timeline] = {}

        for i, fdata in enumerate(ActiveLearningProtocol.iterate_df_files(infdf, metadata)):
            if not is_aggregated:
                out: torch.Tensor = apply_aggregation_strategy(
                    settings["aggregation"],
                    fdata["out"],
                    sliding_window=model_sw,
                    powerset=powerset,
                    frames=model_frames,
                )
                ref = apply_aggregation_strategy(
                    settings["aggregation"],
                    fdata["ref"],
                    sliding_window=model_sw,
                    powerset=None,
                    frames=model_frames,
                )
                f_uem: torch.Tensor = aggregate_sliding_window_tensor(
                    fdata["uem"], model_sw, frames=model_frames
                )[..., 0]
            else:
                out = fdata["out"]
                ref = fdata["ref"]
                f_uem = fdata["uem"]

            f_uem = fix_uem_tensor(f_uem, max_consecutive_wrongs=1).bool()
            f_heur = get_heuristic_tensor_segmentation(
                heuristic=settings["heuristic"],
                sliding_window=model_sw,
                preds=out,
                targets=ref,
                powerset=powerset,
                frames=model_frames,
            )

            if ActiveLearningProtocolDebugFlags.SAVE_OUT in self.debug_flags:
                self.debug_data["out"][fdata["uri"]] = out
                self.debug_data["raw_out"][fdata["uri"]] = fdata["out"]
            if ActiveLearningProtocolDebugFlags.SAVE_REF in self.debug_flags:
                self.debug_data["ref"][fdata["uri"]] = ref
            if ActiveLearningProtocolDebugFlags.SAVE_UEM in self.debug_flags:
                self.debug_data["uem"][fdata["uri"]] = f_uem
            if ActiveLearningProtocolDebugFlags.SAVE_HEUR in self.debug_flags:
                self.debug_data["heur"][fdata["uri"]] = f_heur

            if scope == "dataset":
                # print(f"f_heur.shape = {f_heur.shape}")
                # print(f"adding to cat_uris {torch.full_like(f_heur, i, dtype=torch.long).shape}")
                cat_uris.append(torch.full_like(f_heur, i, dtype=torch.long))
                cat_heur.append(f_heur)
                cat_uem.append(f_uem)
            elif scope == "file":
                if settings["heuristic_direction"] == "bins":
                    if settings["heuristic_bins"] is None:
                        raise ValueError(
                            "Cannot use heuristic_direction='bins' without heuristic_bins !"
                        )
                    al_uem = generate_windows_bins(
                        values=f_heur,
                        fps=model_fps,
                        window_duration=settings["window_duration"],
                        sliding_window_step=settings["sliding_window_step"],
                        bins_count=settings["heuristic_bins"],
                        samples_per_quantile=1,
                    )
                else:
                    al_uem = generate_windows(
                        values=f_heur,
                        fps=model_fps,
                        window_duration=settings["window_duration"],
                        sliding_window_step=settings["sliding_window_step"],
                        annotated_ratio=settings["annotated_ratio"],
                        annotated_duration=settings["annotated_duration"],
                        conv=torch.nn.functional.avg_pool1d,
                        argminmax=argminmax,
                        uem=f_uem,
                        selections_per_iteration=settings["selections_per_iteration"],
                    )
                al_timeline = tensor_to_timeline(al_uem, model_fps, uri=fdata["uri"])
                result[fdata["uri"]] = al_timeline

        # --- return the result (& compute it if scope == 'dataset')
        if scope == "dataset":
            cat_uris = torch.cat(cat_uris, dim=0)
            cat_heur = torch.cat(cat_heur, dim=0)
            cat_uem = torch.cat(cat_uem, dim=0)

            if settings["heuristic_direction"] == "bins":
                if settings["heuristic_bins"] is None:
                    raise ValueError(
                        "Cannot use heuristic_direction='bins' without heuristic_bins !"
                    )

                return generate_windows_multiplefiles_bins(
                    cat_heur,
                    cat_uris,
                    uris=infdf["uri"].unique(),
                    fps=model_fps,
                    window_duration=settings["window_duration"],
                    sliding_window_step=settings["sliding_window_step"],
                    bins_count=settings["heuristic_bins"],
                    samples_per_bin=1,
                    conv=torch.nn.functional.avg_pool1d,
                    t_uem=cat_uem,
                )
            else:
                return generate_windows_multiplefiles(
                    cat_heur,
                    cat_uris,
                    uris=infdf["uri"].unique(),
                    fps=model_fps,
                    window_duration=settings["window_duration"],
                    sliding_window_step=settings["sliding_window_step"],
                    annotated_ratio=settings["annotated_ratio"],
                    annotated_duration=settings["annotated_duration"],
                    selections_per_iteration=settings["selections_per_iteration"],
                    conv=torch.nn.functional.avg_pool1d,
                    argminmax=argminmax,
                    t_uem=cat_uem,
                )
        elif scope == "file":
            return result
        else:
            raise ValueError(f"Uncaught unknown scope {scope} ? How did this not crash ?")

    def create_active_learning_uems(self):
        """Automatically called by the constructor. Computes the UEMs for the given parameters."""
        self.active_learning_uems.clear()

        # Compute the active learning windows for each subset
        for subsets, ss_settings in self.settings.items():
            if isinstance(subsets, str):
                subsets = (subsets,)

            # initialize the settings
            default_settings = get_default_active_learning_protocol_settings()
            default_settings.update(ss_settings)
            ss_settings = default_settings

            # gather all URIs affected
            subsets_uris = set()
            for s in subsets:
                subset_iterator = getattr(self.protocol, f"{s}_iter")
                for f in subset_iterator():
                    subsets_uris.add(f["uri"])
            subsets_uris = list(subsets_uris)

            print(f"Found {len(subsets_uris)} URIs for subsets {subsets}")
            # find/compute the AL windows
            if ss_settings["scope"] == "file":
                result = self._generate_uems(ss_settings, "file", subsets_uris)
            elif ss_settings["scope"] == "dataset":
                result = self._generate_uems(ss_settings, "dataset", subsets_uris)
            else:
                raise ValueError(f"Unknown scope {ss_settings['scope']}")

            # check there is no overlap between what's already been computed and what we got
            result_inter_stored = set(result.keys()).intersection(self.active_learning_uems.keys())
            if len(result_inter_stored) > 0:
                raise ValueError(
                    f"Active learning windows for {result_inter_stored} already computed !"
                )

            self.active_learning_uems.update(result)

            if self.uem_save_path is not None:
                self.save_uems(self.uem_save_path)

    def save_uems(self, filepath: Union[str, Path]):
        """Save the active learning segments in a UEM file at the given path."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            for uri, uem in self.active_learning_uems.items():
                uem.write_uem(f)

    def _any_iter(self, subset: str) -> Iterator[Union[Dict, ProtocolFile]]:
        protocol_subset = getattr(self.protocol, subset)
        for f in protocol_subset():
            f = copy.copy(f)
            uri = f["uri"]
            if uri in self.active_learning_uems:
                f["annotated"] = self.active_learning_uems[uri]
            yield f

    def train_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self._any_iter("train")

    def development_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self._any_iter("development")

    def test_iter(self) -> Iterator[Union[Dict, ProtocolFile]]:
        return self._any_iter("test")
