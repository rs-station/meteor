
from pydantic import BaseModel
import pandas as pd
from .settings import TV_WEIGHT_PARAMETER_NAME, K_PARAMETER_NAME


class MaximizerScanMetadata(BaseModel):
    """Structured data reporting a run of `ScalarMaximizer`"""
    parameter_name: str
    initial_negentropy: float
    optimal_parameter_value: float
    optimal_negentropy: float
    parameter_scan_results: list[list[float]]
    """ a list of [parameter, objective] pairs that were scanned """


class KparameterScanMetadata(MaximizerScanMetadata):
    parameter_name: str = K_PARAMETER_NAME
    map_sampling: float


class TvScanMetadata(MaximizerScanMetadata):
    parameter_name: str = TV_WEIGHT_PARAMETER_NAME
    map_sampling: float


class DiffmapMetadata(BaseModel):
    k_parameter_optimization: KparameterScanMetadata
    tv_weight_optmization: TvScanMetadata


class IterativeDiffmapMetadata(BaseModel):
    iterative_tv: pd.DataFrame
    final_tv_pass: DiffmapMetadata