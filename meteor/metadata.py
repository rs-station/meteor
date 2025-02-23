from pydantic import BaseModel

from .settings import K_PARAMETER_NAME, TV_WEIGHT_PARAMETER_NAME


class EvaluatedPoint(BaseModel):
    parameter_value: float
    objective_value: float


class MaximizerScanMetadata(BaseModel):
    parameter_name: str
    initial_negentropy: float
    optimal_parameter_value: float
    optimal_negentropy: float
    parameter_scan_results: list[EvaluatedPoint]


class KparameterScanMetadata(MaximizerScanMetadata):
    parameter_name: str = K_PARAMETER_NAME


class TvScanMetadata(MaximizerScanMetadata):
    parameter_name: str = TV_WEIGHT_PARAMETER_NAME
    map_sampling: float


class DiffmapMetadata(BaseModel):
    k_parameter_optimization: KparameterScanMetadata | None
    tv_weight_optmization: TvScanMetadata


class TvIterationMetadata(BaseModel):
    iteration: int
    tv_weight: float
    negentropy_after_tv: float
    average_phase_change: float


class IterativeDiffmapMetadata(BaseModel):
    kparameter_metadata: KparameterScanMetadata | None
    iterative_tv_iterations: list[TvIterationMetadata]
    final_tv_pass: TvScanMetadata
