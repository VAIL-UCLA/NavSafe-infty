"""NuRec gRPC rendering for NavSafe."""
from navsafe.render.base import SceneRenderer
from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer

__all__ = ["SceneRenderer", "NuRecGrpcSceneRenderer", "create_renderer"]


def create_renderer(mode: str = "nurec_grpc", **kwargs) -> SceneRenderer:
    """Create the supported NuRec renderer; reject removed backends."""
    if mode != "nurec_grpc":
        raise ValueError(f"Unsupported renderer {mode!r}; use 'nurec_grpc'.")
    return NuRecGrpcSceneRenderer(**kwargs)
