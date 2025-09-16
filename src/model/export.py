import torch
import onnxruntime as ort
from src.config import Config
from src.exception import PipelineError
from src.logger import get_logger

logger = get_logger()

# Check PyTorch version for dynamo compatibility
PYTORCH_VERSION = torch.__version__.split('+')[0]
PYTORCH_MAJOR, PYTORCH_MINOR = map(int, PYTORCH_VERSION.split('.')[:2])
USE_DYNAMO = PYTORCH_MAJOR >= 2 and PYTORCH_MINOR >= 9

def model_to_onnx(model: torch.nn.Module, path: str):
    """Convert PyTorch model to ONNX and create inference session (Model Export Stage)."""
    try:
        model.eval()
        dummy_input = torch.randn(1, Config().context_len, Config().input_size).to(Config().device)
        export_kwargs = {
            "model": model,
            "args": dummy_input,
            "f": path,
            "input_names": ['input'],
            "output_names": ['output'],
            "dynamic_axes": {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            "opset_version": 12
        }
        if USE_DYNAMO:
            export_kwargs["dynamo"] = True
        torch.onnx.export(**export_kwargs)
        return ort.InferenceSession(path)
    except Exception as e:
        logger.error(f"Failed to export model to ONNX at {path}: {e}")
        raise PipelineError(f"Failed to export model to ONNX at {path}: {e}")