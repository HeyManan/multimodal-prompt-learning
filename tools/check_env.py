"""Report the GPU software versions and run CLIP ViT-B/16 once on the GPU."""
import logging
import sys

import torch

from utils.clip_model import BACKBONE, load_clip

IMAGE_SIZE = 224
EXPECTED_FEATURE_SHAPE = (1, 512)


def report_versions():
    logging.info("torch: %s", torch.__version__)
    logging.info("CUDA built with torch: %s", torch.version.cuda)
    logging.info("cuDNN: %s", torch.backends.cudnn.version())
    logging.info("torch.cuda.is_available(): %s", torch.cuda.is_available())


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a usable GPU, found torch.cuda.is_available() == False.")
    logging.info("GPU: %s", torch.cuda.get_device_name(0))


def run_clip_on_gpu(model):
    model = model.eval().cuda()
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=model.dtype).cuda()
    with torch.no_grad():
        features = model.encode_image(image)
    if tuple(features.shape) != EXPECTED_FEATURE_SHAPE:
        raise RuntimeError(
            "Expected image features of shape {}, found {}.".format(
                EXPECTED_FEATURE_SHAPE, tuple(features.shape)
            )
        )
    logging.info("CLIP %s ran on the GPU, feature shape %s", BACKBONE, tuple(features.shape))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout, force=True)
    logging.disable(logging.NOTSET)
    report_versions()
    require_gpu()
    run_clip_on_gpu(load_clip())
    logging.info("Environment check passed.")


if __name__ == "__main__":
    main()
