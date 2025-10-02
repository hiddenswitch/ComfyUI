from typing import TypedDict

from jaxtyping import Float
from torch import Tensor
from typing_extensions import NotRequired

AudioTensor = Float[Tensor, "batch channel length"]


class Audio(TypedDict):
    """A dictionary representing an audio clip."""
    waveform: AudioTensor
    sample_rate: int


class LatentAudio(TypedDict):
    samples: AudioTensor
    type: NotRequired[str]
