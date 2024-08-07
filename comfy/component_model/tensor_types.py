from jaxtyping import Float
from torch import Tensor

ImageBatch = Float[Tensor, "batch height width channels"]
RGBImageBatch = Float[Tensor, "batch height width 3"]
RGBAImageBatch = Float[Tensor, "batch height width 4"]
RGBImage = Float[Tensor, "height width 3"]
