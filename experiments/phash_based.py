import imagehash
import numpy as np
from PIL import Image


def compute_phash_vector_from_image(
    img: Image.Image,
    hash_size: int = 32,
    normalise: bool = True,
    dtype: type = np.float32,
    binary_output: bool = True,
) -> np.ndarray:
    """Compute phash vector from a PIL Image."""
    img_hash = imagehash.phash(img, hash_size=hash_size)
    bits = bin(int(str(img_hash), 16))[2:].zfill(hash_size * hash_size)
    v = np.array([int(b) for b in bits], dtype=dtype if binary_output else int)

    if normalise:
        v = v / (np.linalg.norm(v) + 1e-8)

    return v


def cosine_similarity(a: np.ndarray, b: np.ndarray):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



img1 = Image.open("ORG_39 - コピー (2).jpeg")
img2 = Image.open("Dwg1 (3) - コピー.jpeg")

p1 = compute_phash_vector_from_image(img1)
p2 = compute_phash_vector_from_image(img2)

print(f"cs: {cosine_similarity(p1, p2)}")

