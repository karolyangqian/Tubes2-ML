import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image

# image loader
def load_image(path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

# batch loader
def load_batch(paths: List[str], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    images = [load_image(p, target_size) for p in paths]
    return np.stack(images, axis=0)

# feature extractor
def extract_features(
    paths: List[str],
    keras_model,
    output_path: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> np.ndarray:
    save_path = Path(output_path)

    if save_path.exists():
        return np.load(save_path)

    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = load_batch(batch_paths, target_size)
        features = keras_model.predict(batch, verbose=0)
        all_features.append(features)

    result = np.concatenate(all_features, axis=0)
    np.save(save_path, result)
    return result