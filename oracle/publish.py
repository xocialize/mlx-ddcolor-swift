"""Publish the converted DDColor fp16 weights to mlx-community + group them in a Collection.

Repos (mlx-community naming, preserve upstream checkpoint identity, -fp16 quant suffix):
  tiny     -> mlx-community/DDColor-paper-tiny-fp16
  large    -> mlx-community/DDColor-modelscope-fp16
  artistic -> mlx-community/DDColor-artistic-fp16
"""
import os
from huggingface_hub import HfApi, create_collection, add_collection_item

API = HfApi()
HERE = os.path.dirname(__file__)

VARIANTS = {
    # key: (repo, checkpoint, backbone, blurb)
    "tiny": ("DDColor-paper-tiny-fp16", "ddcolor_paper_tiny.pth", "ConvNeXt-T",
             "Fastest tier (~0.1 s). Lightest backbone — good for batch / on-device."),
    "large": ("DDColor-modelscope-fp16", "ddcolor_modelscope.pth", "ConvNeXt-L",
              "Best general tier. The recommended default — strongest on archival / multi-object scenes."),
    "artistic": ("DDColor-artistic-fp16", "ddcolor_artistic.pth", "ConvNeXt-L",
                 "Stylized tier — more saturated / interpretive palette."),
}


def card(repo, checkpoint, backbone, blurb):
    return f"""---
library_name: mlx
license: apache-2.0
license_link: https://github.com/piddnad/DDColor/blob/main/LICENSE
base_model: piddnad/DDColor-models
pipeline_tag: image-to-image
tags:
  - mlx
  - colorization
  - image-colorization
  - ddcolor
---

# mlx-community/{repo}

[DDColor](https://github.com/piddnad/DDColor) automatic image colorization ({backbone} backbone,
`{checkpoint}`), converted to **Apple MLX** (`-fp16`) for Apple-Silicon inference via the
[`mlx-ddcolor-swift`](https://github.com/xocialize/mlx-ddcolor-swift) Swift package.

{blurb}

A from-scratch MLX-Swift architecture port (no MLX donor) of DDColor: ConvNeXt encoder → quasi-UNet
decoder → DETR-style multi-scale color decoder (100 color queries) → predicts the **ab** chroma
channels, recombined with the source **L** (luminance). Grayscale / desaturated image in → colorized
image at the same resolution out.

## Use with mlx-ddcolor-swift (Swift / MLX)

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-ddcolor-swift", from: "0.1.0")
import DDColor

let colorizer = try DDColorColorizer.fromPretrained(
    weightsPath,                                   // model.safetensors from this repo
    config: DDColorConfig(tier: .large),           // .tiny for paper-tiny
    dtype: .float16)
let colorized: CGImage = colorizer(sourceCGImage)  // grayscale → color, source resolution
```

Or as an MLXEngine `imageColorize` ModelPackage (`MLXDDColor.DDColorPackage`), which resolves this repo
automatically via the Hub.

## Parity

Architecture parity-locked against the PyTorch oracle (piddnad/DDColor) on the CPU stream, fp32:
**max_abs ≤ 1.5e-6** (large) / **3.3e-6** (tiny) on the model's ab output. The cv2-exact LAB pre/post
pipeline matches the reference `ColorizationPipeline` to **~9e-4 mean** BGR error. This `-fp16` build is
visually identical to fp32 (**cosine 1.000000**, mean 0.04/255 on colorized output).

Weights: Apache-2.0 (piddnad/DDColor). Port code: MIT.
"""


def main():
    repos = []
    for key, (repo, checkpoint, backbone, blurb) in VARIANTS.items():
        rid = f"mlx-community/{repo}"
        weights = os.path.join(HERE, "publish", key, "model.safetensors")
        assert os.path.exists(weights), weights
        print(f"[publish] {rid} …")
        API.create_repo(rid, repo_type="model", exist_ok=True)
        API.upload_file(path_or_fileobj=weights, path_in_repo="model.safetensors", repo_id=rid)
        readme = card(repo, checkpoint, backbone, blurb).encode()
        API.upload_file(path_or_fileobj=readme, path_in_repo="README.md", repo_id=rid)
        repos.append((rid, blurb))
        print(f"[publish]   ok → https://huggingface.co/{rid}")

    print("[collection] creating DDColor collection …")
    col = create_collection(title="DDColor (MLX)", namespace="mlx-community", exist_ok=True,
                            description="Apple-MLX fp16 builds of DDColor automatic image colorization "
                                        "(piddnad/DDColor, Apache-2.0). Loaded by mlx-ddcolor-swift.")
    for rid, note in repos:
        try:
            add_collection_item(col.slug, item_id=rid, item_type="model", note=note[:500])
        except Exception as e:
            print(f"[collection]   {rid}: {e}")
    print(f"[collection] ok → https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
