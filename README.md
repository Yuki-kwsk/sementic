# Semantic Attack Demo (CLIP)

This app embeds semantic features from one or more semantic images into a victim image,
while keeping the output visually natural.

Current settings:
- CLIP model: `openai/clip-vit-large-patch14`
- Max semantic images: `10`
- If image sizes differ, processing is aligned to the victim image resolution.

## Objective

The optimization uses:
- Semantic loss: cosine distance between CLIP image embeddings
- Preserve loss: MSE to keep appearance close to victim image
- TV loss: smoothness regularization
- Epsilon constraint: perturbation is clipped within `eps`

For multiple semantic images, their CLIP embeddings are averaged and used as the target semantic feature.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Upload:
1. Semantic images (1 to 10 files)
2. Victim image (1 file)
