from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from semantic_attack import AttackConfig, semantic_attack


MAX_SEMANTIC_IMAGES = 10

st.set_page_config(page_title="Semantic Attack Demo", layout="wide")
st.title("Image-based Semantic Attack")
st.write(
    "Embed semantic features from multiple semantic images into one victim image while keeping the result natural."
)


def main() -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        semantic_files = st.file_uploader(
            f"Semantic images (1-{MAX_SEMANTIC_IMAGES})",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="semantic",
        )
    with col_r:
        victim_file = st.file_uploader(
            "Victim image (1 file)",
            type=["png", "jpg", "jpeg"],
            key="victim",
        )

    st.subheader("Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        steps = st.slider("Optimization steps", 50, 400, 150, 10)
        clip_input_size = st.select_slider("CLIP input size", options=[224, 336], value=224)
    with c2:
        eps = st.slider("Perturbation bound eps (1/255)", 2, 30, 10, 1)
        semantic_weight = st.slider("Semantic weight", 0.1, 3.0, 1.0, 0.1)
    with c3:
        preserve_weight = st.slider("Preserve weight", 5.0, 100.0, 45.0, 1.0)
        tv_weight = st.slider("TV weight", 0.0, 0.2, 0.03, 0.01)

    run = st.button("Generate")
    if not run:
        return

    if victim_file is None:
        st.error("Please upload one victim image.")
        return

    if semantic_files is None or len(semantic_files) == 0:
        st.error("Please upload at least one semantic image.")
        return

    if len(semantic_files) > MAX_SEMANTIC_IMAGES:
        st.error(f"Too many semantic images: {len(semantic_files)}. Limit is {MAX_SEMANTIC_IMAGES}.")
        return

    semantic_imgs = [Image.open(f).convert("RGB") for f in semantic_files]
    victim_img = Image.open(victim_file).convert("RGB")

    cfg = AttackConfig(
        steps=steps,
        eps=eps / 255.0,
        semantic_weight=semantic_weight,
        preserve_weight=preserve_weight,
        tv_weight=tv_weight,
        clip_input_size=clip_input_size,
    )

    progress = st.progress(0)
    status = st.empty()

    def _progress(step: int, total: int, losses) -> None:
        progress.progress(step / total)
        status.text(
            f"step={step}/{total} total={losses['total']:.4f} "
            f"semantic={losses['semantic']:.4f} preserve={losses['preserve']:.6f}"
        )

    with st.spinner("Optimizing..."):
        adv_img, losses = semantic_attack(semantic_imgs, victim_img, cfg, progress_cb=_progress)

    st.success("Done.")
    st.caption(
        f"{len(semantic_imgs)} semantic images are used. If sizes differ, all semantic images are resized to victim resolution during optimization."
    )

    view_sem, view_vic, view_adv = st.columns(3)
    with view_sem:
        st.image(semantic_imgs[0], caption=f"Semantic sample (1/{len(semantic_imgs)})", width="stretch")
    with view_vic:
        st.image(victim_img, caption="Victim", width="stretch")
    with view_adv:
        st.image(adv_img, caption="Attacked", width="stretch")

    st.json(losses)

    buf = io.BytesIO()
    adv_img.save(buf, format="PNG")
    st.download_button(
        "Download attacked image",
        data=buf.getvalue(),
        file_name="semantic_attacked.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
