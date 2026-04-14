from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from semantic_attack import AttackConfig, semantic_attack


MAX_SEMANTIC_IMAGES = 10
PRESET_NAME_KEY = "preset_name"
PRESET_INIT_KEY = "preset_initialized"
DEFAULT_PRESET_NAME = "Balanced (Recommended)"
PRESET_VALUES = {
    "Balanced (Recommended)": {
        "steps": 180,
        "clip_input_size": 224,
        "eps_255": 10,
        "semantic_weight": 1.0,
        "preserve_weight": 45.0,
        "ssim_weight": 2.0,
        "lpips_weight": 0.5,
        "lpips_net": "alex",
        "tv_weight": 0.03,
    },
    "Semantic Strong": {
        "steps": 220,
        "clip_input_size": 224,
        "eps_255": 14,
        "semantic_weight": 1.5,
        "preserve_weight": 35.0,
        "ssim_weight": 1.2,
        "lpips_weight": 0.3,
        "lpips_net": "alex",
        "tv_weight": 0.02,
    },
    "Natural Strong": {
        "steps": 200,
        "clip_input_size": 224,
        "eps_255": 8,
        "semantic_weight": 0.8,
        "preserve_weight": 60.0,
        "ssim_weight": 4.0,
        "lpips_weight": 0.9,
        "lpips_net": "vgg",
        "tv_weight": 0.05,
    },
    "Fast Preview": {
        "steps": 80,
        "clip_input_size": 224,
        "eps_255": 10,
        "semantic_weight": 1.0,
        "preserve_weight": 40.0,
        "ssim_weight": 1.5,
        "lpips_weight": 0.4,
        "lpips_net": "alex",
        "tv_weight": 0.02,
    },
}


def _apply_preset_values(preset_name: str) -> None:
    params = PRESET_VALUES[preset_name]
    for k, v in params.items():
        st.session_state[k] = v

st.set_page_config(page_title="Semantic Attack Demo", layout="wide")
st.title("Image-based Semantic Attack")
st.write(
    "Embed semantic features from multiple semantic images into one victim image while keeping the result natural."
)


def main() -> None:
    if PRESET_INIT_KEY not in st.session_state:
        st.session_state[PRESET_NAME_KEY] = DEFAULT_PRESET_NAME
        _apply_preset_values(DEFAULT_PRESET_NAME)
        st.session_state[PRESET_INIT_KEY] = True

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

    preset_l, preset_r = st.columns([2, 3])
    with preset_l:
        st.selectbox(
            "Recommended preset",
            options=list(PRESET_VALUES.keys()),
            key=PRESET_NAME_KEY,
        )
    with preset_r:
        st.caption(
            "A preset updates all shown parameters. You can tweak sliders after applying."
        )
    if st.button("Apply preset"):
        _apply_preset_values(st.session_state[PRESET_NAME_KEY])
        st.rerun()

    st.subheader("Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        steps = st.slider("Optimization steps", 50, 400, step=10, key="steps")
        clip_input_size = st.select_slider("CLIP input size", options=[224, 336], key="clip_input_size")
    with c2:
        eps_255 = st.slider("Perturbation bound eps (1/255)", 2, 30, step=1, key="eps_255")
        semantic_weight = st.slider("Semantic weight", 0.1, 3.0, step=0.1, key="semantic_weight")
    with c3:
        preserve_weight = st.slider("Preserve weight", 5.0, 100.0, step=1.0, key="preserve_weight")
        ssim_weight = st.slider("SSIM weight", 0.0, 20.0, step=0.5, key="ssim_weight")
    with c4:
        lpips_weight = st.slider("LPIPS weight", 0.0, 5.0, step=0.1, key="lpips_weight")
        lpips_net = st.selectbox("LPIPS backbone", options=["alex", "vgg", "squeeze"], key="lpips_net")
        tv_weight = st.slider("TV weight", 0.0, 0.2, step=0.01, key="tv_weight")
    st.caption("Learning rate is fixed to 0.03 in this demo to keep preset behavior stable.")

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
        eps=eps_255 / 255.0,
        semantic_weight=semantic_weight,
        preserve_weight=preserve_weight,
        ssim_weight=ssim_weight,
        lpips_weight=lpips_weight,
        tv_weight=tv_weight,
        clip_input_size=clip_input_size,
        lpips_net=lpips_net,
    )

    progress = st.progress(0)
    status = st.empty()

    def _progress(step: int, total: int, losses) -> None:
        progress.progress(step / total)
        status.text(
            f"step={step}/{total} total={losses['total']:.4f} "
            f"semantic={losses['semantic']:.4f} preserve={losses['preserve']:.6f} "
            f"ssim={losses['ssim']:.4f} lpips={losses['lpips']:.4f}"
        )

    with st.spinner("Optimizing..."):
        try:
            adv_img, losses = semantic_attack(semantic_imgs, victim_img, cfg, progress_cb=_progress)
        except RuntimeError as exc:
            st.error(str(exc))
            return

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
