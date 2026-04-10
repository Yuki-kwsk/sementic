from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from semantic_attack import AttackConfig, semantic_attack


st.set_page_config(page_title="Semantic Attack Demo", layout="wide")
st.title("画像ベース Semantic Attack")
st.write(
    "意味的攪乱用画像の特徴を、被攪乱画像へ自然さを保ちながら埋め込むデモです。"
)


def main() -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        semantic_file = st.file_uploader(
            "意味的攪乱用画像", type=["png", "jpg", "jpeg"], key="semantic"
        )
    with col_r:
        victim_file = st.file_uploader(
            "被攪乱画像", type=["png", "jpg", "jpeg"], key="victim"
        )

    st.subheader("パラメータ")
    c1, c2, c3 = st.columns(3)
    with c1:
        steps = st.slider("最適化ステップ数", 50, 400, 150, 10)
        clip_input_size = st.select_slider("CLIP入力解像度", options=[224, 336], value=224)
    with c2:
        eps = st.slider("摂動上限 eps (1/255)", 2, 30, 10, 1)
        semantic_weight = st.slider("意味特徴重み", 0.1, 3.0, 1.0, 0.1)
    with c3:
        preserve_weight = st.slider("自然さ保持重み", 5.0, 100.0, 45.0, 1.0)
        tv_weight = st.slider("TV重み", 0.0, 0.2, 0.03, 0.01)

    run = st.button("攪乱を生成")
    if not run:
        return

    if semantic_file is None or victim_file is None:
        st.error("2枚の画像をアップロードしてください。")
        return

    semantic_img = Image.open(semantic_file).convert("RGB")
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

    with st.spinner("最適化中..."):
        adv_img, losses = semantic_attack(semantic_img, victim_img, cfg, progress_cb=_progress)

    st.success("生成が完了しました。")
    st.caption("画像サイズが異なる場合、被攪乱画像のサイズに合わせて処理しています。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(semantic_img, caption="意味的攪乱用画像", use_container_width=True)
    with c2:
        st.image(victim_img, caption="被攪乱画像", use_container_width=True)
    with c3:
        st.image(adv_img, caption="攪乱後画像", use_container_width=True)

    st.json(losses)

    buf = io.BytesIO()
    adv_img.save(buf, format="PNG")
    st.download_button(
        "攪乱後画像をダウンロード",
        data=buf.getvalue(),
        file_name="semantic_attacked.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
