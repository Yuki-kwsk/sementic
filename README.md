# Semantic Attack Demo (Image-Based)

意味的攪乱用画像の特徴を、被攪乱画像へ埋め込むシンプルなデモ実装です。  
自然な見た目を維持するため、以下を同時に最適化します。

- 意味特徴一致: CLIP (`openai/clip-vit-large-patch14`) 画像埋め込みが意味画像に近づく
- 見た目保持: 被攪乱画像から大きく離れない
- 平滑化: TV lossで局所ノイズを抑える
- 摂動上限: `eps` 以内に摂動を制限
- サイズ整合: 画像サイズが異なる場合は被攪乱画像サイズ基準で処理

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行

```bash
streamlit run app.py
```

ブラウザUIで以下を指定してください。

1. 意味的攪乱用画像（特徴を移したい画像）
2. 被攪乱画像（攪乱を加える画像）
3. パラメータ（`steps`, `eps`, `clip_input_size`, 各重み）

## パラメータの目安

- 自然さ優先: `preserve_weight` を上げる / `eps` を下げる
- 意味移植を強く: `semantic_weight` を上げる
- ノイズ感を減らす: `tv_weight` を上げる

## 注意

これは研究用の最小デモです。モデルやタスクに応じて、特徴抽出器や損失設計の調整が必要です。
