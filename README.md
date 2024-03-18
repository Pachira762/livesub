# LiveSub

## 概要

デスクトップの音声をキャプチャし、 [Distil-Whisper](https://github.com/huggingface/distil-whisper) を使用してリアルタイムで文字に起こします。  

https://github.com/Pachira762/livesub/assets/34003267/3ec1d82b-6529-4918-b9fd-54bccbabf99b



## 使い方

- NVIDIA GTX1060 以上のグラフィックボードが必須です。
- 起動するとモデルのダウンロードが開始します。少々お待ちください。  
- ウィンドウはドラッグで移動できます。
- ```ESCAPE``` キーでウィンドウを閉じます。
- 右クリックでメニューが開きます。 使用するモデルや遅延の量、背景の透過度、フォントなどが変えられます。


## ビルド

CUDA 12を使用してビルドしています。  
CUDA 11をインストールして ```.cargo/config``` の ```CUDA_COMPUTE_CAP``` を編集すればより古いグラフィックボードでも動くかもしれません。  
target-feature に ```avx2``` を指定することでパフォーマンスが向上するかもしれません。


## 連絡先

問題の報告や機能の提案など、フィードバックは常に歓迎しています。  
[@pachira762](https://twitter.com/pachira762)
