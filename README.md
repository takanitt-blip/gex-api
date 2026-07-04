# gex-api

SPY オプション市場の EOD データから GEX（Gamma Exposure）水準を計算し、
MT5 EA での可視化・環境判別に利用するためのプロジェクト。

このリポジトリは性質の異なる2種類のコードを含む：

```
gex_engine/   Python データパイプライン
  ・ThetaData REST API から SPY オプションの EOD データを取得
  ・Call Wall / Put Wall / Zero Gamma / Max Pain を計算
  ・結果を gex_history.json に日次追記（GitHub Actions cron 経由）

mt5/          MT5 EA（MQL5）
  ・gex_history.json を GitHub raw 経由で読み込み、MT5 チャート上に
    各水準を階段状に描画する可視化専用 EA
  ・トレード判断・自動売買ロジックは未実装（将来の Step で追加予定）

tools/        検証・分析用スクリプト（CI 非実行）
  ・優位性検証（統計検定）、座標変換等
  ・出力 CSV は個人環境のデータに依存するため .gitignore 対象

gex_history.json
  ・cron が自動更新する現役の履歴ファイル。手動コミットしない
```

## データフロー

```
ThetaData API
      │  (gex_engine/, GitHub Actions cron)
      ▼
gex_history.json（このリポジトリの main ブランチ）
      │  (WebRequest 経由、raw.githubusercontent.com)
      ▼
mt5/Gex_visualizer.mq5（MT5 チャート上で可視化）
```

Python 側と MT5 側はソースコードとしては独立しており、
`gex_history.json` を介した一方向・読み取り専用の実行時依存のみで
繋がっている（同じリポジトリで管理しているのはバージョン管理と
変更履歴を一元化するため）。

## セットアップ

Python 側は `requirements.txt` を参照。MT5 側は `mt5/Gex_visualizer.mq5`
を MetaEditor でコンパイルし、MT5 のチャートにアタッチする。
