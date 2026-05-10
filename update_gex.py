name: Update GEX History

# ─────────────────────────────────────────────────────────────
# 段階 6B (v16): Theta Terminal セットアップの動作確認
# ─────────────────────────────────────────────────────────────
# このワークフローの責務:
#   1. Java 25 セットアップ
#   2. Theta Terminal v3 jar を公式 URL から動的 DL
#   3. secrets.THETA_CREDS を creds.txt として配置
#   4. Terminal をバックグラウンド起動
#   5. ヘルスチェック (リトライループで起動完了を待つ)
#   6. requirements.txt から依存パッケージをインストール
#   7. テストを走らせて 192 tests pass を確認
#   8. Mock Adapter で run_daily.py を実行し、gex_history.json を生成
#   9. 差分があれば自動コミット & push
#
# 段階 6B では:
#   - cron はコメントアウトのまま (段階 6D で UTC 22:30 を有効化)
#   - workflow_dispatch (手動実行) のみ有効
#   - GEX_DATA_SOURCE は "mock" に固定 (段階 6C で "rest" に切替)
#   - Theta Terminal は起動して疎通確認するだけで、Mock 実行には使われない
#     (この段階の目的は「Java + Terminal セットアップが Actions で動くこと」
#      の検証であり、データ取得は段階 6C 以降)
#
# 段階 6C で変更するもの:
#   - GEX_DATA_SOURCE: "rest" に切替
#
# 段階 6D で変更するもの:
#   - cron: '30 22 * * 1-5' を有効化
#
# ─────────────────────────────────────────────────────────────

on:
  # ── 段階 6B ではコメントアウトのまま ──
  # schedule:
  #   - cron: '30 22 * * 1-5'   # UTC 22:30 (ET 17:30 EDT / 18:30 EST)
  workflow_dispatch:           # 手動実行で疎通確認

permissions:
  contents: write              # gex_history.json を commit & push するため

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      # ───── 環境構築 ─────
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Setup Java
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '25'

      - name: Verify Java version
        run: java -version

      # ───── Theta Terminal セットアップ ─────
      - name: Download Theta Terminal v3 jar (from official URL)
        run: |
          curl -L -o ThetaTerminalv3.jar https://download-latest.thetadata.us
          ls -lh ThetaTerminalv3.jar

      - name: Configure creds.txt
        # secrets.THETA_CREDS には改行区切りで
        #   1 行目: メールアドレス
        #   2 行目: パスワード
        # を登録しておく前提
        run: |
          printf '%s\n' "${{ secrets.THETA_CREDS }}" > creds.txt
          # creds.txt の中身は出力しない（パスワード保護）
          # 行数のみ確認
          wc -l creds.txt

      - name: Start Theta Terminal in background
        # nohup でバックグラウンド起動、ログは theta_terminal.log に書き出す
        # 起動失敗時のデバッグのため、ログは後段で必ず確認する
        run: |
          nohup java -jar ThetaTerminalv3.jar > theta_terminal.log 2>&1 &
          echo "Started Theta Terminal with PID $!"
          echo "$!" > theta_terminal.pid

      - name: Wait for Theta Terminal health check
        # 最大 60 秒、1 秒ごとに /v3/stock/list/symbols を叩いて 200 を待つ
        # 段階 6B-1 確認2 で「Stock: FREE 環境でも候補A は 200 で返る」と確認済み
        #
        # curl の -w "%{http_code}" は接続失敗時も "000" を出力するので、
        # || echo "000" は不要（付けると "000000" の二重出力になる）。
        run: |
          set +e  # curl 失敗時に即座に exit しない
          for i in $(seq 1 60); do
            status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:25503/v3/stock/list/symbols?format=csv" 2>/dev/null)
            if [ "$status" = "200" ]; then
              echo "Theta Terminal is ready (took ${i} seconds)"
              exit 0
            fi
            echo "Attempt ${i}/60: HTTP $status, waiting..."
            sleep 1
          done
          echo "❌ Theta Terminal did not become ready within 60 seconds"
          echo "── theta_terminal.log (last 50 lines) ──"
          tail -50 theta_terminal.log || true
          exit 1

      - name: Show Theta Terminal startup log (for confirmation)
        # 起動ログを Actions ログに出力（バージョン番号や Subscription 状態を確認）
        # PROJECT_CONTEXT v16 で「実 API 環境で踏みやすい 3 つの地雷
        # (471 PERMISSION / 476 WRONG_IP / 478 多重起動)」を後から検証するため
        run: |
          echo "── theta_terminal.log (head 30 lines) ──"
          head -30 theta_terminal.log

      # ───── Python 依存と検証 ─────
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest gex_engine/tests/ -q

      # ───── Mock 駆動で daily run ─────
      - name: Run daily GEX update (Mock)
        # 段階 6B では Theta Terminal を起動するが、Mock を使うため
        # 実際にはデータ取得には使われない (段階 6C で rest に切替)
        env:
          GEX_DATA_SOURCE: mock
        run: python -m gex_engine.scripts.run_daily

      # ───── 結果コミット ─────
      - name: Commit gex_history.json if changed
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          git add gex_history.json
          git diff --quiet && git diff --staged --quiet \
            || (git commit -m "Auto-update GEX history (mock, stage 6B)" && git push)

      # ───── 後片付け (任意) ─────
      - name: Stop Theta Terminal
        # ジョブ終了時に runner ごと破棄されるので必須ではないが、
        # 明示的に停止することで「Terminal が動作していた」事実を確認できる
        if: always()
        run: |
          if [ -f theta_terminal.pid ]; then
            pid=$(cat theta_terminal.pid)
            kill "$pid" 2>/dev/null || true
            echo "Stopped Theta Terminal (PID $pid)"
          fi
