//+------------------------------------------------------------------+
//|                                              GEX_Visualizer.mq5  |
//|                                                                    |
//|  GEX 環境判別エンジン Phase 1 - Step 1A/1B/1C                     |
//|                                                                    |
//|  【役割】                                                          |
//|   GitHub から取得した GEX 水準データを、各日付の時間範囲に          |
//|   限定して階段状に描画する。視覚確認専用 EA。                       |
//|                                                                    |
//|  【v1.20 変更点（Step 1B/1C、2026-07-04）】                        |
//|   Step 1B: dateKey（ETの暦日）を StringToTime() でそのまま         |
//|     サーバーローカル時刻と誤読していた欠陥を修正。ET側は米国連邦   |
//|     DST規則（3月第2日曜/11月第1日曜）を計算式で判定し、サーバー側  |
//|     は TimeGMT()/TimeTradeServer() の実行時差分で動的に変換する。  |
//|     固定オフセットのハードコードは一切行わない。                   |
//|   Step 1C: 各エントリの表示区間を「dateKey自身の暦日固定」から     |
//|     「as_of（EOD確定日）+1日 〜 次エントリの as_of+1日まで」に     |
//|     変更。GEXが表す実体（ディーラーのOI）は次の新規取引が成立      |
//|     するまで不変という物理的性質に対応し、休場日・週末を挟んでも   |
//|     壁が途切れず表示され続ける。実機のオブジェクトプロパティで     |
//|     数値一致を確認済み（2026-07-04）。                             |
//|                                                                    |
//|  【v1.21 変更点（監査20連動・データ不完全時の警告表示、2026-07-04）】|
//|   DrawOneDay の call/put/zero <= 0 ガードは「壁を描かない」という   |
//|   形で既に安全側に倒れていたが（0/null/未設定のいずれでもガードに   |
//|   掛かる）、その事実は Print()（エキスパートログ）にしか出ず、      |
//|   チャートだけを見ているトレーダーには「なぜ壁が消えたか」が伝わら  |
//|   なかった。ParseAndDraw でガード発動件数を集計し、1件でもあれば    |
//|   チャート左上にテキスト警告を表示するようにした（最新キー自体が   |
//|   不完全なら赤・最優先、過去日のみなら橙・参考情報）。              |
//|   Python 側（gex-api）の監査20対応（EMPTY_BOTH/EMPTY_ASYMMETRIC/    |
//|   MERGE_MISMATCH の ERROR ログ化）とは独立に、EA 単体でも安全に     |
//|   動作する。data_quality フィールドそのものの読み込み・4状態判別    |
//|   との連動（PC_MT5 EA改修タスク3/4）は依然未実装で、別途対応する。  |
//|                                                                    |
//|  【既知の相違点（要ユーザー確認）】                                |
//|   PC_MT5.md は「Max Pain(マゼンタ,破線)描画を2026-06-20 (v1.10)   |
//|   で実装完了」と記録しているが、本ファイルの実物（v1.00ベース）    |
//|   にも実機のオブジェクトリストにも Max Pain 描画は存在しない。     |
//|   ドキュメントと実装のドリフトの可能性があり、本改修では踏み込ま   |
//|   ず、PC_GOVERNANCE に確認事項として記録した。Max Pain 描画が      |
//|   別途必要であれば、改めて設計・実装すること。                     |
//|                                                                    |
//|  【スコープ外（将来の Step で実装）】                              |
//|   ・状態判別ロジック                                               |
//|   ・状態遷移検出                                                   |
//|   ・トレード判断                                                   |
//+------------------------------------------------------------------+
#property copyright "GEX Environment Detection Engine"
#property version   "1.21"
#property strict

//================================================================
// 入力パラメータ
//================================================================

// データソース
input string InpHistoryUrl = "https://raw.githubusercontent.com/takanitt-blip/gex-api/refs/heads/main/gex_history.json";

// 更新間隔（時間）
input int InpUpdateHours = 1;

// SPY → US500 換算比率
input double InpScaleFactor = 10.0;

// 表示する履歴日数
input int InpDisplayDays = 30;

// 線の色
input color InpColorCallWall  = clrRed;     // Call Wall（赤）
input color InpColorPutWall   = clrLime;    // Put Wall（緑）
input color InpColorZeroGamma = clrYellow;  // Zero Gamma（黄）

// オブジェクト名のプレフィックス（衝突回避）
const string OBJ_PREFIX = "GEXv1_";

// 監査20連動（2026-07-04）: データ不完全時のチャート上警告ラベル名。
// OBJ_PREFIX で始めることで DeleteAllOwnObjects() の一括削除対象に
// 含める（毎回の再描画サイクルで自動的にクリアされ、古い警告が
// 居座らない）。
const string OBJ_WARN_NAME = OBJ_PREFIX + "WARN_INCOMPLETE";


//================================================================
// イベントハンドラ
//================================================================

int OnInit()
{
    Print("[GEX Visualizer] 初期化開始");
    Print("[GEX Visualizer] URL: ", InpHistoryUrl);
    Print("[GEX Visualizer] Scale: ", InpScaleFactor);
    Print("[GEX Visualizer] 表示日数: ", InpDisplayDays);
    
    UpdateGEXLines();
    EventSetTimer(InpUpdateHours * 3600);
    
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    DeleteAllOwnObjects();
}

void OnTick() {}

void OnTimer()
{
    UpdateGEXLines();
}


//================================================================
// メイン処理：GitHub から JSON 取得 → 描画
//================================================================
void UpdateGEXLines()
{
    char post[], result[];
    string headers;
    
    int res = WebRequest("GET", InpHistoryUrl, NULL, NULL, 5000,
                         post, 0, result, headers);
    
    if(res != 200) {
        Print("[GEX Visualizer] HTTP取得エラー: code=", res,
              " GetLastError=", GetLastError());
        return;
    }
    
    string json = CharArrayToString(result);
    
    // 既存の自前オブジェクトをクリアしてから再描画
    DeleteAllOwnObjects();
    
    int drawn = ParseAndDraw(json);
    
    ChartRedraw();
    Print("[GEX Visualizer] 描画完了: ", drawn, " 日分");
}


//================================================================
// Step 1B: ET（America/New_York）DST 判定
//
// 米国連邦規則（2007年〜恒久）:
//   DST開始: 3月第2日曜 02:00 ET
//   DST終了: 11月第1日曜 02:00 ET
// 制度そのものなので、観測に基づく恣意的定数ではない。
//================================================================
datetime GetNthSundayUTC(int year, int month, int nth)
{
    MqlDateTime dt;
    dt.year = year; dt.mon = month; dt.day = 1;
    dt.hour = 0; dt.min = 0; dt.sec = 0;
    datetime first = StructToTime(dt);

    MqlDateTime firstDt;
    TimeToStruct(first, firstDt);
    // day_of_week: 0=日曜
    int daysToFirstSunday = (firstDt.day_of_week == 0) ? 0 : (7 - firstDt.day_of_week);
    int targetDay = 1 + daysToFirstSunday + (nth - 1) * 7;

    dt.day = targetDay;
    return StructToTime(dt);
}

// utcTime 時点でETがDST(EDT, UTC-4)かどうか
bool IsET_DST(datetime utcTime)
{
    MqlDateTime dt;
    TimeToStruct(utcTime, dt);
    int year = dt.year;

    // 3月第2日曜 02:00 ET(EST=UTC-5) = 07:00 UTC
    datetime dstStart = GetNthSundayUTC(year, 3, 2) + 7 * 3600;
    // 11月第1日曜 02:00 ET(EDT=UTC-4) = 06:00 UTC
    datetime dstEnd = GetNthSundayUTC(year, 11, 1) + 6 * 3600;

    return (utcTime >= dstStart && utcTime < dstEnd);
}

int GetET_OffsetHours(datetime utcTimeApprox)
{
    return IsET_DST(utcTimeApprox) ? -4 : -5;
}


//================================================================
// Step 1B: ETの暦日文字列("YYYY.MM.DD") → ET 00:00 に対応する UTC 時刻
//
// StringToTime() はタイムゾーン変換をしないため、まず EST(-5) 仮定で
// 概算UTCを出し、その概算時刻でDST判定をやり直して正しいオフセットを
// 確定する2段階処理（3月末/11月頭の数時間帯の誤判定を避けるため）。
//================================================================
datetime ETDateStrToUTC(string dateStr)
{
    datetime naive = StringToTime(dateStr + " 00:00:00");
    if(naive <= 0) return 0;

    datetime approxUTC = naive + 5 * 3600;
    int offset = GetET_OffsetHours(approxUTC);
    return naive - offset * 3600;
}

// UTC時刻 → 現在のサーバーオフセットでサーバー時間に変換。
// TimeGMT()とTimeTradeServer()の差は実行の都度取得するため、
// サーバー側DST切替にも自動追従（固定値ハードコードなし）。
datetime UTCToServerTime(datetime utcTime)
{
    int serverOffsetSec = (int)(TimeTradeServer() - TimeGMT());
    return utcTime + serverOffsetSec;
}


//================================================================
// Step 1C: JSON から指定キーの as_of 日付文字列を取得し、
// "YYYY.MM.DD" 形式に正規化する（ISO "2026-07-01T00:00:00" → "2026.07.01"）
//================================================================
string GetAsOfDateStr(string json, string dateKey)
{
    int datePos = StringFind(json, "\"" + dateKey + "\"");
    if(datePos < 0) return "";
    int blockEnd = StringFind(json, "}", datePos);
    if(blockEnd < 0) return "";

    int keyPos = StringFind(json, "\"as_of\":", datePos);
    if(keyPos < 0 || keyPos >= blockEnd) return "";

    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return "";
    int valStart = colonPos + 1;
    while(valStart < blockEnd &&
          (StringGetCharacter(json, valStart) == ' ' ||
           StringGetCharacter(json, valStart) == '"')) {
        valStart++;
    }

    int valEnd = StringFind(json, "\"", valStart);
    if(valEnd < 0 || valEnd > blockEnd) return "";

    string raw = StringSubstr(json, valStart, valEnd - valStart);
    int tPos = StringFind(raw, "T");
    string datePart = (tPos > 0) ? StringSubstr(raw, 0, tPos) : raw;
    StringReplace(datePart, "-", ".");
    return datePart;
}


//================================================================
// Step 1C: as_of（EOD確定日）+1日 ET00:00 を、サーバー時間で返す
// = この地図が「現実になった」直後、表示を開始すべき時刻
//
// entry(as_of=T) の表示区間 =
//     [ GetGovernStartServerTime(T), 次entryの GetGovernStartServerTime )
// 休場・週末は自動的にこの区間の内側に飲み込まれる（特別処理不要）。
//================================================================
datetime GetGovernStartServerTime(string json, string dateKey)
{
    string asOfStr = GetAsOfDateStr(json, dateKey);
    if(asOfStr == "") return 0;

    datetime utcAsOfMidnight = ETDateStrToUTC(asOfStr);
    if(utcAsOfMidnight <= 0) return 0;

    datetime utcGovernStart = utcAsOfMidnight + 86400;  // as_of の翌日 ET00:00
    return UTCToServerTime(utcGovernStart);
}


//================================================================
// JSON をパースして全日付を描画
//================================================================
int ParseAndDraw(string json)
{
    // 全日付を収集
    string dates[];
    int dateCount = CollectDates(json, dates);
    
    if(dateCount == 0) {
        Print("[GEX Visualizer] 警告: 日付データが見つかりません");
        return 0;
    }
    
    // 昇順ソート（YYYY.MM.DD は文字列ソートで日付順になる）
    SortStringArrayAscending(dates, dateCount);
    
    string oldest = dates[0];
    string latest = dates[dateCount - 1];
    Print("[GEX Visualizer] データ範囲: ", oldest, " 〜 ", latest,
          " (", dateCount, "日)");
    
    // 描画開始日のインデックスを決定
    int startIdx = 0;
    if(dateCount > InpDisplayDays) {
        startIdx = dateCount - InpDisplayDays;
    }
    
    int drawn = 0;

    // 監査20連動: DrawOneDay が「データ不完全」でスキップした日を集計する。
    // 特に「最新キー自体が不完全」は最優先で伝えるべき信号
    // （＝「今日の壁を信用して戦えるか」に直結する）ため区別して追跡する。
    int    incompleteCount     = 0;
    bool   latestIncomplete    = false;
    string latestIncompleteKey = "";

    for(int i = startIdx; i < dateCount; i++) {
        string dateKey = dates[i];
        bool isLatest = (i == dateCount - 1);

        // Step 1C: 表示区間は dateKey 自身ではなく as_of+1日 基準
        datetime segStart = GetGovernStartServerTime(json, dateKey);
        if(segStart <= 0) {
            Print("[GEX Visualizer] 警告: ", dateKey, " の as_of 解析失敗");
            incompleteCount++;
            if(isLatest) {
                latestIncomplete    = true;
                latestIncompleteKey = dateKey;
            }
            continue;
        }

        datetime segEnd;
        if(isLatest) {
            // 次のエントリがまだ無い＝新しいマップが来るまで未来方向に延長。
            // 次回更新時（新キー追加時）は全オブジェクトを消して引き直すので
            // 伸ばし過ぎても実害なし。
            segEnd = TimeTradeServer() + 5 * 86400;
        } else {
            // 次エントリの「支配開始時刻」の直前まで延長。
            // 休場日・週末を挟んでも、直前の壁が自動的に生き続ける。
            segEnd = GetGovernStartServerTime(json, dates[i + 1]) - 1;
            if(segEnd <= segStart) {
                segEnd = segStart + 86400 - 1;  // 異常時のみ安全側フォールバック
            }
        }

        if(DrawOneDay(json, dateKey, isLatest, segStart, segEnd)) {
            drawn++;
        } else {
            // DrawOneDay 内の call/put/zero <= 0 ガードでスキップされたケース
            // （監査20の fetch_failed プレースホルダ等、call_wall 欠落/null を含む）。
            incompleteCount++;
            if(isLatest) {
                latestIncomplete    = true;
                latestIncompleteKey = dateKey;
            }
        }
    }

    // 監査20連動: 不完全日が1件でもあればチャート上に警告を出す。
    // 0件の場合は何もしない（DeleteAllOwnObjects() で前回の警告表示は
    // 既にクリア済みのため、明示的な消去処理は不要）。
    DrawIncompleteWarning(latestIncomplete, incompleteCount, latestIncompleteKey);

    return drawn;
}


//================================================================
// 監査20連動（2026-07-04）: データ不完全時のチャート上警告表示
//
// 背景: DrawOneDay の call/put/zero <= 0 ガードは「壁を描かない」という
// 形で安全側に倒れるが、その事実は Print()（エキスパートログ）にしか
// 出ない。ログを常時見ているわけではないトレーダーにとっては、
// 「なぜ壁が消えたか」の説明がチャート上のどこにも無い状態になる。
// このガードが1回でも発動したら、チャート上にも一言出す。
//
// 重み付け:
//   最新キー自体が不完全 → 赤・最優先（「今日の地図が壊れている」＝
//     今日戦えるかに直結する最重要シグナル）
//   過去日のみ不完全     → オレンジ・参考情報（表示区間内の古い日の
//     欠落。現在の壁の信頼性には直接影響しない）
//================================================================
void DrawIncompleteWarning(bool latestIncomplete, int incompleteCount,
                            string latestIncompleteKey)
{
    if(incompleteCount == 0) return;

    string text;
    color  clr;
    // 記号は "\u26A0" 等のUnicodeエスケープを使わず素の ASCII に限定する。
    // MQL5コンパイラが \uXXXX 形式の文字列エスケープをサポートしている
    // という確証が無いため（未確認のAPI/言語仕様を明記されていると
    // 断定しない、というこのプロジェクトの方針に合わせる）。
    if(latestIncomplete) {
        text = StringFormat(
            "[!] GEX MAP INCOMPLETE (%s) - walls may be STALE. Check Experts log.",
            latestIncompleteKey);
        clr = clrRed;
    } else {
        text = StringFormat(
            "[!] %d historical GEX day(s) incomplete (older than latest). See Experts log.",
            incompleteCount);
        clr = clrOrange;
    }

    if(ObjectFind(0, OBJ_WARN_NAME) < 0) {
        ObjectCreate(0, OBJ_WARN_NAME, OBJ_LABEL, 0, 0, 0);
    }
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_XDISTANCE,  10);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_YDISTANCE,  20);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_COLOR,      clr);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_FONTSIZE,   10);
    ObjectSetString (0, OBJ_WARN_NAME, OBJPROP_FONT,       "Arial Bold");
    ObjectSetString (0, OBJ_WARN_NAME, OBJPROP_TEXT,       text);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_HIDDEN,     true);
    ObjectSetInteger(0, OBJ_WARN_NAME, OBJPROP_BACK,       false);
}


//================================================================
// JSON から全日付キーを収集
//================================================================
int CollectDates(string json, string &dates[])
{
    ArrayResize(dates, 100);
    int count = 0;
    int pos = 0;
    
    while(true) {
        int start = StringFind(json, "\"", pos);
        if(start < 0) break;
        
        int end = StringFind(json, "\"", start + 1);
        if(end < 0) break;
        
        string key = StringSubstr(json, start + 1, end - start - 1);
        
        if(IsValidDateFormat(key)) {
            if(count >= ArraySize(dates)) {
                ArrayResize(dates, count + 50);
            }
            dates[count] = key;
            count++;
        }
        
        pos = end + 1;
    }
    
    return count;
}


//================================================================
// 1 日分のデータを描画（Step 1C: t1/t2 は呼び出し側=ParseAndDraw が
// as_of 基準で算出済みのものを渡す）
//================================================================
bool DrawOneDay(string json, string dateKey, bool isLatest,
                 datetime t1, datetime t2)
{
    double call = GetValueFromJson(json, dateKey, "call_wall");
    double put  = GetValueFromJson(json, dateKey, "put_wall");
    double zero = GetValueFromJson(json, dateKey, "zero_gamma");
    
    if(call <= 0 || put <= 0 || zero <= 0) {
        Print("[GEX Visualizer] 警告: ", dateKey, " のデータが不完全");
        return false;
    }
    
    // 描画スタイル（最新日は強調）
    int            width    = isLatest ? 2 : 1;
    ENUM_LINE_STYLE solidStyle = isLatest ? STYLE_SOLID : STYLE_DOT;
    ENUM_LINE_STYLE dashStyle  = isLatest ? STYLE_DASH  : STYLE_DOT;
    
    // 価格を US500 スケールに変換
    double callPrice = call * InpScaleFactor;
    double putPrice  = put  * InpScaleFactor;
    double zeroPrice = zero * InpScaleFactor;
    
    // ラベル（最新日のみ）
    string callLabel = isLatest ? StringFormat("Call %.1f", callPrice) : "";
    string putLabel  = isLatest ? StringFormat("Put %.1f",  putPrice)  : "";
    string zeroLabel = isLatest ? StringFormat("Zero %.1f", zeroPrice) : "";
    
    DrawTrendLine(OBJ_PREFIX + "C_" + dateKey,
                  t1, callPrice, t2, callPrice,
                  InpColorCallWall, width, solidStyle, callLabel);
    
    DrawTrendLine(OBJ_PREFIX + "P_" + dateKey,
                  t1, putPrice, t2, putPrice,
                  InpColorPutWall, width, solidStyle, putLabel);
    
    DrawTrendLine(OBJ_PREFIX + "Z_" + dateKey,
                  t1, zeroPrice, t2, zeroPrice,
                  InpColorZeroGamma, 1, dashStyle, zeroLabel);
    
    return true;
}


//================================================================
// JSON から特定キーの数値を取得
// 日付ブロック内に検索範囲を限定し、ネスト誤読を防ぐ
//================================================================
double GetValueFromJson(string json, string dateKey, string key)
{
    // 日付ブロックの開始位置
    int datePos = StringFind(json, "\"" + dateKey + "\"");
    if(datePos < 0) return 0.0;
    
    // 日付ブロックの終了位置（最も近い "}" まで）
    int blockEnd = StringFind(json, "}", datePos);
    if(blockEnd < 0) return 0.0;
    
    // ブロック内でキーを検索
    int keyPos = StringFind(json, "\"" + key + "\":", datePos);
    if(keyPos < 0 || keyPos >= blockEnd) return 0.0;
    
    // 値の開始位置（コロンの直後、空白スキップ）
    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return 0.0;
    int valStart = colonPos + 1;
    while(valStart < blockEnd && StringGetCharacter(json, valStart) == ' ') {
        valStart++;
    }
    
    // 値の終了位置（カンマか "}"）
    int valEnd = StringFind(json, ",", valStart);
    if(valEnd < 0 || valEnd > blockEnd) valEnd = blockEnd;
    
    string valStr = StringSubstr(json, valStart, valEnd - valStart);
    StringTrimLeft(valStr);
    StringTrimRight(valStr);
    
    return StringToDouble(valStr);
}


//================================================================
// 日付フォーマット検証 "YYYY.MM.DD"
//================================================================
bool IsValidDateFormat(string s)
{
    if(StringLen(s) != 10) return false;
    if(StringGetCharacter(s, 4) != '.') return false;
    if(StringGetCharacter(s, 7) != '.') return false;
    return true;
}


// [Step 1C で廃止] 旧 GetDayRange(dateKey, t1, t2):
//   t1 = StringToTime(dateKey+" 00:00:00"), t2 = 同日 23:59:59 固定。
//   欠陥①: dateKey(ETの暦日)をサーバーローカル時刻とナイーブに解釈し、
//     引け直後（最も見たい時間帯）の直後に壁が消えることがあった。
//   欠陥②: 表示区間を暦日1日に固定していたため、休場日・週末を挟むと
//     次のエントリが来るまで壁が画面から消えていた（GEXが表す実体＝
//     ディーラーのOIは次の新規取引まで不変という物理的性質と矛盾）。
//   → GetGovernStartServerTime() + ParseAndDraw 内の区間計算に置換。


//================================================================
// 文字列配列の昇順ソート（少量データなのでバブルソート）
//================================================================
void SortStringArrayAscending(string &arr[], int count)
{
    for(int i = 0; i < count - 1; i++) {
        for(int j = 0; j < count - 1 - i; j++) {
            if(StringCompare(arr[j], arr[j+1]) > 0) {
                string tmp = arr[j];
                arr[j]   = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
}


//================================================================
// トレンドライン（線分）の描画
//================================================================
void DrawTrendLine(string name,
                   datetime t1, double p1,
                   datetime t2, double p2,
                   color clr, int width, ENUM_LINE_STYLE style,
                   string label)
{
    if(ObjectFind(0, name) < 0) {
        ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
    }
    
    ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
    ObjectSetDouble (0, name, OBJPROP_PRICE, 0, p1);
    ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
    ObjectSetDouble (0, name, OBJPROP_PRICE, 1, p2);
    
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
    ObjectSetInteger(0, name, OBJPROP_STYLE, style);
    
    // 線分を「両端で止める」（無限延長しない）
    ObjectSetInteger(0, name, OBJPROP_RAY_LEFT,  false);
    ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
    
    ObjectSetInteger(0, name, OBJPROP_BACK,       true);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
    
    if(StringLen(label) > 0) {
        ObjectSetString(0, name, OBJPROP_TEXT, label);
    }
}


//================================================================
// このEAが作成したオブジェクトをすべて削除
//================================================================
void DeleteAllOwnObjects()
{
    int total = ObjectsTotal(0);
    for(int i = total - 1; i >= 0; i--) {
        string name = ObjectName(0, i);
        if(StringFind(name, OBJ_PREFIX) == 0) {
            ObjectDelete(0, name);
        }
    }
}