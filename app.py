import os
from dotenv import load_dotenv

# 環境変数ファイルを読み込み
load_dotenv()

import json, re, requests
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta, timezone

from flask import Flask, request, render_template, send_from_directory, url_for

# === OpenAI v1 クライアント ===
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("`pip install openai` を実行してください。")

client = OpenAI()  # OPENAI_API_KEY を環境変数から自動取得

# === Unsplash API クライアント ===
def get_unsplash_photo(query, count=1):
    """Unsplash APIを使用して観光地の写真を取得する"""
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("警告: UNSPLASH_ACCESS_KEY が設定されていません")
        return []
    
    try:
        print(f"Unsplash API で写真を検索中: '{query}' (最大{count}枚)")
        url = "https://api.unsplash.com/search/photos"
        headers = {
            "Authorization": f"Client-ID {access_key}"
        }
        params = {
            "query": query,
            "per_page": count,
            "orientation": "landscape",
            "content_filter": "high"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        photos = []
        
        for photo in data.get("results", []):
            photos.append({
                "id": photo["id"],
                "url": photo["urls"]["regular"],
                "thumb": photo["urls"]["thumb"],
                "alt": photo.get("alt_description", query),
                "photographer": photo["user"]["name"],
                "photographer_url": photo["user"]["links"]["html"]
            })
        
        print(f"'{query}' の写真を {len(photos)} 枚取得しました")
        return photos
    except Exception as e:
        print(f"Unsplash API エラー ({query}): {str(e)}")
        return []

def get_destination_photos(destination, attractions_list):
    """旅行プランに含まれる観光地から写真を取得する（最大3枚）"""
    all_photos = {}
    total_photos = 0
    max_photos = 3
    
    # 目的地の写真を取得（最大2枚）
    destination_photos = get_unsplash_photo(destination, 2)
    if destination_photos:
        all_photos[destination] = destination_photos
        total_photos += len(destination_photos)
    
    # 旅行プランに含まれる観光地から写真を取得（残りの枚数分）
    for attraction in attractions_list:
        # 出発地は除外し、目的地と異なる観光地のみ処理
        if (attraction and 
            attraction != destination and 
            len(attraction) > 2 and 
            total_photos < max_photos and
            # 出発地のキーワードを含まない観光地のみ
            not any(departure_keyword in attraction for departure_keyword in ['駅', '空港', '港', 'バス停'])):
            
            # 観光地名から写真を取得
            photos = get_unsplash_photo(attraction, 1)
            if photos:
                all_photos[attraction] = photos
                total_photos += 1
                if total_photos >= max_photos:
                    break
    
    return all_photos

# === PDF生成 (ReportLab) ===
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 日本語フォント（CID、追加DL不要）
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))


def _sanitize_filename(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'[\\/:*?"<>|#%&{}$!`\'@+=]', "_", s)
    return (s[:maxlen] + "…") if len(s) > maxlen else s

def save_text_as_pdf(text: str, out_dir: str = "outputs_pdf", title: str = "LLM出力") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    head = _sanitize_filename((text or "output").splitlines()[0])
    filename = f"{datetime.now():%Y%m%d-%H%M%S}_{head}.pdf"
    out_path = str(Path(out_dir) / filename)

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "JPTitle", parent=styles["Title"],
        fontName="HeiseiKakuGo-W5", fontSize=16, leading=20, spaceAfter=12
    )
    body_style = ParagraphStyle(
        "JPBody", parent=styles["Normal"],
        fontName="HeiseiKakuGo-W5", fontSize=11, leading=16
    )

    story = [Paragraph(title, title_style), Spacer(1, 8)]
    for para in (text or "").split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    return out_path

def save_text_as_summary(text: str, out_dir: str = "outputs_summary", title: str = "LLM出力") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    head = _sanitize_filename((text or "output").splitlines()[0])
    filename = f"{datetime.now():%Y%m%d-%H%M%S}_{head}.txt"
    out_path = str(Path(out_dir) / filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


# Function Calling から呼びやすい薄いラッパ
def save_pdf(text: str, title: str = "LLM出力", out_dir: str = "outputs_pdf") -> dict:
    return {"path": save_text_as_pdf(text, out_dir=out_dir, title=title)}

# === 要約ツール ===
def save_summary(text: str, title: str = "LLM出力", out_dir: str = "outputs_summary") -> dict:
    return {"path": save_text_as_summary(text, out_dir=out_dir, title=title)}

# === Tavily 検索 ===
def call_tavily_search(query, depth="basic", max_results=3, include_answer=False):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 未設定")

    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # パラメータの検証と正規化
    if not query or len(query.strip()) < 3:
        raise ValueError("検索クエリは3文字以上である必要があります")
    
    # 検索深度の検証
    if depth not in ["basic", "advanced"]:
        depth = "basic"
    
    # 最大結果数の制限
    max_results = max(1, min(int(max_results), 10))
    
    payload = {
        "query": query.strip(),
        "search_depth": depth,
        "max_results": max_results,
        "include_answer": bool(include_answer),
        "include_images": False,
        "include_raw_content": False,
        "include_domains": [],
        "exclude_domains": [],
        "search_type": "search"  # 明示的に検索タイプを指定
    }
    
    try:
        print(f"Tavily API リクエスト: {json.dumps(payload, ensure_ascii=False)}")
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if resp.status_code != 200:
            print(f"Tavily API レスポンス: {resp.status_code} - {resp.text}")
        
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        error_detail = f"HTTP {e.response.status_code}: {e.response.text}"
        print(f"Tavily API エラー: {error_detail}")
        
        # 特定のエラーコードに対する詳細な説明
        if e.response.status_code == 432:
            raise RuntimeError(f"Tavily API エラー (432): リクエストパラメータが無効です。詳細: {error_detail}")
        elif e.response.status_code == 401:
            raise RuntimeError("Tavily API エラー (401): APIキーが無効です。APIキーを確認してください。")
        elif e.response.status_code == 429:
            raise RuntimeError("Tavily API エラー (429): レート制限に達しました。しばらく待ってから再試行してください。")
        elif e.response.status_code == 500:
            raise RuntimeError("Tavily API エラー (500): サーバー内部エラーが発生しました。しばらく待ってから再試行してください。")
        else:
            raise RuntimeError(f"Tavily API エラー: {error_detail}")
    except requests.exceptions.RequestException as e:
        print(f"Tavily API リクエストエラー: {str(e)}")
        raise RuntimeError(f"Tavily API リクエストエラー: {str(e)}")

def get_google_maps_route(departure, destination, mode="driving", departure_time=None):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_MAPS_API_KEY が設定されていません"}

    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": departure,
        "destination": destination,
        "mode": mode,
        "language": "ja",
        "key": api_key
    }

    if mode == "transit" and departure_time:
        params["departure_time"] = departure_time  # タイムスタンプで指定

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        if status != "OK":
            # ZERO_RESULTS は通常のエラーにせず、'error'で返す
            if status == "ZERO_RESULTS":
                return {"error": f"指定された出発地({departure})と目的地({destination})の間に経路が見つかりませんでした"}
                print("Google Maps API status:", status, "error_message:", data.get("error_message"))
            else:
                return {"error": f"Google Maps API エラー: {status} - {data.get('error_message')}"}

        route = data["routes"][0]["legs"][0]
        duration_seconds = route["duration"]["value"]
        steps = []
        for step in route["steps"]:
            steps.append({
                "instruction": step["html_instructions"],
                "distance": step["distance"]["text"],
                "duration": step["duration"]["text"],
                "travel_mode": step["travel_mode"]
            })

        return {
            "start_address": route["start_address"],
            "end_address": route["end_address"],
            "distance": route["distance"]["text"],
            "duration": route["duration"]["text"],
            "duration_seconds": duration_seconds,
            "steps": steps
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Google Maps API リクエストエラー: {str(e)}"}


# === ツール定義 ===
function_descriptions = [
    {
        "type": "function",
        "function": {
            "name": "call_tavily_search",
            "description": (
                "Tavily のWeb検索APIで外部情報を取得する。"
                "最新情報・公式情報・比較記事などが必要なときに使用。"
                "検索クエリは日本語でよい。結果(JSON)には要約(answer)・検索結果(results)が含まれる場合がある。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索したい自然文クエリ（例: '国内の生成AIの最新動向 2025 上半期'）",
                        "minLength": 3
                    },
                    "depth": {
                        "type": "string",
                        "description": "探索の深さ。basic=高速/簡易、advanced=網羅/高精度",
                        "enum": ["basic", "advanced"],
                        "default": "basic"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "取得する検索結果件数（小さく始めることを推奨）",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Tavilyの自動要約(answer)を含めるか",
                        "default": False
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_summary",
            "description": "与えられたテキストを要約として保存し、保存先パスを返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":  {"type": "string", "description": "要約として保存する本文（日本語可）"},
                    "title": {"type": "string", "description": "要約のタイトル", "default": "旅行プラン要約"},
                    "out_dir": {"type": "string", "description": "保存先ディレクトリ", "default": "outputs_summary"}
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_pdf",
            "description": "与えられたテキストをPDFに保存し、保存先パスを返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":  {"type": "string", "description": "PDFに保存する本文（日本語可）"},
                    "title": {"type": "string", "description": "PDFのタイトル", "default": "LLM出力"},
                    "out_dir": {"type": "string", "description": "保存先ディレクトリ", "default": "outputs_pdf"}
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }
    }
]

def execute_function(call):
    name = call.function.name
    args = json.loads(call.function.arguments or "{}")

    if name == "call_tavily_search":
        q = args.get("query")
        if not q:
            return {"error": "query is required"}
        return {
            "data": call_tavily_search(
                q,
                args.get("depth", "basic"),
                args.get("max_results", 3),
                args.get("include_answer", False),
            )
        }
    if name == "save_summary":
        return save_summary(
            text=args.get("text", ""),
            title=args.get("title", "旅行プラン要約"),
            out_dir=args.get("out_dir", "outputs_summary"),
        )
    if name == "save_pdf":
        return save_pdf(
            text=args.get("text", ""),
            title=args.get("title", "LLM出力"),
            out_dir=args.get("out_dir", "outputs_pdf"),
        )
    return {"error": f"unknown tool {name}"}

def run_agent(messages, tools, max_steps=5):
    """LLMに考えさせ→必要ならツールを実行→結果を渡して続き、を最大N回繰り返す。"""
    pdf_path = None
    summary_path = None
    for step in range(max_steps):
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        m = res.choices[0].message

        # ツール不要なら最終回答
        if not getattr(m, "tool_calls", None):
            return m.content, pdf_path, summary_path

        # 要求されたツールを全部こなして結果を返す
        messages.append(m.model_dump())  # LLMの"ツール使います"発言も履歴に残す
        for call in m.tool_calls:
            out = execute_function(call)
            # save_pdf の結果からパスを捕捉
            if call.function.name == "save_pdf" and isinstance(out, dict) and "path" in out:
                pdf_path = out["path"]
            # save_summary の結果からパスを捕捉
            elif call.function.name == "save_summary" and isinstance(out, dict) and "path" in out:
                summary_path = out["path"]

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(out, ensure_ascii=False)
                }
            )
    return "(ステップ上限に達しました)", pdf_path, summary_path

# === Flask ===
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/display", methods=["POST", "GET"])
def display():
    maps_info = None 
    topic = request.form.get("query")
    departure = request.form.get("departure", "").strip()
    stay_days = int(request.form.get("stay_days", "1"))
    travel_date = request.form.get("travel_date")
    depth = request.form.get("depth", "basic")
    include_answer = request.form.get("include_answer") == "on"
    max_results = int(request.form.get("max_results", "3") or 3)
    # フォームから交通手段を取得
    travel_mode = request.form.get("travel_mode", "driving")  # デフォルトは車
    travel_time = request.form.get("departure_time", "09:00")  # HH:MM形式

    arrival_time_str = None

    if maps_info and "duration_seconds" in maps_info:
        jst = timezone(timedelta(hours=9))
        travel_datetime = datetime.strptime(travel_date + " " + travel_time, "%Y-%m-%d %H:%M").replace(tzinfo=jst)
        arrival_datetime = travel_datetime + timedelta(seconds=maps_info["duration_seconds"])
        arrival_time_str = arrival_datetime.strftime("%Y-%m-%d %H:%M")

    maps_text = None
    maps_embed_url = None
    if departure:
        maps_info = None
        maps_text = None
        maps_embed_url = None
        arrival_time_str = None  

        if travel_mode == "transit" and departure:
            # 公共交通機関は経路テキストのみ
            jst = timezone(timedelta(hours=9))
            travel_datetime = datetime.strptime(travel_date + " " + travel_time, "%Y-%m-%d %H:%M")
            travel_datetime = travel_datetime.replace(tzinfo=jst)
            departure_timestamp = int(travel_datetime.timestamp())
            maps_info = get_google_maps_route(departure, topic, mode="transit", departure_time=departure_timestamp)
        else:
            # 徒歩・車はマップ表示
            maps_info = get_google_maps_route(departure, topic, mode=travel_mode)

        if maps_info and "error" not in maps_info:
            # 経路概要のみ
            maps_text = (
                f"出発地: {maps_info['start_address']}\n"
                f"目的地: {maps_info['end_address']}\n"
                f"総距離: {maps_info['distance']}\n"
                f"所要時間: {maps_info['duration']}\n"
            )


        # 徒歩・車のみ埋め込みマップURL生成
            if travel_mode in ["driving", "walking"]:
                origin = urllib.parse.quote(departure)
                destination = urllib.parse.quote(topic)
                maps_embed_url = (
                    f"https://www.google.com/maps/embed/v1/directions"
                    f"?key={os.getenv('GOOGLE_MAPS_API_KEY')}"
                    f"&origin={origin}&destination={destination}&mode={travel_mode}"
                )
        else:
            maps_text = maps_info.get("error") if maps_info else "経路情報が取得できません"



    if not topic:
        return render_template("display.html", result_text="目的地を入力してください。", pdf_url=None)
    
    if not travel_date:
        return render_template("display.html", result_text="旅行開始日を入力してください。", pdf_url=None)

    # 日付を日本語形式に変換
    try:
        date_obj = datetime.strptime(travel_date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%Y年%m月%d日")
        day_of_week = ["月", "火", "水", "木", "金", "土", "日"][date_obj.weekday()]
        formatted_date_with_day = f"{formatted_date}({day_of_week})"
        
        # 宿泊日数に応じて終了日を計算
        if stay_days > 1:
            end_date_obj = date_obj + timedelta(days=stay_days-1)
            end_date = end_date_obj.strftime("%Y年%m月%d日")
            end_day_of_week = ["月", "火", "水", "木", "金", "土", "日"][end_date_obj.weekday()]
            formatted_end_date = f"{end_date}({end_day_of_week})"
            date_range = f"{formatted_date_with_day}〜{formatted_end_date}"
        else:
            date_range = formatted_date_with_day
            
    except ValueError:
        date_range = travel_date

    # 旅行タイプを判定
    if departure:
        travel_type = "出発地から目的地への旅行"
        travel_description = f"{departure}から{topic}への{stay_days}日間の旅行プラン"
    else:
        travel_type = "目的地中心の旅行"
        travel_description = f"{topic}での{stay_days}日間の旅行プラン"

    # プロンプト（出発地・宿泊日数を考慮した旅行プラン作成）
    system_msg = {"role": "system", "content": "あなたは旅行のプロです。指定された条件（出発地、目的地、宿泊日数、日付）に基づいて、実用的で詳細な旅行プランを作成してください。所要時間や移動時間、宿泊施設なども考慮してください。"}
    
    user_msg = {
         "role": "user",
         "content": f"""
{travel_description}について以下を段階的に詳しく調べ、必要に応じてWeb検索ツールを使い、最後に要約ツール(save_summary)を呼び出してください。

旅行条件:
- 目的地: {topic}
- 出発地: {departure if departure else "目的地の中心地から開始"}
- 出発時刻: {travel_time}
- 到着予定時刻: {arrival_time_str if arrival_time_str else "不明"}
- 期間: {date_range}（{stay_days}日間）
- 旅行タイプ: {travel_type}


スケジュール作成時は、到着予定時刻以降に観光開始できるようにしてください。

調査項目:
・{date_range}の天気予報を調べる
・その期間の近隣のイベントや特別な催しを調べる
・天気に適した観光地やアクティビティを探す
・その期間のトレンドやおすすめスポットを調べる
{f"・{departure}から{topic}までの交通手段と所要時間を調べる" if departure else "・{topic}周辺の移動手段と所要時間を調べる"}
・宿泊施設の情報を調べる（{stay_days}日間の場合）
・所要時間や移動時間を考慮した{stay_days}日間の詳細なスケジュールを作成 
・スケジュールにある観光地の写真を3枚分調べる

探索のヒント:
- まずアウトラインを作り、次に不足点を補うために検索を行い、最終的に要約ツール(save_summary)を呼び出してください。
- 日付に特化した情報（イベント、天気、営業時間など）を重視してください。
- {stay_days}日間の旅程を考慮した、実現可能なスケジュールを作成してください。
- どこの観光地に行くのかを明確かつ具体的にスケジュールを作成してください。
- 移動時間も明記したスケジュールにしてください。
- 最終的な回答は人間が読みやすい形で作成してください。
- 人間が読みやすい形で、以下の点に注意して作成してください：
  * 各日のスケジュールは時間順に整理
  * 移動時間と所要時間を明確に記載
  * 観光地の詳細情報（営業時間、入場料、見どころ）を含める
  * 食事や休憩時間も適切に配置
  * 天候に応じた代替プランも提案
  * 予算の目安も含める
  * 注意事項や持ち物も具体的に記載
 """
     }

    # Tavilyの使い方をLLMに伝えるため、1度ツールの説明も埋め込んでおく（任意）
    tool_hint = {
        "role": "user",
        "content": (
            f"検索は call_tavily_search を使えます。depth='{depth}', max_results={max_results}, "
            f"include_answer={'true' if include_answer else 'false'} を推奨します。"
        ),
    }

    messages = [system_msg, user_msg, tool_hint]
    result_text, _, summary_path = run_agent(messages, function_descriptions)

    # PDFは自動生成せず、displayページでのみ生成可能にする
    pdf_url = None

    # 観光地の写真を取得
    photos_data = {}
    try:
        # 旅行プランから観光地名を抽出
        attractions_from_plan = []
        
        # 結果テキストから観光地名を抽出
        lines = result_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            # 観光地のキーワードを含む行を検出
            if any(keyword in line for keyword in ['寺', '神社', '公園', '美術館', '博物館', '城', '通り', '市場', 
                'タワー', 'ビル', 'センター', 'プラザ', 'モール', '広場', '橋', '川',
                '山', '海', '湖', '温泉', 'レストラン', 'カフェ', 'ショップ', '観光', '名所']):
                
                # 行から観光地名を抽出（キーワードの前後の文字列を取得）
                for keyword in ['寺', '神社', '公園', '美術館', '博物館', '城', '通り', '市場', 
                    'タワー', 'ビル', 'センター', 'プラザ', 'モール', '広場', '橋', '川',
                    '山', '海', '湖', '温泉', 'レストラン', 'カフェ', 'ショップ']:
                    if keyword in line:
                        # キーワードの前後の文字列を抽出
                        parts = line.split(keyword)
                        for part in parts:
                            part = part.strip()
                            if part and len(part) > 2 and part not in attractions_from_plan:
                                # 一般的でない文字を除去
                                clean_part = re.sub(r'[【】「」『』()（）\[\]{}]', '', part)
                                if clean_part and len(clean_part) > 2:
                                    attractions_from_plan.append(clean_part)
                        break
        
        # 目的地を最初に追加（出発地は除外）
        attractions_list = [topic]
        attractions_list.extend(attractions_from_plan)
        
        # 重複を除去
        attractions_list = list(dict.fromkeys(attractions_list))
        
        # デバッグ情報を出力
        print(f"旅行プランから検出された観光地: {attractions_list}")
        
        # 写真を取得（出発地は除外）
        photos_data = get_destination_photos(topic, attractions_list)
        
        # 写真取得結果を出力
        print(f"取得された写真数: {len(photos_data)}")
        for location, photos in photos_data.items():
            print(f"  - {location}: {len(photos)}枚")
        
        # 写真が3枚未満の場合は、目的地の写真を追加して3枚にする
        total_photos = sum(len(photos) for photos in photos_data.values())
        if total_photos < 3 and topic in photos_data:
            # 目的地から追加の写真を取得
            additional_photos = get_unsplash_photo(topic, 3 - total_photos)
            if additional_photos:
                photos_data[topic].extend(additional_photos)
                print(f"目的地の写真を追加: {len(additional_photos)}枚")
        
    except Exception as e:
        print(f"写真取得エラー: {str(e)}")
        photos_data = {}
    
    # 検索結果から要約データを抽出
    structured_data = extract_structured_data(result_text, topic, date_range, stay_days, departure)
    
    # travel_infoとstructured_dataを定義
    travel_info = {
        "departure": departure if departure else None,
        "date_range": date_range,
        "stay_days": stay_days,
        "travel_type": travel_type
    }

    return render_template("display.html", 
                         result_text=result_text, 
                         pdf_url=pdf_url,
                         photos_data=photos_data,
                         travel_info=travel_info,
                         destination=topic,
                         structured_data=structured_data)

@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    """displayページからPDF生成リクエストを受け取る"""
    data = request.get_json()
    if not data or 'text' not in data:
        return {"error": "テキストが提供されていません"}, 400
    
    try:
        # PDFを生成
        pdf_path = save_text_as_pdf(
            text=data['text'],
            title=data.get('title', '旅行プラン'),
            out_dir="outputs_pdf"
        )
        
        # ダウンロード用のURLを生成
        pdf_url = url_for("download_file", filename=Path(pdf_path).name)
        
        return {"success": True, "pdf_url": pdf_url, "filename": Path(pdf_path).name}
    except Exception as e:
        return {"error": f"PDF生成エラー: {str(e)}"}, 500

    travel_info = {
        "destination": topic,
        "departure": departure,
        "stay_days": stay_days,
        "date_range": date_range,
        "travel_type": travel_type,
    }    

    structured_data = extract_structured_data(result_text, topic, date_range, stay_days, departure)


    return render_template(
        "display.html",
        result_text=result_text,
        pdf_url=pdf_url,
        maps_text=maps_text,           # 追加
        maps_embed_url=maps_embed_url,  # 追加
        travel_info=travel_info,           # ←追加
        structured_data=structured_data    # ←追加
        )


@app.route("/download/<path:filename>")
def download_file(filename):
    directory = Path("outputs_pdf").resolve()
    return send_from_directory(directory, filename, as_attachment=True)

def extract_structured_data(text, destination, date_range, stay_days, departure):
    """テキストから構造化データを抽出する"""
    try:
        # 基本的な情報を抽出
        data = {
            "summary": f"{destination}での{stay_days}日間の旅行プラン",
            "weather": extract_weather_info(text),
            "attractions": extract_attractions_info(text),
            "trends": extract_trends_info(text),
            "schedule": [],
            "transportation": extract_transportation_info(text, departure, destination),
            "accommodation": extract_accommodation_info(text, stay_days),
            "tips": extract_tips_info(text),
            "sections": []  # セクション別の内容を追加
        }
        
        # 日付ごとのスケジュールを抽出
        data["schedule"] = extract_detailed_schedule(text, destination, date_range, stay_days)
        
        return data
    except Exception as e:
        # エラーが発生した場合は基本的なデータを返す
        return {
            "summary": f"{destination}での{stay_days}日間の旅行プラン",
            "weather": "天気情報",
            "attractions": "観光地・アクティビティ情報",
            "trends": "トレンド・おすすめスポット情報",
            "schedule": [{
                "day": 1,
                "date": date_range,
                "activities": [{
                    "time": "詳細は本文をご確認ください",
                    "location": destination,
                    "activity": "旅行プラン",
                    "details": "作成された旅行プラン"
                }]
            }],
            "transportation": "交通手段",
            "accommodation": "宿泊情報" if stay_days > 1 else "日帰り旅行",
            "tips": "旅行のコツ",
            "sections": []
        }

def extract_weather_info(text):
    """天気情報を抽出"""
    weather_keywords = ["天気", "気候", "予報", "気温", "雨", "晴れ", "曇り", "雪", "風"]
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line for keyword in weather_keywords):
            if len(line.strip()) > 5:  # 短すぎる行は除外
                return line.strip()
    
    return "天気情報が含まれています"

def extract_attractions_info(text):
    """観光地・アクティビティ情報を抽出"""
    attraction_keywords = ["観光地", "名所", "寺", "神社", "公園", "美術館", "博物館", "城", "温泉", "レストラン", "カフェ", "ショップ", "アクティビティ", "体験"]
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line for keyword in attraction_keywords):
            if len(line.strip()) > 5:
                return line.strip()
    
    return "観光地・アクティビティ情報が含まれています"

def extract_trends_info(text):
    """トレンド・おすすめスポット情報を抽出"""
    trend_keywords = ["トレンド", "おすすめ", "人気", "最新", "話題", "注目", "流行", "ベスト", "ランキング"]
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line for keyword in trend_keywords):
            if len(line.strip()) > 5:
                return line.strip()
    
    return "トレンド・おすすめスポット情報が含まれています"

def extract_transportation_info(text, departure, destination):
    """交通手段情報を抽出"""
    if departure:
        transport_keywords = ["電車", "バス", "車", "徒歩", "所要時間", "駅", "路線"]
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in transport_keywords):
                if len(line.strip()) > 5:
                    return line.strip()
        
        return f"{departure}から{destination}までの交通手段の情報が含まれています"
    else:
        return f"{destination}周辺の移動手段の情報が含まれています"

def extract_accommodation_info(text, stay_days):
    """宿泊情報を抽出"""
    if stay_days <= 1:
        return "日帰り旅行のため宿泊情報は不要です"
    
    accommodation_keywords = ["ホテル", "旅館", "宿", "チェックイン", "チェックアウト", "宿泊"]
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line for keyword in accommodation_keywords):
            if len(line.strip()) > 5:
                return line.strip()
    
    return "宿泊情報が含まれています"

def extract_tips_info(text):
    """旅行のコツを抽出"""
    tips_keywords = ["コツ", "アドバイス", "注意", "気をつける", "準備", "持ち物", "おすすめ"]
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line for keyword in tips_keywords):
            if len(line.strip()) > 5:
                return line.strip()
    
    return "旅行のコツが含まれています"

def extract_detailed_schedule(text, destination, date_range, stay_days):
    """詳細なスケジュールを抽出"""
    schedule = []
    lines = text.split('\n')
    current_day = 1
    current_activities = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 日目の区切りを検出
        if any(keyword in line for keyword in ['日目', '1日目', '2日目', '3日目', '4日目', '5日目', '6日目', '7日目']):
            if current_activities:
                schedule.append({
                    "day": current_day,
                    "date": date_range,
                    "activities": current_activities
                })
            current_day += 1
            current_activities = []
            continue
        
        # 時間情報を含む行を活動として抽出
        if any(keyword in line for keyword in ['時', '分', 'AM', 'PM', '午前', '午後', '〜', 'から']):
            details = extract_activity_details(line)
            
            current_activities.append({
                "time": line,
                "location": destination,
                "activity": line,
                "details": details
            })
        # 観光地やスポット名が含まれる行も抽出
        elif any(keyword in line for keyword in ['寺', '神社', '公園', '美術館', '博物館', '城', '駅', '通り', '市場', 
            'タワー', 'ビル', 'センター', 'プラザ', 'モール', '広場', '橋', '川',
            '山', '海', '湖', '温泉', 'レストラン', 'カフェ', 'ショップ']):
            if line and len(line) > 3:
                current_activities.append({
                    "time": "時間未定",
                    "location": line,
                    "activity": line,
                    "details": extract_activity_details(line)
                })
    
    # 最後の日のスケジュールを追加
    if current_activities:
        schedule.append({
            "day": current_day,
            "date": date_range,
            "activities": current_activities
        })
    
    # スケジュールが空の場合は基本的なスケジュールを作成
    if not schedule:
        schedule = create_default_schedule(destination, date_range, stay_days)
    
    return schedule

def extract_activity_details(line):
    """活動の詳細情報を抽出"""
    # 営業時間、入場料、見どころなどの情報を抽出
    detail_keywords = ["営業時間", "入場料", "見どころ", "所要時間", "休憩", "食事", "移動"]
    
    for keyword in detail_keywords:
        if keyword in line:
            return f"{keyword}の情報が含まれています"
    
    return line

def create_default_schedule(destination, date_range, stay_days):
    """デフォルトのスケジュールを作成"""
    schedule = []
    
    for day in range(1, stay_days + 1):
        schedule.append({
            "day": day,
            "date": date_range,
            "activities": [{
                "time": "詳細は本文をご確認ください",
                "location": destination,
                "activity": f"{destination}での観光",
                "details": f"{day}日目の観光予定"
            }]
        })
    
    return schedule

if __name__ == "__main__":
    # 環境変数の確認
    print("=== 環境変数確認 ===")
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    
    if tavily_key:
        print(f"✓ TAVILY_API_KEY: {tavily_key[:10]}...{tavily_key[-4:] if len(tavily_key) > 14 else '短すぎる'}")
        if len(tavily_key) < 20:
            print("⚠️  警告: TAVILY_API_KEYが短すぎます")
    else:
        print("❌ TAVILY_API_KEY が設定されていません")
    
    if openai_key:
        print(f"✓ OPENAI_API_KEY: {openai_key[:10]}...{openai_key[-4:] if len(openai_key) > 14 else '短すぎる'}")
    else:
        print("❌ OPENAI_API_KEY が設定されていません")
    
    if unsplash_key:
        print(f"✓ UNSPLASH_ACCESS_KEY: {unsplash_key[:10]}...{unsplash_key[-4:] if len(unsplash_key) > 14 else '短すぎる'}")
    else:
        print("❌ UNSPLASH_ACCESS_KEY が設定されていません")
    
    print("==================")
    
    # Cloud9向け: 0.0.0.0:8080
    app.run(debug=True, host="0.0.0.0", port=8080)