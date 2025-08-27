import os
from dotenv import load_dotenv

# 環境変数ファイルを読み込み
load_dotenv()

import json, re, requests
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, request, render_template, send_from_directory, url_for

# === OpenAI v1 クライアント ===
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("`pip install openai` を実行してください。")

client = OpenAI()  # OPENAI_API_KEY を環境変数から自動取得

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

# Function Calling から呼びやすい薄いラッパ
def save_pdf(text: str, title: str = "LLM出力", out_dir: str = "outputs_pdf") -> dict:
    return {"path": save_text_as_pdf(text, out_dir=out_dir, title=title)}

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
    payload = {
        "query": query,
        "search_depth": depth,       # "basic" or "advanced"
        "max_results": int(max_results),
        "include_answer": bool(include_answer),
        "include_images": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

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
    for step in range(max_steps):
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        m = res.choices[0].message

        # ツール不要なら最終回答
        if not getattr(m, "tool_calls", None):
            return m.content, pdf_path

        # 要求されたツールを全部こなして結果を返す
        messages.append(m.model_dump())  # LLMの"ツール使います"発言も履歴に残す
        for call in m.tool_calls:
            out = execute_function(call)
            # save_pdf の結果からパスを捕捉
            if call.function.name == "save_pdf" and isinstance(out, dict) and "path" in out:
                pdf_path = out["path"]

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(out, ensure_ascii=False)
                }
            )
    return "(ステップ上限に達しました)", pdf_path

# === Flask ===
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/display", methods=["POST"])
def display():
    topic = request.form.get("query")
    departure = request.form.get("departure", "").strip()
    stay_days = int(request.form.get("stay_days", "1"))
    travel_date = request.form.get("travel_date")
    depth = request.form.get("depth", "basic")
    include_answer = request.form.get("include_answer") == "on"
    max_results = int(request.form.get("max_results", "3") or 3)

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
{travel_description}について以下を段階的に詳しく調べ、必要に応じてWeb検索ツールを使い、最後にPDFとして保存してください。

旅行条件:
- 目的地: {topic}
- 出発地: {departure if departure else "目的地の中心地から開始"}
- 期間: {date_range}（{stay_days}日間）
- 旅行タイプ: {travel_type}

調査項目:
・{date_range}の天気予報を調べる
・その期間の近隣のイベントや特別な催しを調べる
・天気に適した観光地やアクティビティを探す
・その期間のトレンドやおすすめスポットを調べる
{f"・{departure}から{topic}までの交通手段と所要時間を調べる" if departure else "・{topic}周辺の移動手段と所要時間を調べる"}
・宿泊施設の情報を調べる（{stay_days}日間の場合）
・所要時間や移動時間を考慮した{stay_days}日間の詳細なスケジュールを作成

探索のヒント:
- まずアウトラインを作り、次に不足点を補うために検索を行い、最終的にPDF保存ツール(save_pdf)を呼び出してください。
- 日付に特化した情報（イベント、天気、営業時間など）を重視してください。
- {stay_days}日間の旅程を考慮した、実現可能なスケジュールを作成してください。
- どこの観光地に行くのかを明確かつ具体的にスケジュールを作成してください。
- 移動時間も明記したスケジュールにしてください。
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
    result_text, pdf_path = run_agent(messages, function_descriptions)

    pdf_url = None
    if pdf_path and Path(pdf_path).exists():
        # ダウンロードリンクを作る
        # /download/<filename> で配信
        pdf_url = url_for("download_file", filename=Path(pdf_path).name)

    return render_template("display.html", result_text=result_text, pdf_url=pdf_url)

@app.route("/download/<path:filename>")
def download_file(filename):
    directory = Path("outputs_pdf").resolve()
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    # Cloud9向け: 0.0.0.0:8080
    app.run(debug=True, host="0.0.0.0", port=8080)
