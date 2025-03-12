import streamlit as st
import asyncio
import aiohttp
import tempfile
import pdfplumber  # テキスト抽出用（PDFがテキスト形式の場合）
from pdf2image import convert_from_path  # PDFを画像に変換
import pytesseract  # Tesseract OCRを利用
from typing import List, Dict

# ---------------------------
# 非同期API呼び出し（GeminiAPI）のサンプル関数
# ---------------------------
async def call_gemini_api(item: str) -> str:
    # StreamlitのsecretからAPIキーを取得
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": item}]}]}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                return f"Gemini APIエラー: {resp.status}"
            response_json = await resp.json()
            # APIレスポンスの形式に合わせて出力を取得（例）
            candidate = response_json.get("candidates", [{}])[0].get("output", "出力がありません")
            return candidate

async def call_image_recognition_api(image: bytes = None) -> str:
    # ダミー実装：実際は画像データを送信して認識結果を取得する処理を実装
    await asyncio.sleep(1)
    return "画像認識API: 補正候補結果"

async def process_item_async(item: Dict) -> Dict:
    # GeminiAPIと画像認識APIを並列実行
    gemini_result, image_result = await asyncio.gather(
        call_gemini_api(item["product"]),
        call_image_recognition_api()
    )
    item["gemini"] = gemini_result
    item["image_recognition"] = image_result
    return item

# ---------------------------
# PDFから品目と価格を抽出する関数（高精度OCR対応）
# ---------------------------
def extract_pdf_data(pdf_path: str) -> List[Dict]:
    extracted_items = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            product = parts[0]
                            try:
                                price = int(parts[1].replace("円", ""))
                            except ValueError:
                                price = 0
                            extracted_items.append({"product": product, "price": price})
    except Exception as e:
        st.error("pdfplumberでの抽出に失敗しました。")
    
    # pdfplumberでテキストが抽出できなかった場合は、OCRで画像から読み取る
    if not extracted_items:
        st.info("テキスト抽出ができなかったため、OCRによる画像認識を実施します。")
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                # 日本語の場合はlang='jpn'を指定
                text = pytesseract.image_to_string(image, lang='jpn')
                lines = text.split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        product = parts[0]
                        try:
                            price = int(parts[1].replace("円", ""))
                        except ValueError:
                            price = 0
                        extracted_items.append({"product": product, "price": price})
        except Exception as e:
            st.error("OCRによる抽出に失敗しました。環境設定をご確認ください。")
    return extracted_items

# ---------------------------
# Streamlitアプリ本体
# ---------------------------
def main():
    st.set_page_config(page_title="粗大ごみ品目管理アプリ", layout="wide")
    st.title("粗大ごみ品目の抽出・選択・合計計算アプリ")

    # サイドバー：PDFアップロード
    st.sidebar.header("PDFアップロード")
    uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロードしてください", type="pdf")

    # セッション内で抽出結果・選択リストを管理
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []

    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        st.sidebar.success("PDFアップロード完了")
        with st.spinner("PDFからデータを抽出中..."):
            extracted = extract_pdf_data(tmp_pdf_path)
            st.session_state.extracted_items = extracted
        st.success("抽出完了！")

    # メインエリア：抽出結果一覧表示
    st.subheader("抽出された品目一覧")
    if st.session_state.extracted_items:
        for idx, item in enumerate(st.session_state.extracted_items):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"品名: **{item['product']}**")
            with col2:
                st.write(f"価格: {item['price']}円")
            with col3:
                if st.button("選択", key=f"select_{idx}"):
                    # 非同期処理でAPI呼び出しを実施
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    processed_item = loop.run_until_complete(process_item_async(item.copy()))
                    loop.close()
                    st.session_state.selected_items.append(processed_item)
                    st.success(f"{item['product']} を選択リストに追加しました")

    # サイドバー：選択リストと合計表示
    st.sidebar.header("選択リスト")
    if st.session_state.selected_items:
        total_price = sum(item["price"] for item in st.session_state.selected_items)
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {total_price}円")
        st.sidebar.markdown("---")
        for s_item in st.session_state.selected_items:
            st.sidebar.write(f"- {s_item['product']} ({s_item['price']}円)")
    else:
        st.sidebar.write("まだ品目が選択されていません。")

    # 詳細表示：選択品目のAPI処理結果等を表示
    st.subheader("選択品目の詳細")
    for item in st.session_state.selected_items:
        st.write(f"**{item['product']}** - {item['price']}円")
        st.write(item.get("gemini", ""))
        st.write(item.get("image_recognition", ""))
        st.markdown("---")

if __name__ == "__main__":
    main()
