import streamlit as st
import asyncio
import aiohttp
import tempfile
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict

# ---------------------------
# 非同期API呼び出し（GeminiAPIと画像認識API）の定義
# ---------------------------
async def call_gemini_api(item_text: str) -> str:
    """
    Gemini API（models/gemini-2.0-flash）に対して、品目のテキストをもとに曖昧なものの補正・予測を実行する。
    Streamlitのsecretsに設定されたGEMINI_API_KEYを利用。
    """
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": item_text}]}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                return f"Gemini APIエラー: {resp.status}"
            response_json = await resp.json()
            candidate = response_json.get("candidates", [{}])[0].get("output", "出力なし")
            return candidate

async def call_image_recognition_api(image: bytes = None) -> str:
    """
    画像認識APIのダミー実装。実際は画像データを送信し、曖昧な品目の認識・補正結果を返す。
    """
    await asyncio.sleep(1)  # 模擬的なAPI応答待ち
    return "画像認識API: 補正候補結果"

async def process_item_async(item: Dict) -> Dict:
    """
    選択された品目に対して、GeminiAPIと画像認識APIを並列に呼び出し、結果をitemに追加する。
    """
    gemini_result, image_result = await asyncio.gather(
        call_gemini_api(item["product"]),
        call_image_recognition_api()
    )
    item["gemini"] = gemini_result
    item["image_recognition"] = image_result
    return item

# ---------------------------
# PDFから品名と価格を抽出する関数
# ---------------------------
def extract_pdf_data(pdf_path: str) -> List[Dict]:
    """
    指定したPDFファイルから、まずpdfplumberでテキスト抽出を試み、失敗またはテキストが取得できなかった場合はOCR（Tesseract + pdf2image）を利用する。
    テキストは「品名 価格円」（例："ソファ 5000円"）というフォーマットを想定。
    """
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
        st.error(f"pdfplumberでの抽出に失敗しました: {e}")

    # テキスト抽出できなかった場合、または抽出結果が空の場合はOCR処理
    if not extracted_items:
        st.info("テキスト抽出ができなかったため、OCR処理を実施します。")
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                ocr_text = pytesseract.image_to_string(image, lang="jpn")
                lines = ocr_text.split('\n')
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
            st.error(f"OCRによる抽出に失敗しました: {e}")
    return extracted_items

def process_pdf_directory(pdf_dir: str = "PDF") -> List[Dict]:
    """
    指定されたディレクトリ内のすべてのPDFファイルを処理し、全品目リストを返す。
    """
    all_items = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        st.error(f"ディレクトリ {pdf_dir} が存在しません。")
        return all_items
    for pdf_file in pdf_path.glob("*.pdf"):
        items = extract_pdf_data(str(pdf_file))
        all_items.extend(items)
    return all_items

# ---------------------------
# Streamlitアプリ本体
# ---------------------------
def main():
    st.set_page_config(page_title="粗大ごみ品目管理アプリ", layout="wide")
    st.title("粗大ごみ品目管理アプリ")
    
    # セッション変数の初期化
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []
    
    # サイドバー：操作メニュー
    st.sidebar.header("操作メニュー")
    pdf_source = st.sidebar.radio("PDF入力元の選択", ("アップロード", "PDFフォルダから読み込み"))
    
    # PDFアップロードの場合
    if pdf_source == "アップロード":
        uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pdf_path = tmp_file.name
            with st.spinner("PDFからデータを抽出中..."):
                extracted = extract_pdf_data(tmp_pdf_path)
                st.session_state.extracted_items = extracted
            st.sidebar.success("PDFアップロードおよび抽出完了")
    else:
        # PDFフォルダから処理する場合（プロジェクトルートのPDFフォルダ内のPDFを対象）
        if st.sidebar.button("PDFフォルダ内のファイルを処理"):
            with st.spinner("PDFフォルダ内のファイルを処理中..."):
                extracted = process_pdf_directory("PDF")
                st.session_state.extracted_items = extracted
            st.sidebar.success("PDFフォルダからの抽出完了")
    
    # サイドバー：選択リスト表示
    st.sidebar.markdown("---")
    st.sidebar.header("選択リスト")
    if st.session_state.selected_items:
        total_price = sum(item["price"] for item in st.session_state.selected_items)
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {total_price}円")
        for s_item in st.session_state.selected_items:
            st.sidebar.write(f"- {s_item['product']} ({s_item['price']}円)")
    else:
        st.sidebar.write("まだ品目が選択されていません。")
    
    # メインエリア：抽出された品目一覧
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
                    # 非同期処理を用いてAPI呼び出しを実施し、結果を追加する
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    processed_item = loop.run_until_complete(process_item_async(item.copy()))
                    loop.close()
                    st.session_state.selected_items.append(processed_item)
                    st.success(f"{item['product']} を選択リストに追加しました")
    else:
        st.info("抽出された品目はありません。PDFをアップロードするか、PDFフォルダを処理してください。")
    
    # メインエリア：選択品目の詳細表示
    st.subheader("選択品目の詳細")
    for item in st.session_state.selected_items:
        st.write(f"**{item['product']}** - {item['price']}円")
        st.write("GeminiAPI結果: ", item.get("gemini", ""))
        st.write("画像認識結果: ", item.get("image_recognition", ""))
        st.markdown("---")
    
if __name__ == "__main__":
    main()
