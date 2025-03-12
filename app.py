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
import pandas as pd

# ---------------------------
# 非同期API呼び出し（GeminiAPIと画像認識API）の定義
# ---------------------------
async def call_gemini_api(item_text: str) -> str:
    """
    Gemini API（models/gemini-2.0-flash）を実際に呼び出して、
    品目のテキストから補正・予測結果を取得します。
    Streamlitのsecretsに設定されたGEMINI_API_KEYを利用します。
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
    画像認識APIの呼び出し実装例です（現時点ではダミー実装）。
    実際のエンドポイントがある場合はここを実装してください。
    """
    await asyncio.sleep(1)  # 模擬的な応答待ち
    return "画像認識API: 補正候補結果"

async def process_item_async(item: Dict) -> Dict:
    """
    選択された品目に対して、GeminiAPIと画像認識APIを並列に呼び出し、
    結果をitemに追加して返します。
    """
    gemini_result, image_result = await asyncio.gather(
        call_gemini_api(item["product"]),
        call_image_recognition_api()
    )
    item["gemini"] = gemini_result
    item["image_recognition"] = image_result
    return item

# ---------------------------
# CSVから品名・収集料金・直接搬入料金を読み込む関数
# ---------------------------
@st.cache_data(show_spinner=False)
def process_csv_directory(csv_dir: str = "CSV") -> List[Dict]:
    """
    指定されたCSVフォルダ内のすべてのCSVファイルを読み込み、
    各行から「品名」「収集料金」「直接搬入料金」を抽出してリスト形式の辞書データに変換して返します。
    CSVのヘッダーは「品名」「収集料金」「直接搬入料金」であることを前提とします。
    """
    all_items = []
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        st.error(f"CSVディレクトリ {csv_dir} が存在しません。")
        return all_items
    for csv_file in csv_path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df.columns = [col.strip() for col in df.columns]
        for _, row in df.iterrows():
            product = str(row.get("品名", "")).strip()
            try:
                collection_price = int(str(row.get("収集料金", "")).replace("円", "").strip())
            except Exception:
                collection_price = 0
            try:
                direct_price = int(str(row.get("直接搬入料金", "")).replace("円", "").strip())
            except Exception:
                direct_price = 0
            all_items.append({
                "product": product,
                "collection_price": collection_price,
                "direct_price": direct_price
            })
    return all_items

# ---------------------------
# PDFから品名・収集料金・直接搬入料金を抽出する関数
# ---------------------------
def parse_line_as_item(line: str) -> Dict:
    """
    1行のテキストから「品名 収集料金 直接搬入料金」を抽出し、
    辞書形式で返します。
    例: 'ソファ 520円 260円' → {product: 'ソファ', collection_price: 520, direct_price: 260}
    """
    parts = line.split()
    if len(parts) >= 3:
        product = parts[0]
        try:
            collection_price = int(parts[1].replace("円", ""))
        except ValueError:
            collection_price = 0
        try:
            direct_price = int(parts[2].replace("円", ""))
        except ValueError:
            direct_price = 0
        return {
            "product": product,
            "collection_price": collection_price,
            "direct_price": direct_price
        }
    return {}

@st.cache_data(show_spinner=False)
def extract_pdf_data(pdf_path: str) -> List[Dict]:
    """
    指定されたPDFファイルから、まずpdfplumberでテキスト抽出を試み、
    失敗またはテキストが取得できなかった場合はOCR（Tesseract + pdf2image）で抽出します。
    各行は parse_line_as_item() に渡し、品名・収集料金・直接搬入料金を取り出します。
    """
    extracted_items = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        item_dict = parse_line_as_item(line)
                        if item_dict:
                            extracted_items.append(item_dict)
    except Exception as e:
        st.error(f"pdfplumberでの抽出に失敗しました: {e}")

    if not extracted_items:
        st.info("テキスト抽出ができなかったため、OCR処理を実施します。")
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                ocr_text = pytesseract.image_to_string(image, lang="jpn")
                lines = ocr_text.split('\n')
                for line in lines:
                    item_dict = parse_line_as_item(line)
                    if item_dict:
                        extracted_items.append(item_dict)
        except Exception as e:
            st.error(f"OCRによる抽出に失敗しました: {e}")
    return extracted_items

@st.cache_data(show_spinner=False)
def process_pdf_directory(pdf_dir: str = "PDF") -> List[Dict]:
    """
    指定されたPDFフォルダ内のすべてのPDFファイルを処理し、
    全品目リストを返します。
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
    st.title("粗大ごみ品目管理アプリ（収集・直接搬入の料金区別版）")

    # セッション変数の初期化
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []

    # サイドバー：入力ファイル形式の選択
    st.sidebar.header("操作メニュー")
    file_format = st.sidebar.radio("入力ファイル形式の選択", ("PDF", "CSV"))

    if file_format == "PDF":
        pdf_source = st.sidebar.radio("PDF入力元の選択", ("アップロード", "PDFフォルダから読み込み"))
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
            if st.sidebar.button("PDFフォルダ内のファイルを処理"):
                with st.spinner("PDFフォルダ内のファイルを処理中..."):
                    extracted = process_pdf_directory("PDF")
                    st.session_state.extracted_items = extracted
                st.sidebar.success("PDFフォルダからの抽出完了")
    else:
        if st.sidebar.button("CSVフォルダ内のファイルを処理"):
            with st.spinner("CSVフォルダ内のファイルを処理中..."):
                extracted = process_csv_directory("CSV")
                st.session_state.extracted_items = extracted
            st.sidebar.success("CSVフォルダからの読み込み完了")

    # サイドバー：選択リスト表示
    st.sidebar.markdown("---")
    st.sidebar.header("選択リスト")
    if st.session_state.selected_items:
        total_price = sum(item["chosen_price"] for item in st.session_state.selected_items)
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {total_price}円")
        for s_item in st.session_state.selected_items:
            st.sidebar.write(
                f"- {s_item['product']} ({'収集' if s_item['chosen_type'] == 'collection' else '直接搬入'}: {s_item['chosen_price']}円)"
            )
    else:
        st.sidebar.write("まだ品目が選択されていません。")

    # メインエリア：抽出された品目一覧（収集料金・直接搬入料金）
    st.subheader("抽出された品目一覧（収集料金・直接搬入料金）")
    if st.session_state.extracted_items:
        for idx, item in enumerate(st.session_state.extracted_items):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            with col1:
                st.write(f"品名: **{item['product']}**")
            with col2:
                st.write(f"収集: {item['collection_price']}円")
            with col3:
                st.write(f"直接搬入: {item['direct_price']}円")
            with col4:
                choice_key = f"choice_{idx}"
                selected_type = st.selectbox("区分を選択", options=["収集", "直接搬入"], key=choice_key)
            with col5:
                if st.button("選択", key=f"select_{idx}"):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    processed_item = loop.run_until_complete(process_item_async(item.copy()))
                    loop.close()
                    if selected_type == "収集":
                        processed_item["chosen_price"] = processed_item["collection_price"]
                        processed_item["chosen_type"] = "collection"
                    else:
                        processed_item["chosen_price"] = processed_item["direct_price"]
                        processed_item["chosen_type"] = "direct"
                    st.session_state.selected_items.append(processed_item)
                    st.success(f"{processed_item['product']} を選択リストに追加しました ({selected_type}: {processed_item['chosen_price']}円)")
    else:
        st.info("抽出された品目はありません。ファイルをアップロードするか、フォルダを処理してください。")

    # メインエリア：選択品目の詳細表示
    st.subheader("選択品目の詳細")
    for item in st.session_state.selected_items:
        st.write(f"**{item['product']}** - {'収集' if item['chosen_type'] == 'collection' else '直接搬入'}: {item['chosen_price']}円")
        st.write("GeminiAPI結果: ", item.get("gemini", ""))
        st.write("画像認識結果: ", item.get("image_recognition", ""))
        st.markdown("---")

if __name__ == "__main__":
    main()
