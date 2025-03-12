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
# 五十音順ソート用簡易変換関数（カタカナ→ヒラガナ）
# ---------------------------
def kana_to_hira(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0x30A0 <= code <= 0x30FF:
            result.append(chr(code - 0x60))
        else:
            result.append(char)
    return "".join(result)

# ---------------------------
# 非同期API呼び出し（GeminiAPIと画像認識API）の定義
# ---------------------------
async def call_gemini_api(item_text: str) -> str:
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
    await asyncio.sleep(1)  # 模擬的な応答待ち
    return "画像認識API: 補正候補結果"

async def process_item_async(item: Dict) -> Dict:
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
                collection_price = int(str(row.get("収集料金", "")).replace("円", "").replace(",", "").strip())
            except Exception:
                collection_price = 0
            try:
                direct_price = int(str(row.get("直接搬入料金", "")).replace("円", "").replace(",", "").strip())
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
    parts = line.split()
    if len(parts) >= 3:
        product = parts[0]
        try:
            collection_price = int(parts[1].replace("円", "").replace(",", ""))
        except ValueError:
            collection_price = 0
        try:
            direct_price = int(parts[2].replace("円", "").replace(",", ""))
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
    st.set_page_config(page_title="粗大ごみ手数料一覧計算アプリ", layout="wide")
    st.title("粗大ごみ手数料一覧計算アプリ")

    # セッション変数の初期化
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []
    # 合計計算結果を保持する変数
    if "calc_done" not in st.session_state:
        st.session_state.calc_done = False
    if "total_price" not in st.session_state:
        st.session_state.total_price = 0

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

    # ---------------------------------
    # 検索バーと並び替えオプション
    # ---------------------------------
    st.subheader("検索・並び替えオプション")
    search_query = st.text_input("検索バー：品名を入力してください（部分一致）", "")
    sort_method = st.radio("並び順", ("なし", "五十音順"))

    filtered_items = []
    if st.session_state.extracted_items:
        if search_query:
            filtered_items = [
                item for item in st.session_state.extracted_items
                if search_query in item["product"]
            ]
        else:
            filtered_items = st.session_state.extracted_items.copy()

        if sort_method == "五十音順":
            filtered_items.sort(key=lambda x: kana_to_hira(x["product"]))
    else:
        st.info("まだ品目データが読み込まれていません。")

    # ---------------------------------
    # サイドバー：選択リストと合計計算ボタン
    # ---------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("選択リスト")
    if st.session_state.selected_items:
        for s_item in st.session_state.selected_items:
            st.sidebar.write(
                f"- {s_item['product']} ({'収集' if s_item['chosen_type'] == 'collection' else '直接搬入'}: {s_item['chosen_price']}円)"
            )
    else:
        st.sidebar.write("まだ品目が選択されていません。")
    if st.sidebar.button("合計を計算する"):
        with st.spinner("合計計算中..."):
            total_price = sum(item["chosen_price"] for item in st.session_state.selected_items)
            st.session_state.total_price = total_price
            st.session_state.calc_done = True
        st.sidebar.success("合計計算完了")
    if st.session_state.get("calc_done", False):
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {st.session_state.total_price}円")
    else:
        st.sidebar.write("合計計算がまだ実行されていません。")

    # ---------------------------------
    # メインエリア：抽出された品目一覧（検索結果）
    # ---------------------------------
    st.subheader("抽出された品目一覧（収集料金・直接搬入料金）")
    if filtered_items:
        for idx, item in enumerate(filtered_items):
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
        st.info("検索結果がありません。")

    # ---------------------------------
    # メインエリア：選択品目の詳細表示
    # ---------------------------------
    st.subheader("選択品目の詳細")
    for item in st.session_state.selected_items:
        st.write(f"**{item['product']}** - {'収集' if item['chosen_type'] == 'collection' else '直接搬入'}: {item['chosen_price']}円")
        st.write("GeminiAPI結果: ", item.get("gemini", ""))
        st.write("画像認識結果: ", item.get("image_recognition", ""))
        st.markdown("---")

if __name__ == "__main__":
    main()
