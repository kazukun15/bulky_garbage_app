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

# 五十音順ソート用に、jaconvなどの外部ライブラリを使わずに
# Python標準だけで簡易的に「カタカナ → ヒラガナ」変換を行う関数を定義します。
# ※正確性を高めたい場合は、jaconvやpykakasiなどを使うのがおすすめです。
def kana_to_hira(text: str) -> str:
    """
    カタカナをヒラガナに変換して返す簡易実装。
    それ以外の文字はそのまま返す。
    """
    result = []
    for char in text:
        code = ord(char)
        # カタカナ(全角)の範囲: U+30A0～U+30FF
        if 0x30A0 <= code <= 0x30FF:
            # カタカナ → ヒラガナ (U+60)だけコードをずらす
            result.append(chr(code - 0x60))
        else:
            result.append(char)
    return "".join(result)

# ---------------------------
# 非同期API呼び出し（GeminiAPIと画像認識API）
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
    # ダミー実装：実際の画像認識APIがある場合はここを実装
    await asyncio.sleep(1)
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
# CSV処理
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
        # もし「大項目」列があれば、同様に row.get("大項目", "") で取得可能
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
                # "category": str(row.get("大項目", "")).strip()  # 大項目の列があれば
            })
    return all_items

# ---------------------------
# PDF処理
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
    st.set_page_config(page_title="粗大ごみ品目管理アプリ", layout="wide")
    st.title("粗大ごみ品目一覧")

    # セッション変数の初期化
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []

    # サイドバー：入力ファイル形式
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
    # 検索バーとソートオプション
    # ---------------------------------
    st.subheader("検索・並び替えオプション")
    search_query = st.text_input("検索バー：品名を入力してください（部分一致）", "")
    sort_method = st.radio("並び順", ("なし", "五十音順"))

    # 検索結果をフィルタ
    filtered_items = []
    if st.session_state.extracted_items:
        if search_query:
            filtered_items = [
                item for item in st.session_state.extracted_items
                if search_query in item["product"]
            ]
        else:
            filtered_items = st.session_state.extracted_items.copy()

        # ソート処理（五十音順）
        if sort_method == "五十音順":
            # カタカナ→ヒラガナ変換した文字列をキーにソート
            filtered_items.sort(key=lambda x: kana_to_hira(x["product"]))
    else:
        st.info("まだ品目データが読み込まれていません。")

    # サイドバー：選択リスト表示
    st.sidebar.markdown("---")
    st.sidebar.header("選択リスト")
    if st.session_state.selected_items:
        total_price = sum(item["chosen_price"] for item in st.session_state.selected_items)
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {total_price}円")
        for s_item in st.session_state.selected_items:
            st.sidebar.write(
                f"- {s_item['product']} "
                f"({'収集' if s_item['chosen_type'] == 'collection' else '直接搬入'}: {s_item['chosen_price']}円)"
            )
    else:
        st.sidebar.write("まだ品目が選択されていません。")

    # メインエリア：抽出された品目一覧（検索結果を表示）
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
                    st.success(
                        f"{processed_item['product']} を選択リストに追加しました "
                        f"({selected_type}: {processed_item['chosen_price']}円)"
                    )
    else:
        st.info("検索結果がありません。")

    # メインエリア：選択品目の詳細表示
    st.subheader("選択品目の詳細")
    for item in st.session_state.selected_items:
        st.write(
            f"**{item['product']}** - "
            f"{'収集' if item['chosen_type'] == 'collection' else '直接搬入'}: {item['chosen_price']}円"
        )
        st.write("GeminiAPI結果: ", item.get("gemini", ""))
        st.write("画像認識結果: ", item.get("image_recognition", ""))
        st.markdown("---")

if __name__ == "__main__":
    main()
