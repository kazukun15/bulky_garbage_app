import streamlit as st
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict
import pandas as pd
from fpdf import FPDF

# ---------------------------
# 五十音順ソート用簡易変換関数（カタカナ→ヒラガナ）
# ---------------------------
def kana_to_hira(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if 0x30A0 <= ord(char) <= 0x30FF else char for char in text)

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
    # ここは実際の画像認識API呼び出しに置き換え可能です（現状は模擬的な遅延）
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
# CSVから品名・収集料金・直接搬入料金を読み込む関数（キャッシュ付き）
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
# PDF出力用関数（FPDF利用）
# ---------------------------
def generate_pdf(selected_items: List[Dict]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # ヘッダー
    pdf.cell(60, 10, "品名", border=1)
    pdf.cell(40, 10, "区分", border=1)
    pdf.cell(40, 10, "料金", border=1)
    pdf.ln()
    for item in selected_items:
        pdf.cell(60, 10, str(item['product']), border=1)
        type_str = "収集" if item.get("chosen_type") == "collection" else "直接搬入"
        pdf.cell(40, 10, type_str, border=1)
        pdf.cell(40, 10, str(item.get("chosen_price", "")), border=1)
        pdf.ln()
    return pdf.output(dest="S").encode("latin1")

# ---------------------------
# Streamlitアプリ本体（CSVのみ利用）
# ---------------------------
def main():
    st.set_page_config(page_title="粗大ごみ品目管理アプリ", layout="wide")
    st.title("粗大ごみ品目管理アプリ（CSV版）")

    # セッション変数の初期化（初回のみ設定）
    if "extracted_items" not in st.session_state:
        st.session_state.extracted_items = []
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []
    if "calc_done" not in st.session_state:
        st.session_state.calc_done = False
    if "total_price" not in st.session_state:
        st.session_state.total_price = 0

    # サイドバー：データリセットボタン
    st.sidebar.header("操作メニュー")
    if st.sidebar.button("データをリセットする"):
        st.session_state.extracted_items = []
        st.session_state.selected_items = []
        st.session_state.calc_done = False
        st.session_state.total_price = 0
        st.sidebar.success("データがリセットされました。")

    # CSV読み込み：CSVフォルダ内のファイルを処理する
    if not st.session_state.extracted_items:
        if st.sidebar.button("CSVフォルダ内のファイルを処理"):
            with st.spinner("CSVフォルダ内のファイルを処理中..."):
                st.session_state.extracted_items = process_csv_directory("CSV")
            st.sidebar.success("CSVフォルダからの読み込み完了")
    else:
        st.sidebar.write("既にCSVデータが読み込まれています。")

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
    # サイドバー：選択リスト、合計計算ボタン、PDF出力ボタン
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
            st.session_state.total_price = sum(item["chosen_price"] for item in st.session_state.selected_items)
            st.session_state.calc_done = True
        st.sidebar.success("合計計算完了")
    if st.session_state.get("calc_done", False):
        st.sidebar.write(f"合計品数: {len(st.session_state.selected_items)}")
        st.sidebar.write(f"合計金額: {st.session_state.total_price}円")
    else:
        st.sidebar.write("合計計算がまだ実行されていません。")
    if st.session_state.selected_items:
        pdf_data = generate_pdf(st.session_state.selected_items)
        st.sidebar.download_button("PDFで出力してダウンロード", data=pdf_data, file_name="selected_items.pdf", mime="application/pdf")

    # ---------------------------------
    # メインエリア：抽出された品目一覧（検索結果）
    # ---------------------------------
    st.subheader("抽出された品目一覧（収集料金・直接搬入料金）")
    if filtered_items:
        for idx, item in enumerate(filtered_items):
            cols = st.columns([3, 2, 2, 2, 2])
            cols[0].write(f"品名: **{item['product']}**")
            cols[1].write(f"収集: {item['collection_price']}円")
            cols[2].write(f"直接搬入: {item['direct_price']}円")
            selected_type = cols[3].selectbox("区分を選択", options=["収集", "直接搬入"], key=f"choice_{idx}")
            if cols[4].button("選択", key=f"select_{idx}"):
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
