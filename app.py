"""Streamlit interactive dashboard for BTH04 requirement 6.

Run from project root: streamlit run app.py
"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "iris_cleaned.csv"
REPORT = ROOT / "reports" / "bao_cao_khoa_hoc_iris.pdf"
METRICS = ROOT / "reports" / "ket_qua_phan_tich.json"
FEATURES = {
    "Chiều dài đài hoa (cm)": "sepal_length_cm",
    "Chiều rộng đài hoa (cm)": "sepal_width_cm",
    "Chiều dài cánh hoa (cm)": "petal_length_cm",
    "Chiều rộng cánh hoa (cm)": "petal_width_cm",
}
SPECIES = ["Setosa", "Versicolor", "Virginica"]
COLORS = {"Setosa": "#0B8F97", "Versicolor": "#4C6FFF", "Virginica": "#E69550"}

st.set_page_config(
    page_title="BTH04 | Iris AI Data Dashboard", page_icon="🌿", layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.block-container {padding-top: 2.15rem; padding-bottom: 3rem;}
h1,h2,h3 {color: #153357;}
[data-testid="stMetric"] {background: #eff5fb; padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError("Chưa có dữ liệu sạch. Hãy chạy python src/run_pipeline.py trước.")
    return pd.read_csv(DATA)


try:
    df = get_data()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

st.caption("BTH04 · YÊU CẦU 6 · MINI-PROJECT AI + PYTHON")
st.title("🌿 Iris Data Explorer")
st.write("Dashboard tương tác: chọn **loài hoa**, **biến đo** và **kiểu biểu đồ** để khám phá 150 mẫu thực tế từ UCI.")

with st.sidebar:
    st.header("Bộ lọc dữ liệu")
    selected = st.multiselect("Chọn loài hoa", options=SPECIES, default=SPECIES)
    graph = st.radio("Kiểu biểu đồ", ["Biểu đồ phân tán", "Biểu đồ hộp", "Biểu đồ phân phối"], index=0)
    y_name = st.selectbox("Biến cần quan sát", list(FEATURES), index=2)
    x_name = st.selectbox("Trục hoành (biểu đồ phân tán)", list(FEATURES), index=3)
    st.divider()
    st.markdown("**Nguồn dữ liệu**: Fisher (1936), UCI ML Repository")
    st.markdown("[Trang nguồn và giấy phép CC BY 4.0](https://archive.ics.uci.edu/dataset/53/iris)")
    st.caption("Mã nguồn dự án: MIT. Số liệu trong báo cáo PDF được tạo từ toàn bộ dữ liệu, không phải riêng bộ lọc hiện tại.")

filtered = df[df["species"].isin(selected)].copy()
all_stats = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
accuracy = all_stats.get("model", {}).get("accuracy_test")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Mẫu đang xem", f"{len(filtered)}/{len(df)}")
m2.metric("Số loài", filtered["species"].nunique())
m3.metric("Số biến đo", 4)
m4.metric("Accuracy mô hình", f"{accuracy:.2%}" if accuracy is not None else "N/A")

if filtered.empty:
    st.warning("Bạn chưa chọn loài hoa nào. Hãy chọn ít nhất một loài trong thanh bên trái.")
    st.stop()

x_col, y_col = FEATURES[x_name], FEATURES[y_name]
if graph == "Biểu đồ phân tán":
    fig = px.scatter(filtered, x=x_col, y=y_col, color="species",
                     hover_data=list(FEATURES.values()), opacity=.86,
                     color_discrete_map=COLORS,
                     category_orders={"species": SPECIES},
                     labels={x_col:x_name,y_col:y_name,"species":"Loài hoa"},
                     title=f"{x_name} so với {y_name}")
    fig.update_traces(marker=dict(size=10,line=dict(width=.5,color="white")))
elif graph == "Biểu đồ hộp":
    fig = px.box(filtered, x="species",y=y_col,color="species",points="all",
                 category_orders={"species":SPECIES},color_discrete_map=COLORS,
                 labels={y_col:y_name,"species":"Loài hoa"},
                 title=f"Phân phối {y_name} theo loài")
else:
    fig = px.histogram(filtered,x=y_col,color="species",nbins=20,
                       barmode="overlay",opacity=.68,
                       color_discrete_map=COLORS,
                       labels={y_col:y_name,"species":"Loài hoa"},
                       title=f"Tần suất {y_name} theo loài")
fig.update_layout(template="plotly_white",height=480,legend_title_text="Loài hoa",
                  title=dict(font=dict(size=19,color="#153357")),
                  margin=dict(l=25,r=25,t=65,b=25))
st.plotly_chart(fig,use_container_width=True)

left,right=st.columns([1.25,1])
with left:
    st.subheader("Thống kê mô tả theo loài")
    grouped=filtered.groupby("species")[[y_col]].agg(["count","mean","std","min","max"]).round(2)
    grouped.columns=["Số mẫu","Trung bình","Độ lệch chuẩn","Nhỏ nhất","Lớn nhất"]
    st.dataframe(grouped,use_container_width=True)
with right:
    st.subheader("Tải dữ liệu và báo cáo")
    st.download_button("Tải CSV đang lọc",data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="iris_filtered.csv",mime="text/csv",use_container_width=True)
    if REPORT.exists():
        st.download_button("Tải báo cáo khoa học PDF (toàn bộ)",data=REPORT.read_bytes(),
                           file_name="bao_cao_khoa_hoc_iris.pdf",mime="application/pdf",use_container_width=True)
    else:
        st.info("Chạy `python src/run_pipeline.py` để sinh báo cáo PDF.")

with st.expander("Xem các bản ghi dữ liệu",expanded=False):
    st.dataframe(filtered, use_container_width=True, height=290)
st.caption("Dữ liệu: UCI Iris, 150 mẫu. Mô hình Logistic Regression là minh họa học thuật, không phải dịch vụ nhận dạng ngoài thực tế.")
