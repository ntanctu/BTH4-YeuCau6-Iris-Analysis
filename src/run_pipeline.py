"""End-to-end Iris analysis: data cleaning -> EDA -> classification -> PDF.

Run from any working directory: python src/run_pipeline.py
The PDF and figures are completely rebuilt from the raw CSV, not hardcoded.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, PageBreak, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "iris_uci_original.csv"
CLEAN = ROOT / "data" / "iris_cleaned.csv"
REPORT = ROOT / "reports" / "bao_cao_khoa_hoc_iris.pdf"
STATS = ROOT / "reports" / "ket_qua_phan_tich.json"
ASSETS = ROOT / "assets"
COLUMNS = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]
PALETTE = {"Setosa": "#0B8F97", "Versicolor": "#4C6FFF", "Virginica": "#E69550"}
NAVY = colors.HexColor("#132A4A")
BLUE = colors.HexColor("#2F64B6")
CYAN = colors.HexColor("#13A3AE")
GRAY = colors.HexColor("#52627B")
LIGHT = colors.HexColor("#EFF5FB")


def cleaning() -> tuple[pd.DataFrame, dict]:
    if not RAW.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW}. Run src/prepare_data.py first.")
    initial = pd.read_csv(RAW, encoding="utf-8")
    if not set(COLUMNS + ["species"]).issubset(initial.columns):
        raise ValueError("Input CSV is missing a required column.")
    raw = initial[COLUMNS + ["species"]].copy()
    before = len(raw)
    for col in COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["species"] = (raw["species"].astype("string").str.strip()
                      .str.replace("Iris-", "", regex=False).str.capitalize())
    mask = raw[COLUMNS].gt(0).all(axis=1) & raw["species"].isin(SPECIES)
    missing_rows = int(raw.isna().any(axis=1).sum())
    invalid_rows = int((~mask & ~raw.isna().any(axis=1)).sum())
    raw = raw.dropna(subset=COLUMNS + ["species"])
    raw = raw.loc[raw[COLUMNS].gt(0).all(axis=1) & raw["species"].isin(SPECIES)].copy()
    # Identical observations can be independent samples; never drop without provenance.
    duplicates_retained = int(raw.duplicated().sum())
    raw.to_csv(CLEAN, index=False, encoding="utf-8")
    info = {
        "raw_rows": before, "clean_rows": len(raw),
        "missing_rows": missing_rows, "invalid_rows": invalid_rows,
        "duplicates_retained": duplicates_retained,
        "classes": {k: int(v) for k, v in raw["species"].value_counts().reindex(SPECIES).items()},
    }
    return raw, info


def train_model(df: pd.DataFrame) -> dict:
    X, y = df[COLUMNS], df["species"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("logreg", LogisticRegression(max_iter=2000, random_state=42))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, Xtr, ytr, cv=folds, scoring="accuracy")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    matrix = confusion_matrix(yte, pred, labels=SPECIES)
    cls = classification_report(yte, pred, labels=SPECIES, output_dict=True, zero_division=0)
    return {
        "method": "StandardScaler + LogisticRegression",
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
        "accuracy_test": float(accuracy_score(yte, pred)),
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std(ddof=1)),
        "cm": matrix.tolist(),
        "macro_f1": float(cls["macro avg"]["f1-score"]),
        "weighted_f1": float(cls["weighted avg"]["f1-score"]),
        "classification_report": cls,
    }


def set_chart_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.94)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10,
        "savefig.dpi": 170, "figure.facecolor": "white",
    })


def chart_boxplot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    sns.boxplot(data=df, x="species", y="petal_length_cm", hue="species",
                legend=False, order=SPECIES, palette=PALETTE, ax=ax,
                width=0.62, linewidth=1.0)
    ax.set(title="Distribution of petal length by species", xlabel="Species", ylabel="Petal length (cm)")
    sns.despine(ax=ax)
    fig.tight_layout()
    p = ASSETS / "01_boxplot_petal.png"; fig.savefig(p, bbox_inches="tight");plt.close(fig);return p


def chart_scatter(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    sns.scatterplot(data=df, x="petal_length_cm", y="petal_width_cm", hue="species",
                    hue_order=SPECIES, palette=PALETTE, s=55, alpha=.88, ax=ax)
    ax.set(title="Petal dimensions: clear separation for Setosa", xlabel="Petal length (cm)", ylabel="Petal width (cm)")
    ax.legend(title="Species", frameon=False, loc="lower right", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    p=ASSETS / "02_scatter_petal.png";fig.savefig(p,bbox_inches="tight");plt.close(fig);return p


def chart_corr(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    corr = df[COLUMNS].corr().rename(index={
        "sepal_length_cm": "Sepal length", "sepal_width_cm": "Sepal width",
        "petal_length_cm": "Petal length", "petal_width_cm": "Petal width",
    }, columns={
        "sepal_length_cm": "Sepal length", "sepal_width_cm": "Sepal width",
        "petal_length_cm": "Petal length", "petal_width_cm": "Petal width",
    })
    sns.heatmap(corr, annot=True, cmap="BrBG", center=0, vmin=-1, vmax=1,
                fmt=".2f", square=True, linewidths=.6, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Pearson correlation among four measurements")
    fig.tight_layout()
    p=ASSETS / "03_correlation.png";fig.savefig(p,bbox_inches="tight");plt.close(fig);return p


def chart_confusion(model: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    sns.heatmap(np.array(model["cm"]), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=SPECIES, yticklabels=SPECIES,
                linewidths=1, linecolor="white", ax=ax)
    ax.set(title="Confusion matrix on 45 held-out samples", xlabel="Predicted species", ylabel="Actual species")
    fig.tight_layout()
    p=ASSETS / "04_confusion.png";fig.savefig(p,bbox_inches="tight");plt.close(fig);return p


def pick_fonts():
    """Use the DejaVu fonts bundled with matplotlib (cross-platform)."""
    from matplotlib import font_manager
    regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="normal"))
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    for path in (regular, bold):
        if not Path(path).exists():
            raise RuntimeError("Matplotlib DejaVu fonts could not be found")
    pdfmetrics.registerFont(TTFont("UVIRegular", regular))
    pdfmetrics.registerFont(TTFont("UVIBold", bold))
    pdfmetrics.registerFontFamily("UVIRegular", normal="UVIRegular", bold="UVIBold")


def make_styles():
    return {
        "hero": ParagraphStyle("Hero", fontName="UVIBold", fontSize=22, leading=29, textColor=NAVY, spaceAfter=12),
        "sub": ParagraphStyle("Sub", fontName="UVIRegular", fontSize=11, leading=17, textColor=GRAY, spaceAfter=12),
        "h1": ParagraphStyle("H1", fontName="UVIBold", fontSize=14, leading=20, textColor=NAVY, spaceBefore=12, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontName="UVIBold", fontSize=11, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5),
        "p": ParagraphStyle("P", fontName="UVIRegular", fontSize=9, leading=15, textColor=NAVY, spaceAfter=7),
        "small": ParagraphStyle("Small", fontName="UVIRegular", fontSize=8, leading=12, textColor=GRAY, spaceAfter=5),
        "cell": ParagraphStyle("Cell", fontName="UVIRegular", fontSize=8, leading=12, textColor=NAVY),
        "cellb": ParagraphStyle("CellB", fontName="UVIBold", fontSize=8, leading=12, textColor=NAVY),
        "kpi": ParagraphStyle("KPI", fontName="UVIBold", fontSize=15, leading=22, textColor=BLUE, alignment=TA_CENTER),
        "kpilabel": ParagraphStyle("KPIlabel", fontName="UVIRegular", fontSize=8, leading=12, textColor=GRAY, alignment=TA_CENTER),
    }


def report_header(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    canvas.setFillColor(NAVY); canvas.rect(0, height - 10, width, 10, stroke=0, fill=1)
    canvas.setFont("UVIBold", 9);canvas.setFillColor(BLUE)
    canvas.drawString(2*cm, height-1.45*cm, "BTH04 / YÊU CẦU 6")
    canvas.setStrokeColor(colors.HexColor("#D8E2EF"));canvas.line(2*cm,1.55*cm,width-2*cm,1.55*cm)
    canvas.setFont("UVIRegular", 8);canvas.setFillColor(GRAY)
    canvas.drawString(2*cm,1.1*cm,"Nguồn: UCI Iris (DOI: 10.24432/C56C76) | Dữ liệu quan sát thực tế")
    canvas.drawRightString(width-2*cm,1.1*cm,f"Trang {doc.page}")
    canvas.restoreState()


def table(rows, widths=None, header=True, pad=7):
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),pad),
        ("BOTTOMPADDING",(0,0),(-1,-1),pad),
        ("LEFTPADDING",(0,0),(-1,-1),7),
        ("RIGHTPADDING",(0,0),(-1,-1),7),
        ("ROWBACKGROUNDS",(0,1 if header else 0),(-1,-1),[colors.white,LIGHT]),
        ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#D6E2EE")),
    ]
    if header:
        cmds += [("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCECF7"))]
    t.setStyle(TableStyle(cmds))
    return t


def sized_img(path, width_cm: float, height_cm: float | None = None):
    import PIL.Image
    wpx,hpx = PIL.Image.open(path).size
    w=width_cm*cm;h=(w*hpx/wpx) if height_cm is None else height_cm*cm
    return RLImage(str(path),width=w,height=h,hAlign="CENTER")


def make_report(df: pd.DataFrame, quality: dict, model: dict, figs: dict) -> None:
    pick_fonts()
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    s=make_styles()
    doc=BaseDocTemplate(str(REPORT), pagesize=(21*cm,29.7*cm),
                        leftMargin=2*cm,rightMargin=2*cm, topMargin=2.2*cm,bottomMargin=2.0*cm,
                        title="Báo cáo khoa học tự động - Phân loại hoa Iris",author="BTH04 - Yêu cầu 6")
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="main",showBoundary=0)
    doc.addPageTemplates(PageTemplate(id="normal",frames=frame,onPage=report_header))
    P=lambda txt, style="p": Paragraph(txt,s[style])
    story=[]

    # 1. Overview and methods
    story += [Spacer(1,.5*cm),P("PHÂN TÍCH DỮ LIỆU IRIS<br/>VÀ MÔ HÌNH PHÂN LOẠI", "hero"),
              P("Mini-project AI + Python | Báo cáo tạo hoàn toàn tự động từ dữ liệu CSV thực tế", "sub")]
    kpis=[]
    for value,label in [
        (str(len(df)),"Mẫu dữ liệu"),("4","Biến số (cm)"),("3","Loài hoa"),
        (f"{model['accuracy_test']*100:.1f}%","Accuracy kiểm tra")]:
        kpis.append([P(value,"kpi"),P(label,"kpilabel")])
    story += [table([[Table([[cell[0]],[cell[1]]],colWidths=[3.7*cm]) for cell in kpis]],widths=[4.25*cm]*4,header=False,pad=9),Spacer(1,.15*cm)]
    story += [P("1. Giới thiệu đề tài", "h1"),
              P("Dữ liệu Iris ghi nhận kích thước đài hoa và cánh hoa thuộc ba loài Setosa, Versicolor và Virginica. "
                "Mục tiêu là khám phá các đặc trưng phân biệt ba loài và thử nghiệm mô hình học máy đơn giản nhằm dự đoán nhãn loài từ bốn phép đo."),
              P("2. Mô tả dữ liệu", "h1"),
              P("Nguồn: UCI Machine Learning Repository, Iris Dataset (R. A. Fisher, 1936; DOI 10.24432/C56C76). "
                "CSV dự án phục dựng đúng các giá trị của tệp UCI <i>iris.data</i> từ bộ Iris có sẵn trong scikit-learn bằng cách hoàn nguyên hai bản ghi đã được scikit-learn hiệu chỉnh. "
                "Dữ liệu UCI được cấp phép CC BY 4.0; mã nguồn dự án sử dụng MIT."),
              table([[P("Biến", "cellb"),P("Ý nghĩa", "cellb"),P("Kiểu", "cellb")]]+
                    [[P(a,"cell"),P(b,"cell"),P("Số thực", "cell")] for a,b in [
                     ("sepal_length_cm","Chiều dài đài hoa (cm)"),
                     ("sepal_width_cm","Chiều rộng đài hoa (cm)"),
                     ("petal_length_cm","Chiều dài cánh hoa (cm)"),
                     ("petal_width_cm","Chiều rộng cánh hoa (cm)")]]+
                    [[P("species","cell"),P("Nhãn loài hoa","cell"),P("Phân loại","cell")]],
                    widths=[5*cm,8*cm,4*cm]),
              P("3. Quy trình làm sạch và phân tích", "h1"),
              P(f"Đọc CSV bằng pandas; ép kiểu số; kiểm tra thiếu, giá trị không hợp lệ và tên loài; chuẩn hóa nhãn. "
                f"Kết quả: <b>{quality['clean_rows']}/{quality['raw_rows']}</b> bản ghi hợp lệ, "
                f"<b>{quality['missing_rows']}</b> bản ghi thiếu dữ liệu, <b>{quality['invalid_rows']}</b> bản ghi sai hợp lệ. "
                f"Có <b>{quality['duplicates_retained']}</b> hàng trùng toàn bộ giá trị; được giữ lại vì phép đo giống nhau không mặc nhiên là dữ liệu nhập trùng."),
              P("Phần học máy sử dụng StandardScaler + Logistic Regression, chia tập theo tỉ lệ 70/30 có phân tầng và cố định random_state=42. "
                "5-fold cross-validation chỉ chạy trên tập huấn luyện để tránh đánh giá nhầm trên tập kiểm tra.","small"),
              PageBreak()]

    # 2. Exploratory visualization
    story += [P("4. Phân tích khám phá dữ liệu (EDA)", "h1"),
              P("4.1. Phân phối chiều dài cánh hoa", "h2"),
              sized_img(figs["box"],16.4),
              P("Biểu đồ hộp cho thấy nhóm Setosa có chiều dài cánh hoa nhỏ hơn rõ rệt, trong khi Versicolor và Virginica có một phần khoảng giá trị giao nhau.","small"),
              P("4.2. Quan hệ giữa chiều dài và chiều rộng cánh hoa", "h2"),
              sized_img(figs["scatter"],16.4),
              P("Đám mây điểm phản ánh tương quan thuận mạnh giữa hai kích thước cánh hoa. Setosa tách biệt tốt; ranh giới giữa Versicolor và Virginica phức tạp hơn.","small"),
              PageBreak()]

    # 3. descriptive stats and correlation
    means=df.groupby("species")[COLUMNS].mean().reindex(SPECIES)
    story += [P("5. Kết quả thống kê mô tả", "h1"),
              P("Trung bình từng đặc trưng theo loài (đơn vị: cm)","p")]
    rows = [[P("Loài", "cellb"),P("Dài đài", "cellb"),P("Rộng đài", "cellb"),P("Dài cánh", "cellb"),P("Rộng cánh", "cellb")]]
    for sp in SPECIES:
        rows.append([P(sp,"cell")]+[P(f"{v:.2f}","cell") for v in means.loc[sp]])
    story += [table(rows,widths=[4.6*cm,3.1*cm,3.1*cm,3.1*cm,3.1*cm],pad=10),
              Spacer(1,.2*cm),P("5.1. Tương quan giữa các biến đo", "h2"),
              sized_img(figs["corr"],14.5)]
    corr=float(df["petal_length_cm"].corr(df["petal_width_cm"]))
    story += [P(f"Hệ số tương quan Pearson giữa chiều dài và chiều rộng cánh hoa đạt <b>r = {corr:.3f}</b>. "
               "Đây là tương quan thống kê trong bộ dữ liệu này, không khẳng định quan hệ nhân quả.","small"),
              P("Diễn giải khoa học", "h2"),
              P("Hai phép đo cánh hoa là những biến có khả năng phân biệt nhóm cao trong các hình minh họa. "
                "Khi triển khai trong thực tế cần kiểm tra thêm dữ liệu ngoài mẫu và độ ổn định của phương pháp đo."),
              PageBreak()]

    # 4. evaluation, limits, credits
    story += [P("6. Đánh giá mô hình phân loại", "h1"),
              P(f"Số mẫu huấn luyện: <b>{model['train_rows']}</b>; số mẫu kiểm tra: <b>{model['test_rows']}</b>. "
                f"Accuracy trên tập kiểm tra: <b>{model['accuracy_test']*100:.2f}%</b>. "
                f"Macro F1: <b>{model['macro_f1']:.3f}</b>. "
                f"Accuracy CV (5 folds trên tập train): <b>{model['cv_mean']*100:.2f}%</b> "
                f"(độ lệch chuẩn {model['cv_std']*100:.2f} điểm phần trăm)."),
              sized_img(figs["cm"],14.2),
              P("Ma trận nhầm lẫn cho biết số dự đoán đúng ở đường chéo. "
                "Các nhãn ngoài đường chéo thể hiện những trường hợp mô hình nhầm giữa các loài.","small"),
              P("7. Kết luận và đề xuất", "h1")]
    if model["accuracy_test"] >= .95:
        summary = "Trong lần chia dữ liệu cố định của bài thực hành, mô hình cho kết quả rất tốt. "
    else:
        summary = "Trong lần chia dữ liệu cố định của bài thực hành, mô hình cho kết quả ở mức khả quan. "
    story += [P(summary + "Kích thước cánh hoa mang nhiều thông tin hữu ích cho phân loại. "
                "Hệ thống đã tự động hóa chu trình từ CSV đến phân tích, đồ thị, đánh giá mô hình và kết xuất PDF."),
              P("Hạn chế: Iris chỉ có 150 mẫu, ba loài cân bằng và không đại diện cho mọi điều kiện thực tế. "
                "Accuracy có thể thay đổi với cách phân chia khác. Đề xuất mở rộng bằng dữ liệu thu thập mới, so sánh mô hình và kiểm thử độc lập trước khi ứng dụng."),
              P("8. Nguồn và điều khoản sử dụng", "h1"),
              P("Fisher, R. (1936). <i>Iris</i> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C56C76. "
                "https://archive.ics.uci.edu/dataset/53/iris - Data: CC BY 4.0 (ghi công tác giả và nguồn).","small"),
              P("Tệp dữ liệu phân tích gốc: data/iris_uci_original.csv; dữ liệu đã chuẩn hóa: data/iris_cleaned.csv. "
                "Mã nguồn: MIT License. Mọi số liệu trong PDF do chương trình tính từ CSV, không dựng thủ công.","small")]
    doc.build(story)


def main():
    ASSETS.mkdir(exist_ok=True,parents=True)
    REPORT.parent.mkdir(exist_ok=True,parents=True)
    df,quality=cleaning()
    model=train_model(df)
    set_chart_style()
    figs={"box":chart_boxplot(df),"scatter":chart_scatter(df),
          "corr":chart_corr(df),"cm":chart_confusion(model)}
    payload={"dataset":"UCI Iris (original values)","doi":"10.24432/C56C76",
             "license":"CC BY 4.0", "quality":quality,"model":model,
             "feature_means":df.groupby("species")[COLUMNS].mean().to_dict()}
    STATS.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    make_report(df,quality,model,figs)
    print(f"CSV input: {RAW}")
    print(f"CSV cleaned: {CLEAN}")
    print(f"PDF: {REPORT}")
    print(f"Accuracy: {model['accuracy_test']:.3f}; CV: {model['cv_mean']:.3f}")

if __name__=="__main__":main()
