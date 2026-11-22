"""Generate a stand-alone offline Plotly HTML dashboard with working controls.

This is a bonus preview; the assigned interactive dashboard is app.py (Streamlit).
"""
from pathlib import Path
import json
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT/'data'/'iris_cleaned.csv')
stats = json.loads((ROOT/'reports'/'ket_qua_phan_tich.json').read_text(encoding='utf-8'))
features = {
    'Chiều dài đài hoa (cm)':'sepal_length_cm',
    'Chiều rộng đài hoa (cm)':'sepal_width_cm',
    'Chiều dài cánh hoa (cm)':'petal_length_cm',
    'Chiều rộng cánh hoa (cm)':'petal_width_cm',
}
colors={'Setosa':'#0B8F97','Versicolor':'#4C6FFF','Virginica':'#E69550'}
json_data=json.dumps(data.to_dict(orient='records'),ensure_ascii=False)
json_features=json.dumps(features,ensure_ascii=False)
json_colors=json.dumps(colors)
accuracy=stats['model']['accuracy_test']
plotlyjs=get_plotlyjs()
html=r'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTH04 - Iris Dashboard (ngoại tuyến)</title>
<style>
:root{font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;color:#142d4c;background:#f4f7fb}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1250px;margin:auto;padding:30px 26px}
.head{background:#122d4c;color:white;border-radius:18px;padding:30px 34px;position:relative;overflow:hidden}
.head:after{content:'✦';position:absolute;right:22px;top:2px;font-size:122px;color:#28608750;}
.eyebrow{font-size:12px;letter-spacing:1.5px;font-weight:700;color:#9ed7e4}.head h1{margin:8px 0 12px;font-size:34px}
.head p{max-width:760px;color:#cbd8e5;line-height:1.65;margin:0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0}
.kpi,.card{background:white;border-radius:15px;padding:20px;box-shadow:0 3px 20px #18375b0e}
.kpi strong{font-size:28px;display:block;color:#2567bb}.kpi span{font-size:13px;color:#5c6f83}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:17px}.filters h3{margin:0 0 18px}
label{display:block;font-size:13px;font-weight:650;margin-bottom:8px}
select{width:100%;background:#f5f9fd;border:1px solid #d6e2f0;border-radius:8px;padding:9px;color:#153657;font-size:13px}
.group{margin-bottom:20px}.check{font-size:13px;font-weight:450;display:flex;align-items:center;gap:7px;margin:8px 0}
#chart{width:100%;height:480px}.lower{display:grid;grid-template-columns:1.2fr 1fr;gap:17px;margin-top:18px}
table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;background:#e5f2fc}th,td{padding:11px 9px;border-bottom:1px solid #e6eef6}
.button{display:block;text-align:center;background:#2369b9;color:white;padding:12px;border:0;border-radius:9px;text-decoration:none;font-weight:700;font-size:14px;cursor:pointer;margin-top:12px}
.secondary{background:#d8eaf7;color:#17446f}.muted{font-size:12px;color:#5e7186;line-height:1.7}
footer{font-size:12px;color:#6c7d92;padding:22px 4px}
@media(max-width:840px){.layout{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.lower{grid-template-columns:1fr}.head h1{font-size:26px}.wrap{padding:18px 10px}}
</style><script>__PLOTLY_JS__</script></head>
<body><div class="wrap">
<div class="head"><div class="eyebrow">BTH04 • YÊU CẦU 6 • AI + PYTHON</div>
<h1>🌿 Iris Data Explorer</h1><p>Trực quan hóa 150 mẫu hoa Iris với bốn biến đo thực tế. Chọn loài, chọn biến, thay đổi biểu đồ và tải báo cáo khoa học được tạo tự động.</p></div>
<section class="kpis">
<div class="kpi"><strong id="nrows">150</strong><span>Mẫu đang chọn / 150</span></div>
<div class="kpi"><strong id="nspecies">3</strong><span>Loài hoa hiển thị</span></div>
<div class="kpi"><strong>4</strong><span>Biến đo theo cm</span></div>
<div class="kpi"><strong>__ACCURACY__%</strong><span>Accuracy kiểm tra</span></div>
</section>
<div class="layout"><aside class="card filters">
<h3>Bộ lọc tương tác</h3>
<div class="group"><label>Loài hoa</label>
<label class="check"><input type="checkbox" value="Setosa" checked> Setosa</label>
<label class="check"><input type="checkbox" value="Versicolor" checked> Versicolor</label>
<label class="check"><input type="checkbox" value="Virginica" checked> Virginica</label></div>
<div class="group"><label for="type">Loại biểu đồ</label><select id="type"><option value="scatter">Phân tán</option><option value="box">Biểu đồ hộp</option><option value="histogram">Phân phối</option></select></div>
<div class="group"><label for="y">Biến muốn quan sát</label><select id="y"></select></div>
<div class="group"><label for="x">Trục X của biểu đồ phân tán</label><select id="x"></select></div>
<p class="muted">Nguồn: UCI ML Repository (Iris). Biểu đồ sẽ cập nhật ngay khi đổi lựa chọn, không cần kết nối Internet.</p>
</aside><main class="card"><div id="chart"></div></main></div>
<div class="lower"><section class="card"><h3>Thống kê mô tả theo loài</h3><table><thead><tr><th>Loài</th><th>Số mẫu</th><th>Trung bình</th><th>ĐLC</th><th>Min</th><th>Max</th></tr></thead><tbody id="tbody"></tbody></table></section>
<section class="card"><h3>Tải sản phẩm</h3><p class="muted">Báo cáo PDF được sinh từ toàn bộ 150 mẫu (không thay đổi khi lọc đồ thị). Dữ liệu CSV xuất ra tuân theo lựa chọn hiện tại.</p>
<a class="button" href="reports/bao_cao_khoa_hoc_iris.pdf" download>Tải báo cáo khoa học PDF</a>
<button class="button secondary" id="download">Tải dữ liệu CSV đang lọc</button>
<p class="muted">Mã nguồn dashboard Streamlit chuẩn bài tập: <code>app.py</code>. Bản HTML này là bản xem ngoại tuyến bổ sung.</p></section></div>
<footer>Fisher (1936), Iris [Dataset], UCI ML Repository · DOI 10.24432/C56C76 · Dữ liệu CC BY 4.0 · Mã nguồn MIT</footer>
</div>
<script>
const DATA=__DATA__, FEATURES=__FEATURES__, COLORS=__COLORS__;
const labels=Object.keys(FEATURES), species=Object.keys(COLORS);
for(let key of ['x','y']){const el=document.getElementById(key);labels.forEach(t=>{const op=document.createElement('option');op.textContent=t;op.value=FEATURES[t];el.appendChild(op)})}
document.getElementById('x').value='petal_width_cm';document.getElementById('y').value='petal_length_cm';
const selected=()=>species.filter(s=>document.querySelector('input[value="'+s+'"]').checked);
const values=(rows,k)=>rows.map(d=>d[k]);
function avg(a){return a.reduce((x,y)=>x+y,0)/a.length}
function render(){
 const s=selected(),filtered=DATA.filter(r=>s.includes(r.species));
 const type=document.getElementById('type').value,x=document.getElementById('x').value,y=document.getElementById('y').value;
 document.getElementById('nrows').textContent=filtered.length;
 document.getElementById('nspecies').textContent=s.length;
 const traces=[];
 for(const sp of s){const rows=filtered.filter(r=>r.species===sp);
  if(type==='scatter')traces.push({type:'scatter',mode:'markers',name:sp,x:values(rows,x),y:values(rows,y),marker:{size:10,opacity:.8,color:COLORS[sp],line:{width:.5,color:'white'}}});
  if(type==='box')traces.push({type:'box',name:sp,y:values(rows,y),marker:{color:COLORS[sp]},boxpoints:'all',jitter:.3,pointpos:0});
  if(type==='histogram')traces.push({type:'histogram',name:sp,x:values(rows,y),opacity:.62,marker:{color:COLORS[sp]},nbinsx:20});
 }
 const fLabel=Object.entries(FEATURES).find(e=>e[1]===y)?.[0]||y;
 const xLabel=Object.entries(FEATURES).find(e=>e[1]===x)?.[0]||x;
 Plotly.react('chart',traces,{title:{text:type==='scatter'?xLabel+' / '+fLabel:'Phân phối '+fLabel,font:{size:19,color:'#123356'}},
  xaxis:{title:type==='scatter'?xLabel:type==='box'?'Loài hoa':fLabel},
  yaxis:{title:type==='histogram'?'Số quan sát':fLabel},barmode:'overlay',paper_bgcolor:'white',plot_bgcolor:'white',
  legend:{orientation:'h',x:0,y:1.12},margin:{l:55,r:18,t:72,b:70},height:480,font:{family:'Arial, sans-serif',size:12,color:'#243f57'}},
  {responsive:true,displaylogo:false});
 let t='';for(const sp of s){const a=values(filtered.filter(d=>d.species===sp),y),m=avg(a),sd=Math.sqrt(a.reduce((v,x)=>v+(x-m)*(x-m),0)/Math.max(1,a.length-1));
 t+='<tr><td>'+sp+'</td><td>'+a.length+'</td><td>'+m.toFixed(2)+'</td><td>'+sd.toFixed(2)+'</td><td>'+Math.min(...a).toFixed(2)+'</td><td>'+Math.max(...a).toFixed(2)+'</td></tr>'}
 document.getElementById('tbody').innerHTML=t||'<tr><td colspan="6">Chưa chọn loài hoa</td></tr>';
 }
 document.querySelectorAll('input,select').forEach(x=>x.addEventListener('change',render));
 document.getElementById('download').addEventListener('click',()=>{
 let f=DATA.filter(r=>selected().includes(r.species));if(!f.length)return;
 let cols=Object.keys(f[0]);let lines=[cols.join(',')].concat(f.map(r=>cols.map(c=>r[c]).join(',')));
 let url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\n')],{type:'text/csv;charset=utf-8'}));
 let a=document.createElement('a');a.href=url;a.download='iris_filtered.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 render();
</script></body></html>'''
html=html.replace('__PLOTLY_JS__',plotlyjs).replace('__DATA__',json_data).replace('__FEATURES__',json_features).replace('__COLORS__',json_colors).replace('__ACCURACY__',f'{accuracy*100:.2f}')
out=ROOT/'dashboard_offline.html'
out.write_text(html,encoding='utf-8')
print(f'Wrote {out}, {out.stat().st_size/1048576:.1f} MB')
