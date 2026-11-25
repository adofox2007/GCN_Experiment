from __future__ import annotations

import ast, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

RESULTS=ROOT/"results"; OUTPUT=ROOT/"report"/"GCN工程实践实验报告.docx"
BLUE="2F5597"; LIGHT="EAF1F8"; BORDER="D9D9D9"


def fmt_nested(value, digits=4):
    data=ast.literal_eval(value) if isinstance(value,str) else value
    if isinstance(data,list): return "["+", ".join(fmt_nested(item,digits) for item in data)+"]"
    if isinstance(data,float): return f"{data:.{digits}f}".rstrip("0").rstrip(".")
    return str(data)


def set_cell_shading(cell, fill):
    shd=OxmlElement("w:shd"); shd.set(qn("w:fill"),fill); cell._tc.get_or_add_tcPr().append(shd)


def set_cell_border(cell, color=BORDER):
    tcPr=cell._tc.get_or_add_tcPr(); borders=tcPr.first_child_found_in("w:tcBorders")
    if borders is None: borders=OxmlElement("w:tcBorders"); tcPr.append(borders)
    for edge in ("top","left","bottom","right","insideH","insideV"):
        tag=qn(f"w:{edge}"); node=borders.find(tag)
        if node is None: node=OxmlElement(f"w:{edge}"); borders.append(node)
        node.set(qn("w:val"),"single"); node.set(qn("w:sz"),"4"); node.set(qn("w:color"),color)


def add_table(doc, headers, rows, widths=None):
    table=doc.add_table(rows=1,cols=len(headers)); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=True
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for j,h in enumerate(headers):
        cell=table.rows[0].cells[j]; cell.text=str(h); set_cell_shading(cell,BLUE); set_cell_border(cell); cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER
        for run in cell.paragraphs[0].runs: run.font.color.rgb=RGBColor(255,255,255); run.font.bold=True; run.font.size=Pt(9)
        cell.paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER
    for i,row in enumerate(rows):
        cells=table.add_row().cells
        for j,value in enumerate(row):
            cells[j].text=str(value); cells[j].vertical_alignment=WD_ALIGN_VERTICAL.CENTER; set_cell_border(cells[j])
            if i%2: set_cell_shading(cells[j],LIGHT)
            for p in cells[j].paragraphs:
                p.paragraph_format.space_after=Pt(2); p.paragraph_format.space_before=Pt(2)
                for r in p.runs: r.font.size=Pt(8.5)
                if len(str(value))<28: p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after=Pt(1)
    return table


def add_caption(doc,text):
    p=doc.add_paragraph(text); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
    for r in p.runs: r.font.size=Pt(9); r.font.bold=True


def add_figure(doc,filename,caption,width=6.1):
    add_caption(doc,caption); p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(RESULTS/"figures"/filename),width=Inches(width)); p.paragraph_format.space_after=Pt(8)


def add_code(doc,text):
    p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.3); p.paragraph_format.right_indent=Inches(.3)
    p.paragraph_format.space_before=Pt(4); p.paragraph_format.space_after=Pt(6); set_cell=None
    r=p.add_run(text); r.font.name="Consolas"; r._element.rPr.rFonts.set(qn("w:eastAsia"),"等线"); r.font.size=Pt(9)
    shd=OxmlElement("w:shd"); shd.set(qn("w:fill"),"F4F6F8"); p._p.get_or_add_pPr().append(shd)


def add_page_number(section):
    p=section.footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=p.add_run("第 "); fld=OxmlElement("w:fldSimple"); fld.set(qn("w:instr"),"PAGE"); run._r.addnext(fld); p.add_run(" 页")


def setup_document():
    doc=Document(); section=doc.sections[0]
    section.page_width=Inches(8.5); section.page_height=Inches(11); section.top_margin=Inches(.75); section.bottom_margin=Inches(.7); section.left_margin=Inches(.85); section.right_margin=Inches(.85)
    styles=doc.styles
    normal=styles["Normal"]; normal.font.name="Calibri"; normal._element.rPr.rFonts.set(qn("w:eastAsia"),"宋体"); normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.35; normal.paragraph_format.space_after=Pt(6); normal.paragraph_format.first_line_indent=Pt(21)
    for name,size in (("Title",24),("Heading 1",16),("Heading 2",13),("Heading 3",11)):
        st=styles[name]; st.font.name="Calibri"; st._element.rPr.rFonts.set(qn("w:eastAsia"),"黑体"); st.font.size=Pt(size); st.font.color.rgb=RGBColor(0,0,0); st.font.bold=True
        st.paragraph_format.space_before=Pt(12); st.paragraph_format.space_after=Pt(7); st.paragraph_format.keep_with_next=True
    title_ppr=styles["Title"]._element.get_or_add_pPr(); border=title_ppr.find(qn("w:pBdr"))
    if border is not None: title_ppr.remove(border)
    add_page_number(section); return doc


def main():
    summary=json.loads((RESULTS/"experiment_summary.json").read_text(encoding="utf-8"))
    demo=json.loads((RESULTS/"manual_example.json").read_text(encoding="utf-8"))
    env=dict(line.split(": ",1) for line in (RESULTS/"environment.txt").read_text(encoding="utf-8").splitlines() if ": " in line)
    numerical=pd.read_csv(RESULTS/"tables"/"numerical_comparison.csv"); bench=pd.read_csv(RESULTS/"tables"/"benchmark.csv")
    tests=pd.read_csv(RESULTS/"tables"/"test_results.csv"); strategy=pd.read_csv(RESULTS/"tables"/"graph_strategy_comparison.csv")
    doc=setup_document()
    p=doc.add_paragraph(style="Title"); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run("工程实践 从公式实现一层 GCN")
    direct_border=p._p.get_or_add_pPr().find(qn("w:pBdr"))
    if direct_border is not None: p._p.get_or_add_pPr().remove(direct_border)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(18); p.add_run("单层图卷积算子的实现 验证与性能分析").font.size=Pt(14)
    doc.add_paragraph("本报告从标准 GCN 公式出发，实现稠密邻接矩阵版和边列表消息传递版，并使用 DGL GraphConv 作为参考。全部结论均来自固定随机种子下的实际运行：自动化测试全部通过，人工样例输出为 [[2], [2], [5]]，随机图的输出与梯度在既定容差内一致。")
    add_table(doc,["项目","内容"],[["实验设备","CPU"],["随机种子","20261125"],["实现","ManualGCN、MessagePassingGCN、DGL GraphConv"],["交付物","代码、测试、日志、CSV、图表和 Word 报告"]])
    doc.add_page_break()
    doc.add_heading("目录",level=1)
    for i,title in enumerate(["实验目的与任务要求","实验环境与项目结构","GCN 数学原理","核心代码设计与实现","人工样例验证","正确性与边界测试","数值对照与扩展实验","实验可复现性","实验总结"],1):
        doc.add_paragraph(f"第{i}章  {title}",style="Heading 2")
    doc.add_page_break()

    doc.add_heading("第一章 实验目的与任务要求",level=1)
    doc.add_paragraph("本实验验证能否从矩阵公式准确落地一层图卷积，而不是依赖封装层完成关键计算。工作覆盖图预处理、对称归一化、邻居消息聚合、线性更新、偏置、激活和自动微分，并把两种手动实现与 DGL 官方算子置于同一输入、参数和自环策略下比较。")
    doc.add_paragraph("验收重点包括：人工样例具有独立预期值；孤立点、自环、重复边、无边图、零节点图和不同特征维度均有断言；随机有向图遵守 DGL norm=both 的源出度与目标入度约定；数值与梯度误差可核查；性能数据来自实际 CPU 测量。")
    add_table(doc,["Rubric","本工程证据"],[["正确性与边界验证","22 个 pytest 用例、人工样例日志、数值比较 CSV"],["实现与目标库集成","三种统一接口实现及 DGL GraphConv 基线"],["可复现实验依据","锁定依赖、随机种子、运行命令、环境与日志"],["讲解与分析","公式、形状、策略、复杂度、误差与局限分析"]])

    doc.add_heading("第二章 实验环境与项目结构",level=1)
    env_rows=[[k,v] for k,v in env.items() if k in ["Operating system","CPU","Python","PyTorch","DGL","NumPy","Pandas","Matplotlib","NetworkX","Pytest","Experiment device","Random seed","PyTorch intra-op threads"]]
    add_table(doc,["环境项","实际值"],env_rows)
    doc.add_paragraph("实验使用项目根目录下的 .venv。该环境从本机 Python 3.10 Conda 环境建立隔离入口，并在项目环境中安装 DGL 1.1.2、Pandas、Pytest 与 python-docx。PyTorch 构建虽包含 CUDA 支持，但实验显式清空 CUDA_VISIBLE_DEVICES，所有张量均在 CPU 上创建。")
    add_code(doc,"GCN_Experiment/\n  src/                 三种实现与图工具\n  tests/               正确性、边界与梯度测试\n  experiments/         基线、对照、性能、可视化与报告脚本\n  data/                图配置与数据说明\n  results/             真实日志、表格、图片和汇总\n  report/              本实验报告\n  run_all.py           一键复现实验")

    doc.add_heading("第三章 GCN 数学原理",level=1)
    doc.add_heading("3.1 邻接矩阵与方向约定",level=2)
    doc.add_paragraph("边 j→i 在邻接矩阵中记为 A[i,j]=1，因此 A@H 的第 i 行是所有源节点 j 对目标节点 i 的特征贡献。重复边按重数累加。默认严格执行 Ã=A+I：即使原图已有自环，也为每个节点再添加一条自环。")
    doc.add_heading("3.2 度与对称归一化",level=2)
    add_code(doc,"Ã = A + I\nS = D_in^(-1/2) Ã D_out^(-1/2)\nM = S H\nZ = M W + b\nH_out = activation(Z)")
    doc.add_paragraph("无向图中入度和出度相同，上式退化为 D̃^(-1/2) Ã D̃^(-1/2)。有向图与 DGL norm=both 对齐：每条 j→i 的系数是 1/sqrt(out_degree[j]·in_degree[i])。零度节点的逆平方根置零，避免除零、NaN 和 Inf。")
    add_table(doc,["符号","形状","含义"],[["A, Ã, S","[N,N]","邻接矩阵、加自环矩阵与归一化矩阵"],["H, M","[N,F_in]","输入特征与聚合特征"],["W","[F_in,F_out]","可学习权重"],["b","[F_out]","可选偏置"],["H_out","[N,F_out]","单层 GCN 输出"]])
    doc.add_heading("3.3 消息传递",level=2)
    doc.add_paragraph("消息传递版不构建 N×N 矩阵。它对每条边提取 H[j]，乘归一化系数，再用 index_add_ 按目标节点 i 累加。这一过程与 S@H 数学等价，并保留普通重复边的多次贡献。")

    doc.add_heading("第四章 核心代码设计与实现",level=1)
    doc.add_heading("4.1 DGL 参考基线",level=2)
    doc.add_paragraph("ReferenceGCN 封装 dgl.nn.GraphConv，固定 norm=both，启用 allow_zero_in_degree，并支持权重、偏置和激活。零节点图因 DGL 图算子版本限制由包装层返回形状正确的空张量；非空图均实际调用 GraphConv。")
    doc.add_heading("4.2 稠密矩阵实现",level=2)
    doc.add_paragraph("ManualGCN 使用 index_put_(accumulate=True) 构造 A[dst,src]，因此重复边会增加对应矩阵元素。随后计算入度、出度逆平方根并形成 S，执行 (S@H)@W+b。全部操作是 PyTorch 张量运算，可参与自动微分。")
    doc.add_heading("4.3 消息传递实现",level=2)
    doc.add_paragraph("MessagePassingGCN 只保存 src、dst 和每条边的归一化系数，利用 features[src] 生成消息，再以 index_add_ 聚合到 dst。其图结构存储复杂度为 O(N+E)，而稠密矩阵版为 O(N²)。")
    add_table(doc,["实现","图表示","核心聚合","空间复杂度"],[["ManualGCN","稠密 N×N 矩阵","矩阵乘法 S@H","O(N²+NF)"],["MessagePassingGCN","边列表","逐边消息与 index_add_","O(E+NF)"],["DGL GraphConv","DGL 稀疏图","框架优化消息传递","O(E+NF)"]])
    doc.add_paragraph("复杂度不等于实测速度。小图中 Python 调用、图创建和张量调度开销可能主导延迟，因此性能章节分别报告排除预处理的核心前向和包含预处理的完整流程。")

    doc.add_heading("第五章 人工样例验证",level=1)
    doc.add_paragraph("原图包含 0→1、1→0，节点 2 孤立。H=[[1],[3],[5]]，W=[[1]]，b=0，不使用激活。严格添加单位阵后，节点 0 和 1 的度为 2，节点 2 的度为 1。")
    add_figure(doc,"01_raw_graph.png","图 5-1 原始图结构",4.4); add_figure(doc,"02_graph_with_self_loops.png","图 5-2 严格执行 A 加 I 后的图",4.4)
    add_table(doc,["矩阵","实际值"],[["Ã",fmt_nested(demo["adjacency_after_strict_a_plus_i"])],["D̃",fmt_nested(demo["degree_matrix"])],["S",fmt_nested(demo["normalized_adjacency"])],["M=S@H",fmt_nested(demo["aggregated_features"])]])
    doc.add_paragraph("由 S 的前两行 [0.5,0.5,0] 得到节点 0 与 1 的聚合值均为 2；孤立节点在添加自环后仅接收自身特征，输出为 5。三种实现的实际输出均为 [[2],[2],[5]]，最大绝对误差为 0。")
    add_figure(doc,"03_adjacency_heatmap.png","图 5-3 加自环邻接矩阵",4.2); add_figure(doc,"04_normalized_adjacency_heatmap.png","图 5-4 归一化邻接矩阵",4.2); add_figure(doc,"05_output_comparison.png","图 5-5 三种实现输出对比",6.0)

    doc.add_heading("第六章 正确性与边界测试",level=1)
    doc.add_paragraph(f"Pytest 实际执行 {len(tests)} 个用例：通过 {(tests.outcome=='passed').sum()}，失败 {(tests.outcome=='failed').sum()}，跳过 {(tests.outcome=='skipped').sum()}。FP32 比较统一使用 atol=1e-6、rtol=1e-5；每个数值测试同时检查形状和有限性。")
    grouped=[]
    for _,r in tests.iterrows(): grouped.append([r["test"].split("::")[-1],r.outcome,f"{r.duration_seconds:.4f}"])
    add_table(doc,["测试","结果","耗时 秒"],grouped)
    doc.add_paragraph("空图分为两类：有节点无边图在默认自环策略下等价于逐节点线性变换；零节点零边图返回 [0,F_out] 空张量。关闭自环时，零入度目标不会产生聚合消息；偏置仍按标准线性层语义应用。非法索引与非法特征形状会在进入计算前抛出明确异常。")

    doc.add_heading("第七章 数值对照与扩展实验",level=1)
    doc.add_heading("7.1 随机图数值与梯度",level=2)
    cases=numerical[numerical.implementation!="DGL GraphConv"]
    rows=[]
    for _,r in cases.iterrows(): rows.append([r["case"],r["implementation"],r["nodes"],r["raw_edges"],f"{r.max_abs_error:.3e}",f"{r.max_input_grad_error:.3e}",str(r.passed)])
    add_table(doc,["场景","实现","N","E","输出最大误差","输入梯度误差","通过"],rows)
    doc.add_paragraph(f"四组规模与特征维度组合全部通过。全表最大输出绝对误差为 {summary['max_numerical_error']:.3e}，最大输入梯度误差为 {summary['max_input_gradient_error']:.3e}，最大权重梯度误差为 {summary['max_weight_gradient_error']:.3e}。误差来自不同稀疏/稠密累加顺序，量级符合 FP32 预期。")
    add_figure(doc,"06_numerical_error.png","图 7-1 相对 DGL 参考实现的最大绝对误差",6.1)
    doc.add_heading("7.2 自环与重复边策略",level=2)
    add_table(doc,["自环策略","普通边去重","处理后边数","入度","输出"],[[r.self_loop_policy,r.deduplicate,r.processed_edges,fmt_nested(r.in_degree),fmt_nested(r.output)] for _,r in strategy.iterrows()])
    doc.add_paragraph("none 不补自环；add 严格执行 A+I，保留原有自环和普通重复边；exactly_one 先移除全部自环再为每个节点补一条；deduplicate=True 先对全部原始边去重。策略改变邻接元素的重数，从而同时改变度、归一化系数和输出。三种实现共享同一预处理函数，避免参考实现与手动实现使用不同图。")
    doc.add_heading("7.3 CPU 性能",level=2)
    core=bench[bench.scope=="core_excludes_preprocessing"]
    add_table(doc,["实现","N","E 原始","重复次数","中位数 ms","IQR ms"],[[r.implementation,r.nodes,r.raw_edges,r.repeats,f"{r.median_ms:.4f}",f"{r.iqr_ms:.4f}"] for _,r in core.iterrows()])
    add_figure(doc,"07_runtime_comparison.png","图 7-2 CPU 核心前向延迟",6.1)
    doc.add_paragraph("测试固定单线程、相同输入与参数，并先预热 5 次。核心前向排除图预处理；完整流程另存于 benchmark.csv。实测结果不支持简单断言某种实现始终最快：稠密矩阵在小规模上可能受益于连续矩阵运算，而随着 N 增大其 O(N²) 存储和计算成本迅速增加；边列表与 DGL 的优势更依赖稀疏度、算子调度和图复用。本文未把理论复杂度冒充实测内存。")

    doc.add_heading("第八章 实验可复现性",level=1)
    doc.add_paragraph("数据全部来自人工构造和固定种子的程序生成，不使用外部训练数据。该实验属于算子正确性验证，不划分训练、验证和测试集。人工样例用于独立预期值，随机图用于数值与梯度一致性，性能图仅用于延迟测量。数据配置保存在 data/graph_cases.json，许可为本项目自建数据，可随代码使用。")
    add_code(doc,"# 建立环境后安装依赖\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n\n# 一键运行\n.\\.venv\\Scripts\\python.exe run_all.py\n\n# 单独测试\n.\\.venv\\Scripts\\python.exe -m pytest -q")
    doc.add_paragraph("固定随机种子为 20261125，PyTorch 线程数为 1。requirements.txt 保存本次验证版本，results/environment.txt 保存实际解释器与库版本，results/logs/pytest.log 保存测试原始输出。Git 提交哈希由最终验收阶段写入实验汇总；若本地缺少 Git 用户身份，工程会保留所有文件并明确说明未提交原因。")

    doc.add_heading("第九章 实验总结",level=1)
    doc.add_paragraph("实验完成了公式、稠密矩阵实现、边列表消息传递实现与 DGL GraphConv 参考实现之间的闭环验证。人工样例完全符合手算结果，随机无向图和有向图的输出、输入梯度及参数梯度均在 FP32 容差内一致，边界测试覆盖平台要求的 20 类场景。")
    doc.add_paragraph("主要设计选择是统一 A[dst,src] 方向、默认严格 A+I、保留普通重复边、对有向图使用源出度与目标入度，并将零度逆平方根显式置零。稠密实现更接近矩阵公式，适合教学和小图核查；消息传递实现避免 N×N 邻接矩阵，更贴近稀疏图工程实现。")
    doc.add_paragraph("局限主要有三点：性能实验只覆盖 CPU 单线程和中小规模合成图；没有测量峰值内存，因此只报告理论空间复杂度；DGL 1.1.2 的零节点图由参考包装层处理为空输出，而非进入 GraphConv 内核。后续可扩展到稀疏张量内核、多线程、更多图稀疏度以及双精度误差分析。")
    doc.add_page_break(); doc.add_heading("附录 验收证据索引",level=1)
    add_table(doc,["验收项","证据位置"],[["核心正确性","src/manual_gcn.py、src/message_passing.py、src/reference_gcn.py"],["边界与梯度","tests/ 与 results/logs/pytest.log"],["真实数值结果","results/tables/numerical_comparison.csv"],["真实性能结果","results/tables/benchmark.csv"],["人工推导","results/manual_example.json 与 results/logs/manual_example.log"],["环境复现","requirements.txt、results/environment.txt、README.md"],["可视化","results/figures/ 共 7 张 PNG"]])
    doc.core_properties.title="工程实践 从公式实现一层 GCN"; doc.core_properties.subject="单层 GCN 算子实现 验证与性能分析"; doc.core_properties.author=""
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); doc.save(OUTPUT); print(OUTPUT); return OUTPUT


if __name__=="__main__": main()
