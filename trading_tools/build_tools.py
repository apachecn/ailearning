"""
Build 4 Excel tools for a Beijing-time, working-professional MES day trader
with a $4000 account, specializing in Al Brooks' H2/L2 and Failed Breakout setups.

Outputs (in ./dist/):
  1_h2l2_scoring_and_trade_log.xlsx
  2_premarket_checklist.xlsx
  3_position_calculator_mes_4000.xlsx
  5_60day_tracker_and_setup_library.xlsx

All files are standalone. Open in Excel / WPS / LibreOffice / Numbers.
"""
from __future__ import annotations
import os
from xlsx_writer import Workbook, Style, Cell, col_letter


# =============================================================
# Color palette (ARGB)
# =============================================================
C_HEADER_BG = "FF1F4E78"   # dark blue
C_HEADER_FG = "FFFFFFFF"   # white
C_SUB_BG    = "FFD9E1F2"   # light blue
C_SECTION_BG = "FFFCE4D6"  # light orange
C_GREEN_BG  = "FFE2EFDA"   # light green
C_RED_BG    = "FFFCE4E4"   # light red
C_YELLOW_BG = "FFFFF2CC"   # light yellow
C_GRAY_BG   = "FFF2F2F2"   # gray
C_INPUT_BG  = "FFFFF9C4"   # soft yellow (input fields)
C_OUTPUT_BG = "FFC6EFCE"   # soft green (computed outputs)
C_WARN_BG   = "FFFFC7CE"   # soft red (warnings)


# =============================================================
# Reusable styles
# =============================================================
def s_title():
    return Style(bold=True, font_size=16, font_color=C_HEADER_FG,
                 fill_color=C_HEADER_BG, align_h="center", align_v="center",
                 border=True)

def s_header():
    return Style(bold=True, font_size=11, font_color=C_HEADER_FG,
                 fill_color=C_HEADER_BG, align_h="center", align_v="center",
                 border=True, wrap=True)

def s_subheader():
    return Style(bold=True, font_size=11, fill_color=C_SUB_BG,
                 align_h="center", align_v="center", border=True, wrap=True)

def s_section():
    return Style(bold=True, font_size=12, fill_color=C_SECTION_BG,
                 align_h="left", align_v="center", border=True)

def s_body(wrap=True, align_h="left", align_v="top", bold=False):
    return Style(font_size=11, align_h=align_h, align_v=align_v,
                 wrap=wrap, border=True, bold=bold)

def s_input(num_fmt=None):
    return Style(font_size=11, fill_color=C_INPUT_BG, align_h="center",
                 align_v="center", border=True, num_fmt=num_fmt)

def s_output(num_fmt=None, bold=True):
    return Style(font_size=11, fill_color=C_OUTPUT_BG, align_h="center",
                 align_v="center", border=True, bold=bold, num_fmt=num_fmt)

def s_warn(num_fmt=None, bold=True):
    return Style(font_size=11, fill_color=C_WARN_BG, align_h="center",
                 align_v="center", border=True, bold=bold, num_fmt=num_fmt)

def s_note():
    return Style(font_size=10, font_color="FF595959", wrap=True,
                 align_h="left", align_v="top", border=True)


# =============================================================
# Tool 1: H2/L2 Scoring + Trade Log
# =============================================================
def build_tool1(path: str) -> None:
    wb = Workbook()

    # ----------- Sheet 1: Instructions -----------
    s = wb.add_sheet("使用说明")
    s.freeze = (1, 0)
    s.col_widths[1] = 4
    s.col_widths[2] = 100

    s.merge(1, 1, 1, 2)
    s.write(1, 1, "H2/L2 + Failed Breakout 打分与交易日志 — 使用说明", style=s_title())
    s.row_heights[1] = 30

    instructions = [
        ("一、文件结构", [
            "本文件包含 3 个工作表：",
            "  • 使用说明（本表）",
            "  • 打分模板（每次交易前填写，用于决定是否入场）",
            "  • 交易日志（每笔交易后填写，累积数据用于周/月复盘）",
        ]),
        ("二、打分模板使用流程", [
            "1. 发现潜在 setup 时，打开「打分模板」新的一行",
            "2. 逐项填写 4 个维度的评分（每项 0/15/25 三档）",
            "3. 总分会自动计算（右下角），≥75 才考虑入场",
            "4. 75–89 分：正常仓位（1 张 MES）",
            "5. ≥90 分：高质量 setup，可在账户 >$5000 后考虑 2 张（当前 1 张）",
            "6. <75 分：放弃，等待下一次机会",
        ]),
        ("三、交易日志字段说明", [
            "• 日期/时段：用北京时间记录（如 2026-05-09 22:15）",
            "• Setup：H2 Long / L2 Short / Failed Breakout Up / Failed Breakout Down",
            "• 入场/止损/目标：以 ES 点数记录（MES = ES 的 1/10 乘数）",
            "• 止损点数 = |入场 - 止损| ，目标点数同理",
            "• R:R 会根据止损/目标自动算出",
            "• 结果：+1R / +2R / -1R / BE (保本) / 时间止损等",
            "• 执行评分（0–10）：与盈亏无关，只评你是否严格按规则执行",
            "• 情绪标签：平静 / FOMO / 报复 / 犹豫 / 疲劳等（供复盘）",
        ]),
        ("四、打分四维度详解（Al Brooks H2 Long 示例）", [
            "维度 1：趋势清晰度（0–25）",
            "    25 分：价格在 20EMA 上方，EMA 明确上倾（斜率目测 >30°）",
            "    15 分：在 EMA 上方但 EMA 走平",
            "    0 分：在 EMA 下方（不做多头 H2）",
            "",
            "维度 2：Swing 结构（0–25）",
            "    25 分：最近 1 小时内有清晰的 2 个 HH + 2 个 HL",
            "    15 分：只有 1 组 HH + HL",
            "    0 分：结构混乱、无法识别",
            "",
            "维度 3：回调深度（0–25）",
            "    25 分：回调到前推进段的 38.2%–61.8%（黄金区）",
            "    15 分：回调到 23.6%–38.2% 或 61.8%–78.6%",
            "    0 分：回调超过 80%（趋势可能已破）",
            "",
            "维度 4：入场 bar 质量（0–25）",
            "    25 分：强 bull bar，收盘位于高点 75% 以上，相对大小中等以上",
            "    15 分：中性 bull bar",
            "    0 分：doji 或弱 bar（等下一根确认）",
        ]),
        ("五、L2 Short 是 H2 Long 的镜像", [
            "所有规则完全对称：",
            "  • 下降趋势 + 价格在 20EMA 下方",
            "  • 2 个 LH + 2 个 LL",
            "  • 第二次反弹深度 38.2%–61.8%",
            "  • 入场 bar：强 bear bar",
        ]),
        ("六、Failed Breakout 补充打分（可在维度 1 栏补注）", [
            "• 前置区间：开盘 30min 内形成 5–15 点区间",
            "• 突破幅度：至少 3 点",
            "• 跟进失败：突破后 3 根 K 线内无法创新高/低",
            "• 反向信号 bar：收盘回到区间内",
            "• 目标：区间另一端",
        ]),
        ("七、风险与纪律硬规则（刻在 DNA 里）", [
            "• 单笔最大风险：账户的 2% = $80（按 $4000 起始）",
            "• 日亏损上限：账户的 4% = $160，触及立即停手",
            "• 日盈利 ≥ +1.5R 后：剩余时段只平不开",
            "• 绝不持仓过夜",
            "• FOMC / CPI / NFP 日：不交易",
            "• 连亏 3 笔：强制冷却 1 小时或直接停手",
            "• 工作日疲劳：主动放弃当晚交易，不硬撑",
        ]),
        ("八、周/月复盘指标（看日志数据）", [
            "• 执行评分平均值（目标 ≥8/10，这比盈亏更重要）",
            "• 入场前打分与最终 R 的相关性（打分是否有预测力）",
            "• Setup 分类胜率（H2 vs L2 vs Failed Breakout）",
            "• 时段胜率（开盘前 30min / 30–90min / 90min+）",
            "• 情绪标签与结果相关性（哪些情绪在毒害你）",
        ]),
    ]

    row = 3
    for section_title, lines in instructions:
        s.merge(row, 1, row, 2)
        s.write(row, 1, section_title, style=s_section())
        s.row_heights[row] = 22
        row += 1
        for line in lines:
            s.write(row, 1, "", style=s_body(wrap=False))
            s.write(row, 2, line, style=s_body())
            s.row_heights[row] = 18
            row += 1
        row += 1  # blank line

    # ----------- Sheet 2: Scoring Template -----------
    s2 = wb.add_sheet("打分模板")
    s2.freeze = (3, 2)

    # column widths
    widths = {1: 16, 2: 16, 3: 12, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 14, 10: 30}
    for c, w in widths.items():
        s2.col_widths[c] = w

    s2.merge(1, 1, 1, 10)
    s2.write(1, 1, "入场前打分模板（每次发现 setup 时填一行）", style=s_title())
    s2.row_heights[1] = 28

    s2.merge(2, 1, 2, 2)
    s2.write(2, 1, "基本信息", style=s_header())
    s2.merge(2, 3, 2, 3)
    s2.write(2, 3, "Setup 类型", style=s_header())
    s2.merge(2, 4, 2, 7)
    s2.write(2, 4, "四维度打分（每项 0/15/25）", style=s_header())
    s2.merge(2, 8, 2, 8)
    s2.write(2, 8, "总分", style=s_header())
    s2.merge(2, 9, 2, 9)
    s2.write(2, 9, "决策", style=s_header())
    s2.merge(2, 10, 2, 10)
    s2.write(2, 10, "备注", style=s_header())

    headers = ["日期(北京)", "时间(HH:MM)", "Setup", "维度1趋势", "维度2结构",
               "维度3回调", "维度4信号bar", "总分", "是否入场", "备注"]
    for i, h in enumerate(headers, start=1):
        s2.write(3, i, h, style=s_subheader())
    s2.row_heights[3] = 32

    # 20 blank rows with formulas
    for r in range(4, 24):
        for c in range(1, 3):
            s2.write(r, c, "", style=s_input())
        s2.write(r, 3, "", style=s_input())
        for c in range(4, 8):
            s2.write(r, c, "", style=s_input(num_fmt="0"))
        s2.write(r, 8, formula=f"IF(COUNTA(D{r}:G{r})=0,\"\",SUM(D{r}:G{r}))",
                 style=s_output(num_fmt="0"))
        s2.write(r, 9,
                 formula=(f'IF(H{r}="","",'
                          f'IF(H{r}>=90,"✅ 高质量-入场",'
                          f'IF(H{r}>=75,"✅ 入场",'
                          f'"❌ 放弃")))'),
                 style=s_output())
        s2.write(r, 10, "", style=s_input())
        s2.row_heights[r] = 22

    # ----------- Sheet 3: Trade Log -----------
    s3 = wb.add_sheet("交易日志")
    s3.freeze = (3, 2)

    widths3 = {
        1: 11, 2: 7, 3: 16, 4: 6, 5: 10, 6: 10, 7: 10,
        8: 8, 9: 8, 10: 8, 11: 9, 12: 9, 13: 10, 14: 8,
        15: 10, 16: 14, 17: 30
    }
    for c, w in widths3.items():
        s3.col_widths[c] = w

    s3.merge(1, 1, 1, 17)
    s3.write(1, 1, "交易日志（每笔成交后填写，用于周/月复盘）", style=s_title())
    s3.row_heights[1] = 28

    headers3 = [
        "日期", "编号", "Setup", "方向",
        "入场价", "止损价", "目标价",
        "止损点数", "目标点数", "R:R",
        "合约数", "结果(R)", "盈亏($)",
        "执行分(0-10)", "情绪标签", "离场原因", "复盘笔记"
    ]
    for i, h in enumerate(headers3, start=1):
        s3.write(2, i, h, style=s_header())
    s3.row_heights[2] = 32

    # subheader hint row
    hints = [
        "YYYY-MM-DD", "#1..", "H2/L2/FBU/FBD", "L/S",
        "点", "点", "点",
        "=|E-F|", "=|G-E|", "=I/H",
        "张", "+/-数字", "=L*H*5*K",
        "0-10", "平静/FOMO/等", "目标/止损/时间/主动", "1-2句"
    ]
    for i, h in enumerate(hints, start=1):
        s3.write(3, i, h, style=s_subheader())
    s3.row_heights[3] = 22

    # 60 blank rows with formulas
    for r in range(4, 64):
        # inputs
        s3.write(r, 1, "", style=s_input())   # date
        s3.write(r, 2, "", style=s_input())   # index
        s3.write(r, 3, "", style=s_input())   # setup
        s3.write(r, 4, "", style=s_input())   # direction
        s3.write(r, 5, "", style=s_input(num_fmt="0.00"))  # entry
        s3.write(r, 6, "", style=s_input(num_fmt="0.00"))  # stop
        s3.write(r, 7, "", style=s_input(num_fmt="0.00"))  # target
        # computed
        s3.write(r, 8,
                 formula=f'IF(OR(E{r}="",F{r}=""),"",ABS(E{r}-F{r}))',
                 style=s_output(num_fmt="0.00"))
        s3.write(r, 9,
                 formula=f'IF(OR(E{r}="",G{r}=""),"",ABS(G{r}-E{r}))',
                 style=s_output(num_fmt="0.00"))
        s3.write(r, 10,
                 formula=f'IF(OR(H{r}="",I{r}="",H{r}=0),"",I{r}/H{r})',
                 style=s_output(num_fmt="0.00"))
        # inputs
        s3.write(r, 11, "", style=s_input(num_fmt="0"))   # contracts
        s3.write(r, 12, "", style=s_input(num_fmt="0.00")) # result R
        # P&L = result_R * stop_points * 5 * contracts (MES multiplier = $5)
        s3.write(r, 13,
                 formula=f'IF(OR(L{r}="",H{r}="",K{r}=""),"",L{r}*H{r}*5*K{r})',
                 style=s_output(num_fmt="$#,##0.00"))
        s3.write(r, 14, "", style=s_input(num_fmt="0"))   # exec score
        s3.write(r, 15, "", style=s_input())              # emotion
        s3.write(r, 16, "", style=s_input())              # exit reason
        s3.write(r, 17, "", style=s_input())              # notes
        s3.row_heights[r] = 22

    # Summary block at top-right of sheet 3 (rows 1-2 header covers)
    # Add a summary sheet instead would complicate; put a compact summary below row 64
    sum_row = 65
    s3.merge(sum_row, 1, sum_row, 17)
    s3.write(sum_row, 1, "统计汇总（自动）", style=s_section())
    s3.row_heights[sum_row] = 22

    labels = [
        ("交易笔数", f'COUNTA(B4:B63)'),
        ("胜率 (%)",
         f'IF(COUNTA(L4:L63)=0,"",COUNTIF(L4:L63,">0")/COUNTA(L4:L63)*100)'),
        ("累计 R",
         f'IF(COUNTA(L4:L63)=0,"",SUM(L4:L63))'),
        ("累计盈亏 ($)",
         f'IF(COUNTA(M4:M63)=0,"",SUM(M4:M63))'),
        ("平均 R:R",
         f'IF(COUNTA(J4:J63)=0,"",AVERAGE(J4:J63))'),
        ("执行分平均",
         f'IF(COUNTA(N4:N63)=0,"",AVERAGE(N4:N63))'),
        ("最大单笔盈亏",
         f'IF(COUNTA(M4:M63)=0,"",MAX(M4:M63))'),
        ("最大单笔亏损",
         f'IF(COUNTA(M4:M63)=0,"",MIN(M4:M63))'),
    ]
    start = sum_row + 1
    for i, (label, formula) in enumerate(labels):
        col_label = 1 + (i % 4) * 4
        col_value = col_label + 1
        row_here = start + (i // 4) * 2
        s3.merge(row_here, col_label, row_here, col_label + 1)
        s3.write(row_here, col_label, label, style=s_subheader())
        s3.merge(row_here + 1, col_label, row_here + 1, col_label + 1)
        num_fmt = "0.00"
        if "盈亏" in label:
            num_fmt = "$#,##0.00"
        elif "笔数" in label:
            num_fmt = "0"
        elif "胜率" in label:
            num_fmt = "0.0"
        s3.write(row_here + 1, col_label, formula=formula,
                 style=s_output(num_fmt=num_fmt))
        s3.row_heights[row_here] = 22
        s3.row_heights[row_here + 1] = 22

    wb.save(path)


# =============================================================
# Tool 2: Pre-market Checklist
# =============================================================
def build_tool2(path: str) -> None:
    wb = Workbook()

    # ----------- Sheet 1: Daily Checklist -----------
    s = wb.add_sheet("开盘前Checklist")
    s.freeze = (2, 0)
    widths = {1: 4, 2: 50, 3: 10, 4: 40}
    for c, w in widths.items():
        s.col_widths[c] = w

    s.merge(1, 1, 1, 4)
    s.write(1, 1, "MES 开盘前 Checklist — 每日打印一份，必须全部打钩才开盘",
            style=s_title())
    s.row_heights[1] = 30

    headers = ["#", "检查项", "完成(Y/N)", "备注 / 数值"]
    for i, h in enumerate(headers, start=1):
        s.write(2, i, h, style=s_header())
    s.row_heights[2] = 24

    items = [
        ("【一、身心状态】", None, None),
        ("今晚睡眠充足、精神状态 ≥7/10？", "Y/N", "状态差 → 今晚不交易"),
        ("今天工作未遭遇重大情绪冲击（争执、坏消息）？", "Y/N", "情绪化 → 今晚不交易"),
        ("未饮酒、未服用影响判断的药物？", "Y/N", ""),
        ("晚餐完毕，准备好水/咖啡（不过量）？", "Y/N", ""),
        ("已告知家人/室友接下来 2 小时不被打扰？", "Y/N", ""),
        ("", None, None),
        ("【二、事件与宏观风险】", None, None),
        ("今晚是否有 FOMC 议息会议？（有 → 不交易）", "Y/N", "纽约时间 14:00"),
        ("今晚是否有 CPI / PPI 数据发布？（有 → 不交易）", "Y/N", "08:30 ET"),
        ("今晚是否有 NFP 非农数据？（有 → 不交易）", "Y/N", "08:30 ET，仅月初"),
        ("今晚是否有重要财报（AAPL/MSFT/NVDA 等）？", "Y/N", "注意盘后波动"),
        ("VIX 当前数值？（>25 仓位减半，>30 不交易）", "数值", ""),
        ("ES overnight session 是否出现异常 gap（>10 点）？", "Y/N", "留意开盘回补"),
        ("", None, None),
        ("【三、技术面准备】", None, None),
        ("已标注前日 ES 高点（Prior Day High）", "数值", ""),
        ("已标注前日 ES 低点（Prior Day Low）", "数值", ""),
        ("已标注 overnight 高点（Globex High）", "数值", ""),
        ("已标注 overnight 低点（Globex Low）", "数值", ""),
        ("已标注最近周高 / 周低", "数值", ""),
        ("20EMA 方向：上 / 平 / 下", "方向", "日内 5min 图"),
        ("当前价位最近的心理整数位（50/100 点倍数）", "数值", ""),
        ("", None, None),
        ("【四、交易计划】", None, None),
        ("今晚主 Setup：H2 Long / L2 Short / Failed Breakout / 观望", "选择", ""),
        ("止损计划：点数 _____ 对应金额 _____", "数值", "单笔 ≤ $80"),
        ("目标计划：至少 1:1.5 R:R", "Y/N", ""),
        ("计划最多交易笔数：___（建议 ≤2）", "数值", ""),
        ("日亏损上限 $160 已设定在交易软件（如有）", "Y/N", ""),
        ("日盈利 ≥ +1.5R 停手规则已写在便签纸上", "Y/N", "贴在显示器边"),
        ("", None, None),
        ("【五、软件与硬件】", None, None),
        ("交易软件已登录，账户余额显示正确", "Y/N", ""),
        ("下单快捷键测试正常（买入 / 卖出 / 平仓）", "Y/N", ""),
        ("图表时间框架：1min / 5min / 15min 三窗齐备", "Y/N", ""),
        ("网络连接稳定（ping <50ms 到 broker）", "Y/N", ""),
        ("手机通知静音（避免分心）", "Y/N", ""),
        ("备用下单通道（电话/APP）已确认可用", "Y/N", ""),
        ("", None, None),
        ("【六、心理最后一问】", None, None),
        ("我今晚交易是为了执行计划，不是为了赚回昨天？", "Y/N", "关键心理"),
        ("如果今晚 0 笔机会，我能接受吗？", "Y/N", "不交易也是交易"),
        ("我理解：资金管理 > setup 质量 > 执行纪律 > 盈亏", "Y/N", "内化这个排序"),
    ]

    row = 3
    idx = 1
    for item in items:
        text, ytype, note = item
        if text == "":
            s.write(row, 1, "", style=s_body(wrap=False))
            for c in range(2, 5):
                s.write(row, c, "", style=s_body(wrap=False))
            s.row_heights[row] = 10
            row += 1
            continue
        if text.startswith("【"):
            s.merge(row, 1, row, 4)
            s.write(row, 1, text, style=s_section())
            s.row_heights[row] = 24
            row += 1
            continue
        s.write(row, 1, idx, style=s_body(align_h="center", wrap=False))
        s.write(row, 2, text, style=s_body())
        s.write(row, 3, "", style=s_input())
        s.write(row, 4, note or "", style=s_note())
        s.row_heights[row] = 26
        idx += 1
        row += 1

    # ----------- Sheet 2: Key Levels Worksheet -----------
    s2 = wb.add_sheet("关键价位表")
    s2.freeze = (2, 0)
    widths = {1: 22, 2: 14, 3: 14, 4: 14, 5: 14, 6: 14}
    for c, w in widths.items():
        s2.col_widths[c] = w

    s2.merge(1, 1, 1, 6)
    s2.write(1, 1, "每日关键价位记录（开盘前填满）", style=s_title())
    s2.row_heights[1] = 28

    s2.write(2, 1, "日期(北京)", style=s_header())
    s2.write(2, 2, "前日高", style=s_header())
    s2.write(2, 3, "前日低", style=s_header())
    s2.write(2, 4, "隔夜高", style=s_header())
    s2.write(2, 5, "隔夜低", style=s_header())
    s2.write(2, 6, "VIX", style=s_header())
    s2.row_heights[2] = 24

    for r in range(3, 63):
        s2.write(r, 1, "", style=s_input())
        for c in range(2, 6):
            s2.write(r, c, "", style=s_input(num_fmt="0.00"))
        s2.write(r, 6, "", style=s_input(num_fmt="0.00"))
        s2.row_heights[r] = 22

    # ----------- Sheet 3: Event Calendar -----------
    s3 = wb.add_sheet("事件日历参考")
    widths = {1: 18, 2: 40, 3: 30}
    for c, w in widths.items():
        s3.col_widths[c] = w

    s3.merge(1, 1, 1, 3)
    s3.write(1, 1, "高风险事件参考表（参考数据源：经济日历）", style=s_title())
    s3.row_heights[1] = 28

    s3.write(2, 1, "事件", style=s_header())
    s3.write(2, 2, "影响描述", style=s_header())
    s3.write(2, 3, "规避策略", style=s_header())
    s3.row_heights[2] = 24

    events = [
        ("FOMC 议息会议",
         "美联储利率决议 + 新闻发布会，当日可能 30+ 点单边波动",
         "当日不交易；第二天开盘等待稳定后再评估"),
        ("CPI（消费者物价指数）",
         "通胀数据，与降息预期强挂钩，5min 图常见 15 点瞬间跳空",
         "发布前 15 分钟至后 30 分钟不持仓"),
        ("PPI（生产者物价指数）",
         "CPI 的前瞻指标，影响较 CPI 略小",
         "同 CPI，但可容忍"),
        ("NFP（非农就业）",
         "每月第一个周五，就业+工资双指标，历史最大单分钟波动"
         "品种之一",
         "当晚整晚不交易"),
        ("GDP 初值 / 修正",
         "季度数据，初值影响较大",
         "发布前后 30 分钟避开"),
        ("Powell / 鲍威尔讲话",
         "任何央行行长即兴讲话都可能引发波动",
         "讲话时段不主动开仓"),
        ("关键财报（AAPL/MSFT/NVDA/GOOGL/AMZN/META/TSLA）",
         "盘后公布，但影响次日开盘和 overnight session",
         "财报后当晚 MES 开盘时观察 gap，不博反向"),
        ("OPEC 会议 / 地缘事件",
         "影响原油进而影响股指风险偏好",
         "有消息当天减半仓或观望"),
        ("期权到期（周/月/季末）",
         "Pinning 效应、Gamma 挤压，波动模式异常",
         "减少日内反转交易，以趋势为主"),
        ("银行 / 金融危机事件",
         "如 2023 年 SVB 事件，跨夜可能引发 50+ 点 gap",
         "保持平仓隔夜原则不变，重大事件日甚至日内也减仓"),
    ]
    for i, (evt, desc, strat) in enumerate(events, start=3):
        s3.write(i, 1, evt, style=s_body(bold=True))
        s3.write(i, 2, desc, style=s_body())
        s3.write(i, 3, strat, style=s_body())
        s3.row_heights[i] = 45

    wb.save(path)


# =============================================================
# Tool 3: Position Calculator ($4000 MES)
# =============================================================
def build_tool3(path: str) -> None:
    wb = Workbook()

    # ----------- Sheet 1: Main Calculator -----------
    s = wb.add_sheet("仓位计算器")
    s.col_widths[1] = 4
    s.col_widths[2] = 30
    s.col_widths[3] = 18
    s.col_widths[4] = 4
    s.col_widths[5] = 45

    s.merge(1, 1, 1, 5)
    s.write(1, 1, "MES 仓位计算器（$4000 账户版）", style=s_title())
    s.row_heights[1] = 30

    # ---- INPUTS section ----
    s.merge(3, 1, 3, 5)
    s.write(3, 1, "输入区（黄色单元格，修改这些值）", style=s_section())
    s.row_heights[3] = 24

    row = 4
    inputs = [
        ("账户净值 ($)", 4000.00, "$#,##0.00", "实时更新，回撤后要调低"),
        ("单笔风险百分比 (%)", 2.00, "0.00",
         "保守 1.5%，标准 2%，激进 2.5%（$4000 不建议超 2%）"),
        ("止损点数 (ES 点)", 8.00, "0.00",
         "从图表结构量取，H2 典型 5–12 点，Failed Breakout 典型 3–6 点"),
        ("MES 合约乘数 ($/点)", 5.00, "0.00",
         "MES 固定 $5/点，勿改"),
        ("单笔佣金+滑点 ($)", 2.50, "0.00",
         "往返估算：IBKR/AMP/Tradovate 约 $1–$4"),
    ]
    # these map to row numbers that formulas reference
    # account=C4, risk%=C5, stop=C6, mult=C7, cost=C8
    for i, (label, value, fmt, note) in enumerate(inputs):
        r = row + i
        s.write(r, 2, label, style=s_body(bold=True, wrap=False))
        s.write(r, 3, value, style=s_input(num_fmt=fmt))
        s.write(r, 5, note, style=s_note())
        s.row_heights[r] = 22

    # ---- OUTPUTS section ----
    row_out = row + len(inputs) + 1
    s.merge(row_out, 1, row_out, 5)
    s.write(row_out, 1, "计算结果区（绿色 = 安全可执行，红色 = 超风险警告）",
            style=s_section())
    s.row_heights[row_out] = 24
    row_out += 1

    # Formulas
    # MaxLoss $ = C4 * C5 / 100
    # TheoreticalContracts = MaxLoss / (StopPts * Mult)
    # RecommendedContracts = FLOOR(Theoretical, 1) (at least 0)
    # Actual risk per contract $ = StopPts * Mult
    # Actual risk total $ = RecommendedContracts * Actual risk per contract
    # Actual risk % = Actual risk total / Account
    # Capital efficiency note
    outputs = [
        ("每笔最大允许亏损 ($)", "C4*C5/100", "$#,##0.00",
         "这是你今晚最不能突破的数字"),
        ("理论合约数", "C4*C5/100/(C6*C7)", "0.00",
         "数学上的精确值（通常不是整数）"),
        ("建议下单合约数", "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))",
         "0",
         "取下整；若为 0，说明止损太宽或资金不足（今晚不交易这个 setup）"),
        ("每张合约实际风险 ($)", "C6*C7", "$#,##0.00",
         "= 止损点数 × $5"),
        ("总实际风险 ($)", "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7",
         "$#,##0.00",
         "建议合约数 × 每张风险（≤ 最大允许亏损）"),
        ("实际风险占账户 (%)",
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7/C4*100",
         "0.00",
         "应 ≤ 输入的风险百分比"),
        ("加上成本后真实亏损 ($)",
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7 + "
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C8",
         "$#,##0.00",
         "包括佣金/滑点"),
        ("1R 盈利预估 (目标=1:1) ($)",
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7 - "
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C8",
         "$#,##0.00",
         "扣除成本后的净 1R 收益（实战中 R:R 多为 1.5–2.0）"),
        ("1.5R 盈利预估 ($)",
         "1.5*IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7 - "
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C8",
         "$#,##0.00",
         "日内盈利目标达到即停手"),
        ("2R 盈利预估 ($)",
         "2*IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C6*C7 - "
         "IF(C4*C5/100/(C6*C7)<1,0,FLOOR(C4*C5/100/(C6*C7),1))*C8",
         "$#,##0.00",
         "trailing 让利润奔跑的常见落点"),
    ]
    for i, (label, formula, fmt, note) in enumerate(outputs):
        r = row_out + i
        s.write(r, 2, label, style=s_body(bold=True, wrap=False))
        # highlight the "recommended contracts" row
        if "建议下单" in label:
            s.write(r, 3, formula=formula,
                    style=Style(bold=True, font_size=14, fill_color="FFFFC000",
                                align_h="center", align_v="center", border=True,
                                num_fmt=fmt))
            s.row_heights[r] = 28
        else:
            s.write(r, 3, formula=formula, style=s_output(num_fmt=fmt))
            s.row_heights[r] = 22
        s.write(r, 5, note, style=s_note())

    # ---- WARNINGS section ----
    row_warn = row_out + len(outputs) + 1
    s.merge(row_warn, 1, row_warn, 5)
    s.write(row_warn, 1, "⚠️ 自动检查（出现红色警告需立即重评估）", style=s_section())
    s.row_heights[row_warn] = 24

    # indices: contracts row is at row_out + 2 (建议下单合约数)
    # "建议下单" index in outputs list is 2
    contracts_row = row_out + 2
    actual_risk_pct_row = row_out + 5

    checks = [
        ("止损点数是否 <4 点？",
         f"IF(C6<4,\"⚠️ 止损过紧，易被噪音打掉\",\"✓ OK\")"),
        ("止损点数是否 >15 点？",
         f"IF(C6>15,\"⚠️ 止损过宽，setup 选点有问题\",\"✓ OK\")"),
        ("建议合约数是否 = 0？",
         f"IF(D{contracts_row}=0,\"⚠️ 资金不足以承载此止损，放弃此笔\",\"✓ OK\")"),
        ("实际风险 % 是否超过设定？",
         f"IF(D{actual_risk_pct_row}>C5,\"⚠️ 风险超标（罕见）\",\"✓ OK\")"),
        ("单笔风险 % 是否 >3%？",
         "IF(C5>3,\"⚠️ 风险百分比过激进\",\"✓ OK\")"),
        ("账户净值是否 <$2000？",
         "IF(C4<2000,\"⚠️ 账户过小，考虑暂停实盘\",\"✓ OK\")"),
    ]
    for i, (label, formula) in enumerate(checks):
        r = row_warn + 1 + i
        s.write(r, 2, label, style=s_body(wrap=False))
        s.write(r, 3, formula=formula, style=s_body(align_h="center"))
        s.row_heights[r] = 22

    # ---- Daily circuit breaker tracker ----
    row_cb = row_warn + len(checks) + 2
    s.merge(row_cb, 1, row_cb, 5)
    s.write(row_cb, 1, "日内熔断跟踪（当天进行中填写）", style=s_section())
    s.row_heights[row_cb] = 24

    cb_items = [
        ("本日已交易笔数", 0, "0", "≥2 后慎开新仓"),
        ("本日已亏损累计 ($)", 0, "$#,##0.00",
         "达到 $160（账户 4%）立即停手"),
        ("本日已盈利累计 ($)", 0, "$#,##0.00",
         "达到 1.5R（约 $60–$120）只平不开"),
        ("本日连续亏损次数", 0, "0", "=3 → 强制冷却 1 小时"),
    ]
    for i, (label, value, fmt, note) in enumerate(cb_items):
        r = row_cb + 1 + i
        s.write(r, 2, label, style=s_body(bold=True, wrap=False))
        s.write(r, 3, value, style=s_input(num_fmt=fmt))
        s.write(r, 5, note, style=s_note())
        s.row_heights[r] = 22

    # circuit breaker status
    r = row_cb + 1 + len(cb_items)
    s.write(r, 2, "熔断状态（自动）", style=s_body(bold=True, wrap=False))
    cb_loss_row = row_cb + 2
    cb_profit_row = row_cb + 3
    cb_consec_row = row_cb + 4
    formula = (
        f"IF(C{cb_loss_row}<=-C4*4/100,\"🛑 已触日亏损熔断 — 停止交易\","
        f"IF(C{cb_profit_row}>=C4*3/100,\"🟢 达成日盈利目标 — 只平不开\","
        f"IF(C{cb_consec_row}>=3,\"🟡 连亏 3 — 冷却 1 小时\","
        f"\"✅ 正常交易\")))"
    )
    s.write(r, 3, formula=formula,
            style=Style(bold=True, font_size=13, fill_color="FFFFF2CC",
                        align_h="center", align_v="center", border=True))
    s.row_heights[r] = 32

    # ----------- Sheet 2: Scenario Table -----------
    s2 = wb.add_sheet("场景速查表")
    s2.col_widths[1] = 14
    s2.col_widths[2] = 14
    for c in range(3, 13):
        s2.col_widths[c] = 11

    s2.merge(1, 1, 1, 12)
    s2.write(1, 1, "场景速查：不同账户 / 止损下的建议合约数（风险 2% 固定）",
             style=s_title())
    s2.row_heights[1] = 28

    s2.merge(2, 1, 3, 1)
    s2.write(2, 1, "账户 ($)", style=s_header())
    s2.merge(2, 2, 2, 12)
    s2.write(2, 2, "止损点数（ES/MES 点）", style=s_header())
    stops = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]
    for i, sp in enumerate(stops):
        s2.write(3, 2 + i, sp, style=s_subheader())
    s2.row_heights[2] = 22
    s2.row_heights[3] = 22

    accounts = [2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000]
    for i, acct in enumerate(accounts):
        r = 4 + i
        s2.write(r, 1, acct, style=s_body(align_h="center", bold=True,
                                           wrap=False))
        for j, sp in enumerate(stops):
            # contracts = floor(acct * 0.02 / (sp * 5))
            contracts = int((acct * 0.02) // (sp * 5))
            if contracts == 0:
                style = s_warn()
                txt = 0
            elif contracts >= 5:
                style = Style(bold=True, font_size=11,
                              fill_color="FFB4C7E7", align_h="center",
                              align_v="center", border=True)
                txt = contracts
            else:
                style = s_output()
                txt = contracts
            s2.write(r, 2 + j, txt, style=style)
        s2.row_heights[r] = 22

    # legend
    legend_row = 4 + len(accounts) + 1
    s2.merge(legend_row, 1, legend_row, 12)
    s2.write(legend_row, 1, "图例：红色 = 资金不足以开仓（止损太宽 or 账户太小）；"
             "绿色 = 1–4 张正常；蓝色 = 5+ 张（高级别仓位，需更成熟后再用）",
             style=s_note())
    s2.row_heights[legend_row] = 40

    wb.save(path)


# =============================================================
# Tool 5: 60-Day Tracker + Setup Library Guide
# =============================================================
def build_tool5(path: str) -> None:
    wb = Workbook()

    # ----------- Sheet 1: 60-Day Tracker -----------
    s = wb.add_sheet("60天实盘追踪")
    s.freeze = (3, 2)
    widths = {1: 11, 2: 8, 3: 9, 4: 9, 5: 9, 6: 9, 7: 9, 8: 10, 9: 10,
              10: 11, 11: 11, 12: 12, 13: 10, 14: 28}
    for c, w in widths.items():
        s.col_widths[c] = w

    s.merge(1, 1, 1, 14)
    s.write(1, 1, "前 60 天实盘追踪表 — 关注执行质量，不是每日盈亏",
            style=s_title())
    s.row_heights[1] = 28

    # instructions below title
    s.merge(2, 1, 2, 14)
    s.write(2, 1,
            "核心原则：这 60 天的目标是【建立纪律】，不是赚钱。"
            "执行分平均 ≥8、规则遵守率 100% 就是成功，哪怕账户小幅亏损。",
            style=s_note())
    s.row_heights[2] = 30

    headers = [
        "日期", "交易日#", "笔数", "胜", "负", "平",
        "当日R", "当日$", "累计$",
        "执行分", "规则违反?", "情绪分(1-10)", "睡眠分(1-10)", "当日关键复盘"
    ]
    for i, h in enumerate(headers, start=1):
        s.write(3, i, h, style=s_header())
    s.row_heights[3] = 32

    for i in range(60):
        r = 4 + i
        s.write(r, 1, "", style=s_input())
        s.write(r, 2, i + 1, style=s_body(align_h="center", wrap=False))
        s.write(r, 3, "", style=s_input(num_fmt="0"))
        s.write(r, 4, "", style=s_input(num_fmt="0"))
        s.write(r, 5, "", style=s_input(num_fmt="0"))
        s.write(r, 6, "", style=s_input(num_fmt="0"))
        s.write(r, 7, "", style=s_input(num_fmt="0.00"))
        s.write(r, 8, "", style=s_input(num_fmt="$#,##0.00"))
        if i == 0:
            s.write(r, 9, formula="H4", style=s_output(num_fmt="$#,##0.00"))
        else:
            s.write(r, 9,
                    formula=f"IF(H{r}=\"\",I{r-1},I{r-1}+H{r})",
                    style=s_output(num_fmt="$#,##0.00"))
        s.write(r, 10, "", style=s_input(num_fmt="0"))
        s.write(r, 11, "", style=s_input())  # rule violation Y/N
        s.write(r, 12, "", style=s_input(num_fmt="0"))
        s.write(r, 13, "", style=s_input(num_fmt="0"))
        s.write(r, 14, "", style=s_input())
        s.row_heights[r] = 22

    # summary row
    sr = 4 + 60 + 1
    s.merge(sr, 1, sr, 14)
    s.write(sr, 1, "60 天汇总（自动）", style=s_section())
    s.row_heights[sr] = 24
    summary_labels = [
        ("交易日天数", "COUNTA(A4:A63)", "0"),
        ("总笔数", "SUM(C4:C63)", "0"),
        ("总胜", "SUM(D4:D63)", "0"),
        ("总负", "SUM(E4:E63)", "0"),
        ("胜率 (%)",
         'IF(SUM(D4:D63)+SUM(E4:E63)=0,"",SUM(D4:D63)/(SUM(D4:D63)+SUM(E4:E63))*100)',
         "0.00"),
        ("累计 R", "SUM(G4:G63)", "0.00"),
        ("累计 $", "SUM(H4:H63)", "$#,##0.00"),
        ("执行分均值",
         'IF(COUNTA(J4:J63)=0,"",AVERAGE(J4:J63))', "0.00"),
        ("规则违反次数",
         'COUNTIF(K4:K63,"Y")', "0"),
        ("情绪分均值",
         'IF(COUNTA(L4:L63)=0,"",AVERAGE(L4:L63))', "0.00"),
        ("睡眠分均值",
         'IF(COUNTA(M4:M63)=0,"",AVERAGE(M4:M63))', "0.00"),
        ("最大回撤($, 估算)",
         'IF(COUNTA(I4:I63)=0,"",MIN(I4:I63)-MAX(I4:I63))', "$#,##0.00"),
    ]
    for i, (label, formula, fmt) in enumerate(summary_labels):
        col = 1 + (i % 4) * 3
        row_here = sr + 1 + (i // 4) * 2
        s.merge(row_here, col, row_here, col + 1)
        s.write(row_here, col, label, style=s_subheader())
        s.merge(row_here + 1, col, row_here + 1, col + 1)
        s.write(row_here + 1, col, formula=formula, style=s_output(num_fmt=fmt))
        s.row_heights[row_here] = 22
        s.row_heights[row_here + 1] = 22

    # ----------- Sheet 2: Milestones -----------
    s2 = wb.add_sheet("阶段里程碑")
    widths = {1: 12, 2: 22, 3: 45, 4: 40}
    for c, w in widths.items():
        s2.col_widths[c] = w

    s2.merge(1, 1, 1, 4)
    s2.write(1, 1, "分阶段评估里程碑（前 60 天 + 之后）", style=s_title())
    s2.row_heights[1] = 28

    s2.write(2, 1, "阶段", style=s_header())
    s2.write(2, 2, "目标", style=s_header())
    s2.write(2, 3, "评估指标", style=s_header())
    s2.write(2, 4, "失败退出标准", style=s_header())
    s2.row_heights[2] = 28

    milestones = [
        ("Day 1–10",
         "纯熟悉期",
         "每天按时坐下盯盘；完成 checklist；记录 ≥3 条当日观察",
         "< 7 天出现在交易桌前 → 重新评估时间可行性"),
        ("Day 11–30",
         "setup 识别期",
         "能在 5min 图上口头讲出 H2/L2 的出现；每晚复盘至少 1 个图例；"
         "执行分 ≥7",
         "打分与结果无相关 → 回到案例学习，不加仓"),
        ("Day 31–60",
         "执行稳定期",
         "规则违反 ≤ 5 次；执行分均值 ≥8；累计亏损 ≤ 20%（$800）",
         "累计亏损 >30%（$1200）→ 强制停手 2 周复盘"),
        ("Day 61–120",
         "盈亏平衡期",
         "月均盈亏 -2% ~ +2%；胜率稳定 45%+；平均 R:R ≥ 1.5",
         "累计月数亏损 2 个月 → 回到模拟盘"),
        ("Day 121–240",
         "正期望确认期",
         "累计盈利（扣成本）；月度 MAR（收益/回撤）>1.5",
         "不满足则不进入下一阶段"),
        ("Day 241+",
         "加仓决策期",
         "账户 ≥ $6000 才考虑 2 张 MES；先做 3 个月 2 张模拟",
         "切勿因短期盈利而跳过验证"),
    ]
    for i, (phase, goal, metric, fail) in enumerate(milestones, start=3):
        s2.write(i, 1, phase, style=s_body(bold=True))
        s2.write(i, 2, goal, style=s_body(bold=True))
        s2.write(i, 3, metric, style=s_body())
        s2.write(i, 4, fail, style=s_body())
        s2.row_heights[i] = 60

    # ----------- Sheet 3: Setup Library Guide -----------
    s3 = wb.add_sheet("Setup图库构建指南")
    s3.col_widths[1] = 4
    s3.col_widths[2] = 100

    s3.merge(1, 1, 1, 2)
    s3.write(1, 1, "自建 H2/L2 + Failed Breakout 图库的完整流程", style=s_title())
    s3.row_heights[1] = 28

    guide = [
        ("为什么必须自建图库", [
            "Al Brooks 书里的案例是 S&P 多年前的数据，波动率、流动性、算法主导度都已变化。",
            "你自己时段（北京时间开盘后 2 小时）的 setup 有特殊纹理，别人的图库捕捉不到。",
            "自建过程本身就是刻意练习，200 张图看下来，你对 H2/L2 的识别会进入直觉层。",
            "图库是你专属的「第二大脑」，未来每一次犹豫都可以翻看。",
        ]),
        ("目标数量与时间", [
            "H2 Long：60 张 A 级 + 60 张 B 级 + 30 张失败案例 = 150 张",
            "L2 Short：同上 150 张",
            "Failed Breakout：共 100 张（分 Up/Down）",
            "合计约 400 张，用 3–6 个月完成。",
            "每天花 20 分钟看 3–5 张历史图，累计 60 天基本完成核心框架。",
        ]),
        ("数据源（免费为主）", [
            "• TradingView 免费版即可（ES1! 或 MES1! 合约，5min + 1min 图）",
            "• Barchart.com 免费历史日内数据",
            "• 你自己的交易软件：回看自己经历过的每一天",
            "• 社区资源：Al Brooks 官方 YouTube 频道大量免费案例",
            "• Lance Beggs 的 YourTradingCoach 博客有免费的 price action 图例",
        ]),
        ("标注工具", [
            "TradingView：直接截图 + 画线工具标注",
            "命名规则：YYYYMMDD_时段_Setup_等级.png",
            "示例：20250509_2145BJT_H2L_A.png",
            "存储：云盘分文件夹（H2-A / H2-B / H2-Failed / L2-A ...）",
        ]),
        ("每张图必须标注的 6 个元素", [
            "1. 20EMA 的位置与斜率",
            "2. 最近 2 个 HH + 2 个 HL（或 LH + LL）的连线",
            "3. 第一次回调（Pullback 1）起止点",
            "4. 第二次回调（Pullback 2）起止点",
            "5. 入场信号 bar（圈出来）",
            "6. 如果是历史复盘，标注实际止损位与最终结果（+多少 R / 止损 / 保本）",
        ]),
        ("A / B / C 等级判定标准", [
            "A 级（理想案例）：4 项打分全部 25，趋势极清晰，信号 bar 完美",
            "B 级（常见案例）：总分 75–89，有 1–2 项是 15 分",
            "C 级（勉强案例）：总分 60–74，不交易但可以学习",
            "失败案例：打分够高但最终止损 → 分析为何失败（极其宝贵）",
        ]),
        ("使用图库复盘的方法", [
            "每周日晚：翻看上周所有新加入的图 + 随机抽查 20 张旧图",
            "问自己：当时如果再遇到，我能多快识别？能否口头讲出 setup 的 4 要素？",
            "将失败案例单独再看一遍：下次见到类似的，跳过还是照做？",
            "每月一次：删除重复性过高的图（保留代表性即可，不要囤积）",
        ]),
        ("进阶：建立你自己的「特征清单」", [
            "随着图看得多，你会发现当晚当地有一些特殊规律，比如：",
            "• 北京时间 23:00 前后（美东 11:00）常见午休回调 → H2 胜率高",
            "• 某些财报日第二天开盘 30min 内，Failed Breakout 概率高",
            "• VIX 突然 >20 的日子，H2 容易变成 H2 失败",
            "把这些「你专属的规律」写进个人特征清单，每 30 天扩充一次。",
            "这才是 Al Brooks 书里学不到的、你真正的 edge。",
        ]),
        ("常见错误", [
            "❌ 囤图不复盘（有 500 张但每张只看过一次 = 等于没做）",
            "❌ 只收集成功案例（失败案例才是 edge 的来源）",
            "❌ 标注太潦草（3 个月后自己都看不懂）",
            "❌ 不分级（所有 H2 等价，失去筛选意义）",
            "❌ 数据源混乱（用 SPY + QQQ + ES 混标，周期单位乱）",
        ]),
        ("起步 7 天行动计划", [
            "Day 1：创建文件夹结构，学会 TradingView 截图 + 画线",
            "Day 2–3：找 10 张 Al Brooks 书里或 YouTube 里的经典 H2 图，自己重画",
            "Day 4：回看本周美股任意 1 天 MES 5min 图，手动找 H2，标 3–5 张",
            "Day 5：同上，找 L2",
            "Day 6：同上，找 Failed Breakout",
            "Day 7：整理一周的 15–20 张图，按 A/B/C 分类，写 1 份总结",
            "从第 8 天起：每个交易日结束后至少收 2 张（自己当晚的 + 历史的）",
        ]),
    ]

    row = 3
    for title, lines in guide:
        s3.merge(row, 1, row, 2)
        s3.write(row, 1, title, style=s_section())
        s3.row_heights[row] = 24
        row += 1
        for line in lines:
            s3.write(row, 1, "", style=s_body(wrap=False))
            s3.write(row, 2, line, style=s_body())
            s3.row_heights[row] = 22
            row += 1
        row += 1

    # ----------- Sheet 4: Weekly Review Template -----------
    s4 = wb.add_sheet("周复盘模板")
    widths = {1: 30, 2: 65}
    for c, w in widths.items():
        s4.col_widths[c] = w

    s4.merge(1, 1, 1, 2)
    s4.write(1, 1, "每周日晚复盘模板（20 分钟内完成）", style=s_title())
    s4.row_heights[1] = 28

    s4.write(2, 1, "问题", style=s_header())
    s4.write(2, 2, "回答", style=s_header())
    s4.row_heights[2] = 22

    questions = [
        "1. 本周交易了几天？几笔？",
        "2. 胜率、R:R、累计 R、累计 $？",
        "3. 执行分平均分？有没有规则违反？",
        "4. 最得意的一笔（贴图+1 句总结）",
        "5. 最糟糕的一笔（贴图+1 句原因诊断）",
        "6. 被噪音止损的比例高吗？（>30% → 止损太紧）",
        "7. 错过了几个 A 级 setup？原因？",
        "8. 情绪有无异常（FOMO / 报复 / 疲劳高发）？",
        "9. 睡眠均分？工作日有无被交易影响？",
        "10. 本周新入库了多少张 setup 图？",
        "11. 本周最大的认知升级是什么？（1–3 句）",
        "12. 下周需要特别注意的 1 件事？",
    ]
    for i, q in enumerate(questions, start=3):
        s4.write(i, 1, q, style=s_body(bold=True))
        s4.write(i, 2, "", style=s_input())
        s4.row_heights[i] = 50

    wb.save(path)


# =============================================================
# Main
# =============================================================
def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
    os.makedirs(out_dir, exist_ok=True)

    tools = [
        ("1_h2l2_scoring_and_trade_log.xlsx", build_tool1),
        ("2_premarket_checklist.xlsx", build_tool2),
        ("3_position_calculator_mes_4000.xlsx", build_tool3),
        ("5_60day_tracker_and_setup_library.xlsx", build_tool5),
    ]
    for fname, builder in tools:
        p = os.path.join(out_dir, fname)
        builder(p)
        size = os.path.getsize(p)
        print(f"  ✓ {fname}  ({size:,} bytes)")
    print(f"\nAll saved to: {out_dir}")


if __name__ == "__main__":
    main()
