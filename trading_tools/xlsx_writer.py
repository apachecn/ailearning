"""
Minimal XLSX writer using only Python standard library.

Supports: multiple sheets, strings, numbers, formulas, basic styles
(bold, font size, fill color, borders, text wrap, column widths,
merged cells, frozen panes).

Why: the sandbox cannot install openpyxl/xlsxwriter; xlsx is just a
zip of XML parts, which we assemble by hand.
"""
from __future__ import annotations

import zipfile
import html
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ---------- Style model ----------

@dataclass(frozen=True)
class Style:
    bold: bool = False
    font_size: int = 11
    font_color: str = "FF000000"      # ARGB
    fill_color: Optional[str] = None  # ARGB, None = no fill
    border: bool = False
    wrap: bool = False
    align_h: Optional[str] = None     # left / center / right
    align_v: Optional[str] = None     # top / center / bottom
    num_fmt: Optional[str] = None     # e.g. "0.00", "0.00%", "$#,##0.00"


# ---------- Cell ----------

@dataclass
class Cell:
    value: Any = None       # str / int / float / None
    formula: Optional[str] = None
    style: Optional[Style] = None


# ---------- Sheet ----------

@dataclass
class Sheet:
    name: str
    rows: List[List[Cell]] = field(default_factory=list)
    col_widths: Dict[int, float] = field(default_factory=dict)   # 1-indexed col -> width
    row_heights: Dict[int, float] = field(default_factory=dict)  # 1-indexed row -> height
    merges: List[str] = field(default_factory=list)              # e.g. "A1:C1"
    freeze: Optional[Tuple[int, int]] = None                      # (freeze_rows, freeze_cols)

    def set(self, row: int, col: int, cell: Cell) -> None:
        """row, col are 1-indexed."""
        while len(self.rows) < row:
            self.rows.append([])
        r = self.rows[row - 1]
        while len(r) < col:
            r.append(Cell())
        r[col - 1] = cell

    def write(self, row: int, col: int, value: Any = None,
              formula: Optional[str] = None, style: Optional[Style] = None) -> None:
        self.set(row, col, Cell(value=value, formula=formula, style=style))

    def merge(self, r1: int, c1: int, r2: int, c2: int) -> None:
        self.merges.append(f"{col_letter(c1)}{r1}:{col_letter(c2)}{r2}")


# ---------- Helpers ----------

def col_letter(col: int) -> str:
    """1 -> A, 27 -> AA"""
    s = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        s = chr(65 + rem) + s
    return s


def _xml_escape(s: str) -> str:
    return html.escape(s, quote=True)


# ---------- Workbook ----------

class Workbook:
    def __init__(self) -> None:
        self.sheets: List[Sheet] = []
        self._shared_strings: List[str] = []
        self._shared_map: Dict[str, int] = {}
        self._styles: List[Style] = [Style()]  # index 0 = default
        self._style_map: Dict[Style, int] = {Style(): 0}

    def add_sheet(self, name: str) -> Sheet:
        s = Sheet(name=name)
        self.sheets.append(s)
        return s

    def _intern_string(self, s: str) -> int:
        if s not in self._shared_map:
            self._shared_map[s] = len(self._shared_strings)
            self._shared_strings.append(s)
        return self._shared_map[s]

    def _intern_style(self, st: Optional[Style]) -> int:
        if st is None:
            return 0
        if st not in self._style_map:
            self._style_map[st] = len(self._styles)
            self._styles.append(st)
        return self._style_map[st]

    # ---------- XML parts ----------

    def _content_types(self) -> str:
        parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>',
        ]
        for i in range(len(self.sheets)):
            parts.append(
                f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
                f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        parts.append('</Types>')
        return "".join(parts)

    def _root_rels(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )

    def _workbook_xml(self) -> str:
        sheets_xml = "".join(
            f'<sheet name="{_xml_escape(s.name)}" sheetId="{i+1}" r:id="rId{i+1}"/>'
            for i, s in enumerate(self.sheets)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{sheets_xml}</sheets>'
            '</workbook>'
        )

    def _workbook_rels(self) -> str:
        rels = []
        for i in range(len(self.sheets)):
            rels.append(
                f'<Relationship Id="rId{i+1}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i+1}.xml"/>'
            )
        rels.append(
            f'<Relationship Id="rId{len(self.sheets)+1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
        )
        rels.append(
            f'<Relationship Id="rId{len(self.sheets)+2}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
            'Target="sharedStrings.xml"/>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels) +
            '</Relationships>'
        )

    def _shared_strings_xml(self) -> str:
        items = "".join(
            f'<si><t xml:space="preserve">{_xml_escape(s)}</t></si>'
            for s in self._shared_strings
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            f'count="{len(self._shared_strings)}" uniqueCount="{len(self._shared_strings)}">'
            f'{items}</sst>'
        )

    def _styles_xml(self) -> str:
        # Collect unique fonts, fills, borders, numFmts
        fonts: List[Tuple[bool, int, str]] = []
        fonts_map: Dict[Tuple[bool, int, str], int] = {}
        fills: List[Optional[str]] = [None, None]  # indices 0,1 reserved per spec
        fills_map: Dict[Optional[str], int] = {}
        borders_count = 2  # 0 = none, 1 = all-sides
        numfmts: List[str] = []
        numfmts_map: Dict[str, int] = {}

        def intern_font(b: bool, size: int, color: str) -> int:
            key = (b, size, color)
            if key not in fonts_map:
                fonts_map[key] = len(fonts)
                fonts.append(key)
            return fonts_map[key]

        def intern_fill(color: Optional[str]) -> int:
            if color is None:
                return 0
            if color not in fills_map:
                fills_map[color] = len(fills)
                fills.append(color)
            return fills_map[color]

        def intern_numfmt(fmt: str) -> int:
            if fmt not in numfmts_map:
                numfmts_map[fmt] = 164 + len(numfmts)  # custom codes start at 164
                numfmts.append(fmt)
            return numfmts_map[fmt]

        # Seed default font
        intern_font(False, 11, "FF000000")

        xfs: List[Dict] = [{}]  # index 0 default
        for st in self._styles[1:]:
            font_id = intern_font(st.bold, st.font_size, st.font_color)
            fill_id = intern_fill(st.fill_color)
            border_id = 1 if st.border else 0
            numfmt_id = intern_numfmt(st.num_fmt) if st.num_fmt else 0
            xfs.append({
                "fontId": font_id,
                "fillId": fill_id,
                "borderId": border_id,
                "numFmtId": numfmt_id,
                "align_h": st.align_h,
                "align_v": st.align_v,
                "wrap": st.wrap,
            })

        # Build fonts XML
        fonts_xml_parts = []
        for b, size, color in fonts:
            bold_tag = "<b/>" if b else ""
            fonts_xml_parts.append(
                f'<font>{bold_tag}<sz val="{size}"/><color rgb="{color}"/><name val="Calibri"/></font>'
            )
        fonts_xml = f'<fonts count="{len(fonts)}">' + "".join(fonts_xml_parts) + '</fonts>'

        # Build fills XML (indices 0 none, 1 gray125 per spec, then custom)
        fills_xml_parts = [
            '<fill><patternFill patternType="none"/></fill>',
            '<fill><patternFill patternType="gray125"/></fill>',
        ]
        for color in fills[2:]:
            fills_xml_parts.append(
                f'<fill><patternFill patternType="solid"><fgColor rgb="{color}"/><bgColor indexed="64"/></patternFill></fill>'
            )
        fills_xml = f'<fills count="{len(fills_xml_parts)}">' + "".join(fills_xml_parts) + '</fills>'

        # Borders
        borders_xml = (
            f'<borders count="{borders_count}">'
            '<border><left/><right/><top/><bottom/><diagonal/></border>'
            '<border>'
            '<left style="thin"><color rgb="FF808080"/></left>'
            '<right style="thin"><color rgb="FF808080"/></right>'
            '<top style="thin"><color rgb="FF808080"/></top>'
            '<bottom style="thin"><color rgb="FF808080"/></bottom>'
            '<diagonal/>'
            '</border>'
            '</borders>'
        )

        # Num formats
        numfmts_xml_parts = []
        for fmt_str, code in sorted(numfmts_map.items(), key=lambda x: x[1]):
            numfmts_xml_parts.append(f'<numFmt numFmtId="{code}" formatCode="{_xml_escape(fmt_str)}"/>')
        numfmts_xml = (
            f'<numFmts count="{len(numfmts_xml_parts)}">' + "".join(numfmts_xml_parts) + '</numFmts>'
            if numfmts_xml_parts else ''
        )

        # cellXfs
        cellxfs_parts = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>']
        for xf in xfs[1:]:
            apply_align = xf["align_h"] or xf["align_v"] or xf["wrap"]
            align_parts = []
            if xf["align_h"]:
                align_parts.append(f'horizontal="{xf["align_h"]}"')
            if xf["align_v"]:
                align_parts.append(f'vertical="{xf["align_v"]}"')
            if xf["wrap"]:
                align_parts.append('wrapText="1"')
            align_xml = f'<alignment {" ".join(align_parts)}/>' if apply_align else ''
            apply_attrs = []
            if xf["fontId"] != 0:
                apply_attrs.append('applyFont="1"')
            if xf["fillId"] != 0:
                apply_attrs.append('applyFill="1"')
            if xf["borderId"] != 0:
                apply_attrs.append('applyBorder="1"')
            if xf["numFmtId"] != 0:
                apply_attrs.append('applyNumberFormat="1"')
            if apply_align:
                apply_attrs.append('applyAlignment="1"')
            cellxfs_parts.append(
                f'<xf numFmtId="{xf["numFmtId"]}" fontId="{xf["fontId"]}" '
                f'fillId="{xf["fillId"]}" borderId="{xf["borderId"]}" xfId="0" '
                f'{" ".join(apply_attrs)}>{align_xml}</xf>'
            )
        cellxfs_xml = f'<cellXfs count="{len(cellxfs_parts)}">' + "".join(cellxfs_parts) + '</cellXfs>'

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + numfmts_xml
            + fonts_xml
            + fills_xml
            + borders_xml
            + '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            + cellxfs_xml
            + '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            + '</styleSheet>'
        )

    def _sheet_xml(self, sheet: Sheet) -> str:
        # Column widths
        cols_xml = ""
        if sheet.col_widths:
            col_parts = []
            for col, width in sorted(sheet.col_widths.items()):
                col_parts.append(f'<col min="{col}" max="{col}" width="{width}" customWidth="1"/>')
            cols_xml = '<cols>' + "".join(col_parts) + '</cols>'

        # Sheet data
        rows_xml_parts = []
        for r_idx, row in enumerate(sheet.rows, start=1):
            cells_xml_parts = []
            for c_idx, cell in enumerate(row, start=1):
                if cell.value is None and cell.formula is None:
                    continue
                ref = f"{col_letter(c_idx)}{r_idx}"
                style_id = self._intern_style(cell.style)
                style_attr = f' s="{style_id}"' if style_id != 0 else ''
                if cell.formula is not None:
                    # formula cell
                    cells_xml_parts.append(
                        f'<c r="{ref}"{style_attr}><f>{_xml_escape(cell.formula)}</f></c>'
                    )
                elif isinstance(cell.value, bool):
                    cells_xml_parts.append(
                        f'<c r="{ref}"{style_attr} t="b"><v>{1 if cell.value else 0}</v></c>'
                    )
                elif isinstance(cell.value, (int, float)):
                    cells_xml_parts.append(
                        f'<c r="{ref}"{style_attr}><v>{cell.value}</v></c>'
                    )
                else:
                    s_idx = self._intern_string(str(cell.value))
                    cells_xml_parts.append(
                        f'<c r="{ref}"{style_attr} t="s"><v>{s_idx}</v></c>'
                    )
            height_attr = ''
            if r_idx in sheet.row_heights:
                height_attr = f' ht="{sheet.row_heights[r_idx]}" customHeight="1"'
            rows_xml_parts.append(
                f'<row r="{r_idx}"{height_attr}>' + "".join(cells_xml_parts) + '</row>'
            )
        sheetdata_xml = '<sheetData>' + "".join(rows_xml_parts) + '</sheetData>'

        # Merges
        merges_xml = ""
        if sheet.merges:
            merge_parts = [f'<mergeCell ref="{m}"/>' for m in sheet.merges]
            merges_xml = f'<mergeCells count="{len(merge_parts)}">' + "".join(merge_parts) + '</mergeCells>'

        # Freeze
        sheetviews_xml = ""
        if sheet.freeze:
            fr_rows, fr_cols = sheet.freeze
            top_left = f'{col_letter(fr_cols + 1)}{fr_rows + 1}'
            pane_state = "frozen"
            x_split = f' xSplit="{fr_cols}"' if fr_cols > 0 else ''
            y_split = f' ySplit="{fr_rows}"' if fr_rows > 0 else ''
            sheetviews_xml = (
                '<sheetViews><sheetView workbookViewId="0">'
                f'<pane{x_split}{y_split} topLeftCell="{top_left}" state="{pane_state}"/>'
                '</sheetView></sheetViews>'
            )

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + sheetviews_xml
            + cols_xml
            + sheetdata_xml
            + merges_xml
            + '</worksheet>'
        )

    # ---------- Save ----------

    def save(self, path: str) -> None:
        # Must generate sheet XMLs first to populate shared strings/styles
        sheet_xmls = [self._sheet_xml(s) for s in self.sheets]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", self._content_types())
            z.writestr("_rels/.rels", self._root_rels())
            z.writestr("xl/workbook.xml", self._workbook_xml())
            z.writestr("xl/_rels/workbook.xml.rels", self._workbook_rels())
            z.writestr("xl/sharedStrings.xml", self._shared_strings_xml())
            z.writestr("xl/styles.xml", self._styles_xml())
            for i, xml in enumerate(sheet_xmls):
                z.writestr(f"xl/worksheets/sheet{i+1}.xml", xml)
