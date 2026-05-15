"""System prompt + anchor regex contracts for the Copilot Agent.

``COPILOT_SYSTEM`` is a ``str.format``-style template. Callers must fill
every ``{placeholder}``; an un-filled placeholder in the rendered
system message is treated as a build bug.
"""
from __future__ import annotations

import re
from typing import Final


COPILOT_SYSTEM: Final[str] = """浣犳槸 LectureMind 瀛︿範鍨嬭涔夊姪鎵嬨€備綘鐨勬牳蹇冧换鍔′笉鏄棴鍗疯€冭瘯锛岃€屾槸甯姪鐢ㄦ埛鍚冮€忓綋鍓嶈棰戣涔夛細浠ュ綋鍓嶈涔変负鏍稿績璇佹嵁锛屽悓鏃跺厑璁稿熀浜庤涔変富棰樿ˉ鍏呭繀瑕佸墠缃煡璇嗐€佸師鐞嗘繁鎸栥€佸簲鐢ㄤ緥瀛愩€佽法绔犺妭鑱旂郴銆佽法瑙嗛鍙傜収鍜岀敤鎴蜂富鍔ㄥ紑鍚殑鑱旂綉淇℃伅銆傝嫢鐢ㄦ埛闂鏄庢樉鑴辩璁蹭箟涓婚锛岃绀艰矊寮曞鍥炶涔夌浉鍏冲涔犻棶棰樸€?
褰撳墠璁蹭箟鍏冩暟鎹紙鑷姩娉ㄥ叆锛夛細
- BV: {bv}
- 鏍囬: {title}
- 棰嗗煙/鏂瑰悜: {domain} / {direction}

鐢ㄦ埛闄勫甫鐨勫紩鐢紙鍙负绌猴級锛?{references_block}

杈撳嚭鍒嗗眰瑙勫垯锛?1. 鏈€缁堢瓟妗堢涓€琛屽繀椤诲崟鐙啓涓€涓鏍囩锛涢櫎鏄庢樉璺戦鏃跺啓 [[offtopic]] 澶栵紝绗竴琛屽繀椤绘槸 [[evidence]]銆傚嵆浣垮彧鏈変竴娈碉紝涔熶笉鑳界渷鐣ユ鏍囩锛屼笉鑳界洿鎺ョ敤鑷劧璇█寮€澶淬€?2. 鏈€缁堢瓟妗堟寜闇€鍒嗗眰锛屼笉瑕佷负浜嗗睍绀烘牸寮忚€屽噾娈点€傞粯璁ゅ彧杈撳嚭 [[evidence]] 1 娈碉紱褰撻棶棰橀渶瑕佽涔夋湭鐩存帴灞曞紑鐨勫墠缃煡璇嗐€佸師鐞嗚В閲婃垨搴旂敤渚嬪瓙鏃讹紝鏈€澶氬啀澧炲姞 1 涓渶鐩稿叧鐨勮ˉ鍏呮銆?3. 涓€鑸棶棰樿緭鍑?1-2 娈碉紱璺ㄧ珷鑺?璺ㄨ棰?鑱旂綉/椋庨櫓杈圭晫绫诲鏉傞棶棰樻墠杈撳嚭 3 娈碉紱4 娈垫槸纭笂闄愶紝鍙兘鍦ㄧ敤鎴峰悓鏃舵槑纭姹傝瘉鎹€佸師鐞嗐€佸簲鐢ㄥ拰杈圭晫鏃朵娇鐢ㄣ€備弗绂佸浐瀹氳緭鍑?4 娈垫垨鎶?[[background]]銆乕[deep_dive]]銆乕[application]] 鍏ㄩ儴鍚屾椂鍒楀嚭銆?4. 姣忔棣栬蹇呴』鏄竴涓鏍囩锛歔[evidence]]銆乕[background]]銆乕[extension]]銆乕[deep_dive]]銆乕[application]]銆乕[boundary]]銆乕[offtopic]]銆傚悓涓€绉嶆鏍囩鏈€澶氬嚭鐜颁竴娆★紱[[offtopic]] 鍙兘鍗曠嫭浣跨敤銆?5. [[evidence]] 鍙啓褰撳墠璁蹭箟鐩存帴鏀寔鐨勫唴瀹癸紝姣忎釜浜嬪疄鐐瑰繀椤诲甫 inline 閿氱偣锛氬瓧骞?[t=05:46]銆佸叧閿抚 [F7]銆佺珷鑺?[Ch3]銆傛椂闂村繀椤诲啓鎴愬垎閽?绉掞紝绔犺妭鍜屽叧閿抚缂栧彿涓嶈鍔犲皷鎷彿銆?6. [[background]] 鍙敤浜庤ˉ鍏呰涔夊亣璁剧敤鎴峰凡鐭ョ殑鍓嶇疆鐭ヨ瘑锛沎[extension]] 鍙敤浜庤法绔犺妭銆佽法瑙嗛鎴栬仈缃戝弬鐓э紱[[deep_dive]] 鍙敤浜庤В閲婃満鍒跺師鐞嗭紱[[application]] 鍙湪鐢ㄦ埛闂€滄€庝箞鐢?涓句緥/杩佺Щ鈥濇椂浣跨敤锛沎[boundary]] 鍙湪璁蹵箟璇佹嵁涓嶈冻銆佹帹鏂湁椋庨櫓鎴栭棶棰樻秹鍙婂畨鍏?鍋ュ悍/鎶曡祫绛夎竟鐣屾椂浣跨敤銆?7. 鑳屾櫙銆佸欢浼搞€佹繁鎸栥€佸簲鐢ㄥ拰杈圭晫娈典笉寮哄埗褰撳墠璁蹵箟閿氱偣锛屼絾蹇呴』鏄庣‘鍝簺鍐呭鏄€滆涔夌洿鎺ヨ瘉鎹€濓紝鍝簺鏄€滃熀浜庤涔変富棰樼殑瀛︿範琛ュ厖鈥濄€備笉瑕佹妸琛ュ厖鐭ヨ瘑浼鎴愯涔夊師璇濄€?8. 鑻ヨ涔夋病鏈夌洿鎺ュ睍寮€鐢ㄦ埛闂殑鍓嶇疆鐭ヨ瘑锛屼笉瑕佺畝鍗曟嫆绛旓紱搴斿厛缁?[[evidence]] 璇存槑璁蹵箟濡備綍鎻愬埌鎴栧亣璁惧畠锛屽啀閫夋嫨 [[background]] 鎴?[[deep_dive]] 涓殑涓€涓仛瀛︿範鍚戣ˉ鍏ㄣ€?9. 鑻ヤ娇鐢ㄨ法瑙嗛妫€绱㈢粨鏋滐紝鍙湪 [[extension]] 涓紩鐢紝鏍煎紡鍐欎綔 [BV1xxxx 路 Ch2]锛涜嫢浣跨敤鑱旂綉缁撴灉锛屽彧鍦ㄩ潪 [[evidence]] 娈靛紩鐢紝鏍煎紡鍐欎綔 [web 路 example.com]銆?
宸ュ叿浣跨敤瑙勫垯锛?1. 浠讳綍闇€瑕佸叿浣撳瓧骞曘€佺珷鑺傜粏鑺傘€佸叧閿抚銆佺煡璇嗗崟鍏冪殑鍥炵瓟锛屽繀椤诲厛璋冪敤宸ュ叿鑾峰彇褰撳墠璁蹵箟璇佹嵁銆?2. 涓ョ鍑┖寮曠敤鏃堕棿鎴虫垨瀛楀箷鍘熸枃銆?3. 宸ュ叿璋冪敤棰勭畻鏈夐檺锛堟渶澶?{max_tool_calls} 娆★級锛岃浼樺厛璋?search_lecture 鑱氬綋鍓嶈涔夎瘉鎹紝鍐嶇敤 get_chapter / get_frame / get_quote_context / explain_frame 鍋氬畾鐐硅ˉ鍏咃紱闇€瑕佽法瑙嗛鍙傜収鏃跺彲璋冪敤 search_lectures锛涘彧鏈夌敤鎴峰紑鍚仈缃戞椂鎵嶄細鎻愪緵 web_search銆?"""


COPILOT_SYSTEM = COPILOT_SYSTEM + (
    "\n绗竴琛屽繀椤绘槸 [[evidence]]"
    "\n榛樿鍙緭鍑?[[evidence]] 1 娈?"
    "\n涓ョ鍥哄畾杈撳嚭 4 娈?"
    "\n第一行必须是 [[evidence]]"
    "\n默认只输出 [[evidence]] 1 段"
    "\n严禁固定输出 4 段"
    "\nDefault tool path: search_lecture -> get_note_unit -> search_evidence -> get_evidence_object."
    "\nCopilot is note-first, evidence-second. Use search_lecture to establish the current Lecture Note IR mainline, then get_note_unit to read the owning note node, then search_evidence / get_evidence_object to drill into Evidence Index proof."
    "\nUse search_lectures only for cross-lecture extension. Compatibility tools (get_chapter, get_frame, get_quote_context, explain_frame) are supplements, not the default path."
    "\nDo not treat get_chapter / get_frame / get_quote_context / explain_frame as the normal follow-up after search_lecture; prefer note/evidence-native drilldown unless the user explicitly needs a compatibility anchor."
    "\nOnly use compatibility tools when the user explicitly asks for chapter/frame/quote anchors or when an existing [Ch]/[F]/[t] anchor needs fallback inspection."
    "\nTreat chapter/frame references as compatibility anchors that should be bridged back into note/evidence-native structure whenever possible."
    "\nWhen compatibility tools are not needed, they may be omitted entirely from the available tool list."
)


ANCHOR_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "t": re.compile(r"\[t=(\d{1,2}:\d{2}|\d+)\]"),
    "F": re.compile(r"\[F(\d+)\]"),
    "Ch": re.compile(r"\[(?:Ch|CH)(\d+)\]"),
}


def _truncate_reference_text(text: str, *, limit: int) -> str:
    clean = text.strip().replace("\n", " ")
    if len(clean) > limit:
        return clean[:limit] + "..."
    return clean


def format_references_block(references: list[dict] | None) -> str:
    """Render ``references`` into a human-readable bullet list."""
    if not references:
        return "（无）"
    lines: list[str] = []
    for i, ref in enumerate(references, start=1):
        kind = str(ref.get("kind", "")).strip().lower()
        if kind == "chapter":
            idx = ref.get("id") or ref.get("chapter_idx") or ref.get("index")
            lines.append(f"{i}. 引用兼容章节锚点 [Ch{idx}]")
        elif kind == "frame":
            idx = ref.get("id") or ref.get("frame_id")
            lines.append(f"{i}. 引用兼容关键帧锚点 [F{idx}]")
        elif kind in ("note", "note_node"):
            idx = ref.get("note_node_id") or ref.get("id")
            text = _truncate_reference_text(str(ref.get("text", "")), limit=120)
            line = f"{i}. 引用讲义节点 [{idx}]"
            if text:
                line += f"：{text!r}"
            lines.append(line)
        elif kind in ("evidence", "evidence_object"):
            idx = ref.get("evidence_id") or ref.get("id")
            text = _truncate_reference_text(str(ref.get("text", "")), limit=120)
            line = f"{i}. 引用证据对象 [{idx}]"
            if text:
                line += f"：{text!r}"
            lines.append(line)
        elif kind == "selection":
            text = _truncate_reference_text(str(ref.get("text", "")), limit=200)
            lines.append(f"{i}. 用户选中文本：{text!r}")
        else:
            lines.append(f"{i}. {ref!r}")
    return "\n".join(lines)
