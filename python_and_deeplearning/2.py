# -*- coding: utf-8 -*-

import re, csv, os, json, time, random, argparse
from typing import List, Dict, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------- Data sources ----------
# Fandom MediaWiki API (designed for bots; safer than scraping HTML pages)
WIKI_API = "https://counterstrike.fandom.com/api.php"
WIKI_CASE_CATEGORY = "Category:Counter-Strike:_Global_Offensive_Weapon_Cases"  # all weapon cases live here :contentReference[oaicite:1]{index=1}

# SteamDT OpenAPI (official; use it only for enrichment / prices / marketHashName)
STEAMDT_BASE = "https://open.steamdt.com"

RARITY_KEEP = ("Covert","Classified","Restricted","Gold")
RARITY_MAP = {"Covert":"隐秘级","Classified":"保密级","Restricted":"受限级","Gold":"金色（罕见特殊）"}

HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; cs2-roster/3.0)"}

# ---------- Utils ----------
def wiki_api(params: Dict) -> Dict:
    p = {"format":"json"}; p.update(params)
    for k in range(4):
        r = requests.get(WIKI_API, params=p, headers=HEADERS, timeout=30)
        if r.status_code==200: return r.json()
        time.sleep(1.2*(k+1))
    raise RuntimeError(f"Wiki API failure: {params}")

def get_all_case_pages() -> List[Dict]:
    """List all weapon-case pages from the Fandom category (no HTML scraping)."""
    # https://www.mediawiki.org/wiki/API:Categorymembers  :contentReference[oaicite:2]{index=2}
    out, cmcontinue = [], None
    while True:
        params = {
            "action":"query","list":"categorymembers","cmtitle":WIKI_CASE_CATEGORY,
            "cmlimit":"500","cmtype":"page"
        }
        if cmcontinue: params["cmcontinue"] = cmcontinue
        j = wiki_api(params)
        out.extend(j.get("query",{}).get("categorymembers",[]))
        cmcontinue = j.get("continue",{}).get("cmcontinue")
        if not cmcontinue: break
        time.sleep(0.25)
    return out

def parse_case_html(html: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse one case page HTML into items:
      - Covert / Classified / Restricted  -> guns (weapon | finish)
      - Gold -> knives/gloves variants
    Fandom 使用 wikia-gallery 结构，稀有度通过 CSS class 标识
    """
    soup = BeautifulSoup(html, "html.parser")
    items, errors = [], []
    
    # CSS class 到稀有度的映射
    rarity_class_map = {
        'rare': 'Covert',
        'mythical': 'Classified',
        'ancient': 'Restricted',
        'legendary': 'Gold'
    }
    
    # 查找所有 gallery 项目
    galleries = soup.find_all('div', class_='wikia-gallery-item')
    for g in galleries:
        caption = g.find('div', class_='lightbox-caption')
        if not caption:
            continue
        
        # 获取稀有度
        rarity_span = caption.find('span', class_=lambda x: x and any(r in x for r in rarity_class_map.keys()))
        if not rarity_span:
            continue
        
        rarity_classes = [c for c in rarity_span.get('class', []) if c in rarity_class_map]
        if not rarity_classes:
            continue
        rarity_en = rarity_class_map[rarity_classes[0]]
        
        # 获取物品名称（格式：武器 - 涂装）
        item_text = caption.get_text(' ', strip=True)
        if not item_text:
            continue
        
        # 转换为标准格式（武器 | 涂装）
        item_name = item_text.replace(' - ', ' | ')
        w, f = _split_weapon_finish(item_name)
        items.append(_mk_row(item_name, rarity_en, w, f))
    
    # 旧的基于标题的解析（保留作为备用）
    sections = soup.select("h2, h3")
    def harvest_between(start, end, rarity_en):
        # 抓取 start 到 end 之间的 <ul><li> 或 .wikitable 的文本链接
        blocks = []
        node = start.next_sibling
        while node and node != end:
            if getattr(node, "name", None) in ("ul","ol"):
                blocks.append(node)
            if getattr(node, "name", None) == "table" and "wikitable" in (node.get("class") or []):
                blocks.append(node)
            node = node.next_sibling
        for blk in blocks:
            # li 列表
            for li in blk.select("li"):
                txt = li.get_text(" ", strip=True)
                if not txt: continue
                name = _normalize_item_name(txt)
                if name:
                    w, f = _split_weapon_finish(name)
                    items.append(_mk_row(name, rarity_en, w, f))
            # wikitable（按行第一列超链接）
            for tr in blk.select("tr"):
                a = tr.find("a")
                if a and a.get_text(strip=True):
                    name = _normalize_item_name(a.get_text(strip=True))
                    if name:
                        w, f = _split_weapon_finish(name)
                        items.append(_mk_row(name, rarity_en, w, f))

    # 遍历小节，定位“Covert / Classified / Restricted / Knives / Gloves”
    for i, h in enumerate(sections):
        title = h.get_text(" ", strip=True)
        ttl = title.lower()
        nxt = None
        if i+1 < len(sections): nxt = sections[i+1]
        if "covert" in ttl:
            harvest_between(h, nxt, "Covert")
        elif "classified" in ttl:
            harvest_between(h, nxt, "Classified")
        elif "restricted" in ttl:
            harvest_between(h, nxt, "Restricted")
        elif "knives" in ttl or "gloves" in ttl or "rare special" in ttl or "gold" in ttl:
            # 视为金位
            harvest_between(h, nxt, "Gold")
    if not items:
        errors.append("no-items-detected")
    return items, errors

def _normalize_item_name(s: str) -> str:
    # 通常写成 “AK-47 | Case Hardened”，有时会附加说明，尽量清理
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉括号内的纯注释词，如 (StatTrak) / (Souvenir) / (Factory New) 等
    s = re.sub(r"\s\((StatTrak|Souvenir|Factory New|Minimal Wear|Field-Tested|Well-Worn|Battle-Scarred)\)$","", s, flags=re.I)
    # 常见分隔符变体
    s = s.replace(" – ", " | ").replace(" — ", " | ").replace(" - ", " | ")
    return s

_WEAPONS = {"AK-47","AUG","AWP","CZ75-Auto","Desert Eagle","Dual Berettas","FAMAS","Five-SeveN","G3SG1",
            "Galil AR","Glock-18","M249","M4A1-S","M4A4","MAC-10","MAG-7","MP5-SD","MP7","MP9","Negev",
            "Nova","P2000","P250","P90","PP-Bizon","R8 Revolver","SCAR-20","SG 553","SSG 08","Sawed-Off",
            "Tec-9","UMP-45","USP-S","XM1014","Zeus x27"}
def _split_weapon_finish(name: str) -> Tuple[str,str]:
    # “Weapon | Finish” 或 “Weapon Finish” 两种都尽量兼容
    if " | " in name:
        parts = name.split(" | ", 1)
        return parts[0], parts[1]
    for w in sorted(_WEAPONS, key=len, reverse=True):
        if name.startswith(w + " "): return w, name[len(w)+1:]
        if name == w: return w, ""
    return "", ""

def _mk_row(item_name_en, rarity_en, weapon, finish):
    return {
        "case_name_en":"", "case_name_zh":"", "release_date":"",  # 填充在上层
        "rarity_en":rarity_en, "rarity_zh":RARITY_MAP.get(rarity_en,""),
        "item_name_en":item_name_en, "item_name_zh":"",
        "weapon":weapon, "finish":finish,
        "source_url":""
    }

# ---------- SteamDT enrichment (optional) ----------
def fetch_steamdt_base(api_key: str, cache_path="steamdt_base.json") -> pd.DataFrame:
    # 官方OpenAPI：/open/cs2/v1/base（每日一次足够）。:contentReference[oaicite:3]{index=3}
    if os.path.exists(cache_path):
        try:
            j = json.load(open(cache_path,"r",encoding="utf-8"))
            return pd.json_normalize(j.get("data",[]))
        except Exception:
            pass
    h = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(f"{STEAMDT_BASE}/open/cs2/v1/base", headers=h, timeout=40)
    r.raise_for_status()
    j = r.json()
    json.dump(j, open(cache_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return pd.json_normalize(j.get("data",[]))

def _norm_mhn(s: str) -> str:
    if not isinstance(s,str): return ""
    s = s.replace("StatTrak™ ","").replace("Souvenir ","")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r" \((Factory New|Minimal Wear|Field-Tested|Well-Worn|Battle-Scarred)\)$","", s, flags=re.I)
    return s

def enrich_with_steamdt(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    base = fetch_steamdt_base(api_key)
    base["norm"] = base.apply(lambda r: _norm_mhn(r.get("marketHashName") or r.get("name") or ""), axis=1)
    base = base[base["norm"]!=""].drop_duplicates(subset=["norm"])
    def match(row):
        w, f = (row.get("weapon") or "").strip(), (row.get("finish") or "").strip()
        base_name = (w + (" | " if w and f else "") + f).strip()
        norm = _norm_mhn(base_name)
        hit = base[base["norm"]==norm]
        return (hit.iloc[0]["marketHashName"] if len(hit)>0 else "")
    df["steamdt_market_hash_name"] = df.apply(match, axis=1)
    return df

# ---------- Main flow ----------
def main(out_items="cs2_case_items_full.csv", out_cases="cs2_cases.csv", link_steamdt=False):
    # 1) 拉所有武器箱页面
    members = get_all_case_pages()
    cases_rows, all_rows = [], []

    for i, m in enumerate(members, 1):
        pageid, title = m["pageid"], m["title"]
        # 2) 用 action=parse 拿到该页面的 HTML（只解析正文，不含皮肤/导航外壳） :contentReference[oaicite:4]{index=4}
        j = wiki_api({"action":"parse","page":title,"prop":"text|sections"})
        html = j.get("parse",{}).get("text",{}).get("*","")
        items, errs = parse_case_html(html)

        # 尝试从简介中抽发行日期（并不强制）
        release_date = ""
        if "sections" in j.get("parse",{}):
            # 有的页面在 lead 里会写，比如 "... released on February 6, 2024"
            lead_html = html[: html.find("<h2")] if "<h2" in html else html
            mdate = re.search(r"released on ([A-Za-z]+ \d{1,2}(?:st|nd|rd|th)?, \d{4})", BeautifulSoup(lead_html,"html.parser").get_text(" ",strip=True))
            if mdate: release_date = mdate.group(1)

        # 填公共字段
        for r in items:
            r["case_name_en"] = title.replace(" Case"," Case")  # 保持原名
            r["release_date"] = release_date
            r["source_url"] = f"https://counterstrike.fandom.com/wiki/{title.replace(' ','_')}"
        cases_rows.append({"case_name_en": title, "case_name_zh":"", "release_date":release_date,
                           "url": f"https://counterstrike.fandom.com/wiki/{title.replace(' ','_')}"})
        all_rows.extend(items)
        print(f"[{i}/{len(members)}] {title}: {len(items)} items {'('+';'.join(errs)+')' if errs else ''}")
        time.sleep(0.25 + random.random()*0.15)

    items_df = pd.DataFrame(all_rows)
    cases_df = pd.DataFrame(cases_rows)

    if link_steamdt:
        api_key = os.getenv("STEAMDT_API_KEY","").strip()
        if api_key:
            try:
                items_df = enrich_with_steamdt(items_df, api_key)
            except Exception as e:
                print("SteamDT enrichment failed:", e)
        else:
            print("WARN: STEAMDT_API_KEY not set; skip SteamDT enrichment.")

    # 输出（保持你之前的字段）
    items_cols = ["case_name_en","case_name_zh","release_date","rarity_en","rarity_zh",
                  "item_name_en","item_name_zh","weapon","finish","source_url","steamdt_market_hash_name"]
    for c in items_cols:
        if c not in items_df.columns: items_df[c] = ""
    items_df = items_df[items_cols]

    cases_df.to_csv(out_cases, index=False, encoding="utf-8-sig")
    items_df.to_csv(out_items, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_items}, {out_cases}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_items", nargs="?", default="cs2_case_items_full.csv")
    ap.add_argument("out_cases", nargs="?", default="cs2_cases.csv")
    ap.add_argument("--link-steamdt", action="store_true", help="enrich with SteamDT base (env STEAMDT_API_KEY)")
    args = ap.parse_args()
    main(args.out_items, args.out_cases, link_steamdt=args.link_steamdt)
