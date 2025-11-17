
# -*- coding: utf-8 -*-
"""
connectors_market.py
====================
提供“带磨损约束的最低可成交价”抓取器（可选 CSFloat / BUFF / YouPin），并内置混合回退逻辑。

设计目标：
- 只对“候选主料 Top-N”做外部抓取，减少调用量；
- 有网用真实接口；没网/接口不可用时，稳健回退到本地价格 + 溢价模型；
- 不强绑定某一平台：接口 URL、KEY、Cookies 都可通过环境变量或构造参数传入。

⚠️ 注意：
- 这些第三方接口可能有频率限制与登录校验。生产使用时请加缓存与速率控制（此处给出基本实现）。
- BUFF / YouPin 并无公开稳定文档，字段或 URL 有可能变更；保留 CSFloat Inspect 兜底链路可保证磨损可得。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import time, os, json, math
import requests
import pandas as pd
from pathlib import Path

# ------------------------------
# 名称映射表加载
# ------------------------------
_NAME_MAPPING: Optional[Dict[str, str]] = None

def _load_name_mapping(csv_path: str = "data/name_mapping.csv") -> Dict[str, str]:
    """加载名称映射表：name → market_hash_name"""
    global _NAME_MAPPING
    if _NAME_MAPPING is not None:
        return _NAME_MAPPING
    
    mapping = {}
    path = Path(csv_path)
    if path.exists():
        try:
            df = pd.read_csv(path, encoding="utf-8")
            if "name" in df.columns and "market_hash_name" in df.columns:
                for _, row in df.iterrows():
                    name = str(row["name"]).strip()
                    market_name = str(row["market_hash_name"]).strip()
                    if name and market_name:
                        mapping[name.casefold()] = market_name
                print(f"✓ 已加载名称映射表：{len(mapping)} 条规则（{csv_path}）")
        except Exception as e:
            print(f"⚠ 加载名称映射表失败：{e}")
    
    _NAME_MAPPING = mapping
    return mapping

def _get_market_name(item_name: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """获取市场搜索名称（优先使用映射表）"""
    if mapping is None:
        mapping = _load_name_mapping()
    
    key = item_name.strip().casefold()
    return mapping.get(key, item_name)  # 找不到映射则返回原名

# ------------------------------
# 小工具：稳健请求（带基本重试）
# ------------------------------
def _http_get(url: str, params: Dict[str,Any]|None=None, headers: Dict[str,str]|None=None, cookies: Dict[str,str]|None=None, timeout: float=10.0, retries: int=2) -> Optional[requests.Response]:
    err = None
    for i in range(max(1, retries)):
        try:
            resp = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=timeout)
            if resp.status_code == 200:
                return resp
            # 429/5xx: 稍等再试
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.8 * (i+1))
                continue
            # 其它状态码直接放弃
            err = RuntimeError(f"GET {url} status={resp.status_code} text={resp.text[:200]}")
            break
        except Exception as e:
            err = e
            time.sleep(0.6 * (i+1))
    return None

# ------------------------------
# CSFloat
# ------------------------------
@dataclass
class CSFloatConfig:
    market_api: str = os.getenv("CSFLOAT_MARKET_API", "https://csfloat.com/api/v1/listings")
    inspect_api: str = os.getenv("CSFLOAT_INSPECT_API", "https://api.csfloat.com/")  # 兼容常见反代
    api_key: Optional[str] = os.getenv("CSFLOAT_API_KEY")  # 市场 API 若需 key 可放此处

class CSFloatClient:
    """CSFloat：优先使用 Market API 的浮漂过滤；也可用 Inspect API 通过 inspect link 取 float。"""
    def __init__(self, cfg: Optional[CSFloatConfig]=None):
        self.cfg = cfg or CSFloatConfig()

    def get_lowfloat_listing_price(self, market_hash_name: str, max_float: float, limit: int=50) -> Optional[float]:
        """
        直接从 CSFloat 市场拉"float ≤ max_float"的最低价。
        注意：部分部署可能需要 X-API-Key；本函数支持从环境变量读取。
        """
        # 使用名称映射表
        search_name = _get_market_name(market_hash_name)
        
        # 检查缓存
        cache_key = f"csfloat:{search_name}:{max_float:.4f}"
        cached = _global_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # 请求节流
        _global_rate_limiter.wait()
        
        params = {
            "market_hash_name": search_name,
            "sort_by": "lowest_price",
            "order": "asc",
            "max_float": f"{max(0.0, min(1.0, max_float)):.6f}",
            "limit": str(limit),
        }
        headers = {}
        if self.cfg.api_key:
            headers["X-API-Key"] = self.cfg.api_key
        resp = _http_get(self.cfg.market_api, params=params, headers=headers, timeout=10, retries=2)
        if not resp:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        # 期望结构：{ "data": [ {"price": 12.34, "float_value": 0.0123, ...}, ... ] }
        items = data.get("data") or data.get("results") or []
        best = None
        for it in items:
            fl = it.get("float_value") or it.get("float") or it.get("paintwear")
            pr = it.get("price") or it.get("unit_price") or it.get("converted_price")
            try:
                if fl is None or pr is None:
                    continue
                if float(fl) <= max_float:
                    best = min(best, float(pr)) if best is not None else float(pr)
            except Exception:
                continue
        # 保存到缓存
        if best is not None:
            _global_cache.set(cache_key, best)
        return best

    def inspect_float(self, inspect_link: str) -> Optional[float]:
        """
        通过 Inspect API 获取磨损。多种后端兼容：?url= / ?s=1&url=
        """
        # 常见两种形式：<api>?url=INSPECT 或 <api>?s=1&url=INSPECT
        for q in ("url", "s=1&url"):
            url = self.cfg.inspect_api.rstrip("/") + "/"
            sep = "&" if "?" in url else "?"
            resp = _http_get(url, params=None, headers=None, timeout=10)
            # 某些部署要求直接拼接 query；退而求其次尝试直接构造
            full = f"{self.cfg.inspect_api}?{q}={inspect_link}"
            resp = _http_get(full, timeout=10, retries=2)
            if not resp:
                continue
            try:
                j = resp.json()
            except Exception:
                continue
            # 兼容常见字段
            item = j.get("iteminfo") or j.get("data") or j
            for k in ("floatvalue","float_value","paintwear","wear"):
                if k in item:
                    try:
                        return float(item[k])
                    except Exception:
                        pass
        return None

# ------------------------------
# BUFF (需要登录态 Cookie)
# ------------------------------
@dataclass
class BuffConfig:
    goods_search_api: str = os.getenv("BUFF_GOODS_API", "https://buff.163.com/api/market/goods")
    sell_order_api: str = os.getenv("BUFF_SELL_ORDER_API", "https://buff.163.com/api/market/goods/sell_order")
    cookies_json: Optional[str] = os.getenv("BUFF_COOKIES_JSON")  # 路径或直接 JSON 字符串
    csrf: Optional[str] = os.getenv("BUFF_CSRF")  # 可选

def _load_cookies(cookies_json: Optional[str]) -> Dict[str,str]:
    if not cookies_json:
        return {}
    try:
        # 既支持路径也支持直接 JSON 文本
        if os.path.exists(cookies_json):
            with open(cookies_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(cookies_json)
    except Exception:
        return {}

class BuffClient:
    def __init__(self, cfg: Optional[BuffConfig]=None):
        self.cfg = cfg or BuffConfig()
        self.cookies = _load_cookies(self.cfg.cookies_json)

    def _find_goods_id(self, market_hash_name: str) -> Optional[int]:
        # 使用名称映射表
        search_name = _get_market_name(market_hash_name)
        
        # 请求节流
        _global_rate_limiter.wait()
        
        params = {"game":"csgo", "search": search_name, "page_num":"1", "page_size":"50"}
        resp = _http_get(self.cfg.goods_search_api, params=params, cookies=self.cookies, timeout=10, retries=2)
        if not resp:
            return None
        try:
            data = resp.json()
            items = (data.get("data") or {}).get("items") or []
            for it in items:
                # 兼容 name / market_hash_name 字段
                if search_name.lower() in (it.get("market_hash_name","") or it.get("name","")).lower():
                    return it.get("id") or it.get("goods_id")
        except Exception:
            return None
        return None

    def get_lowfloat_listing_price(self, market_hash_name: str, max_float: float, limit: int=50) -> Optional[float]:
        # 使用名称映射表
        search_name = _get_market_name(market_hash_name)
        
        # 检查缓存
        cache_key = f"buff:{search_name}:{max_float:.4f}"
        cached = _global_cache.get(cache_key)
        if cached is not None:
            return cached
        
        gid = self._find_goods_id(search_name)
        if not gid:
            return None
        params = {
            "game":"csgo", "goods_id": str(gid), "page_num": "1", "page_size": str(limit),
            "sort_by": "paintwear.asc",  # 低磨损优先
        }
        headers = {}
        if self.cfg.csrf:
            headers["x-csrf-token"] = self.cfg.csrf
        resp = _http_get(self.cfg.sell_order_api, params=params, headers=headers, cookies=self.cookies, timeout=10, retries=2)
        if not resp:
            return None
        try:
            data = resp.json()
            items = (data.get("data") or {}).get("items") or []
        except Exception:
            return None
        best = None
        for it in items:
            fl = it.get("paintwear") or it.get("float") or it.get("wear")
            pr = it.get("price") or it.get("unit_price")
            try:
                if fl is None or pr is None:
                    continue
                if float(fl) <= max_float:
                    best = min(best, float(pr)) if best is not None else float(pr)
            except Exception:
                continue
        # 保存到缓存
        if best is not None:
            _global_cache.set(cache_key, best)
        return best

# ------------------------------
# YouPin 898 (需要登录态)
# ------------------------------
@dataclass
class YouPinConfig:
    # 这些接口为“社区共识”，平台可能随时调整
    search_api: str = os.getenv("YOUPIN_SEARCH_API", "https://api.youpin898.com/api/homepage/commodity/search")
    list_api: str   = os.getenv("YOUPIN_LIST_API",   "https://api.youpin898.com/api/homepage/commodity/getcommodityonlist")
    cookies_json: Optional[str] = os.getenv("YOUPIN_COOKIES_JSON")

class YouPinClient:
    def __init__(self, cfg: Optional[YouPinConfig]=None):
        self.cfg = cfg or YouPinConfig()
        self.cookies = _load_cookies(self.cfg.cookies_json)

    def _find_commodity_id(self, market_hash_name: str) -> Optional[str]:
        # 简易搜索：返回最相似的一个
        # 使用名称映射表
        search_name = _get_market_name(market_hash_name)
        
        # 请求节流
        _global_rate_limiter.wait()
        
        params = {"key": search_name, "appid": 730, "pageIndex":1, "pageSize":20}
        resp = _http_get(self.cfg.search_api, params=params, cookies=self.cookies, timeout=10, retries=2)
        if not resp:
            return None
        try:
            data = resp.json()
            items = (data.get("Data") or data.get("data") or {}).get("List") or []
            for it in items:
                nm = it.get("CommodityName") or it.get("MarketHashName") or it.get("Name")
                cid = it.get("CommodityId") or it.get("Id")
                if not nm or not cid: 
                    continue
                if search_name.lower() in str(nm).lower():
                    return str(cid)
        except Exception:
            return None
        return None

    def get_lowfloat_listing_price(self, market_hash_name: str, max_float: float, limit: int=50) -> Optional[float]:
        # 使用名称映射表
        search_name = _get_market_name(market_hash_name)
        
        # 检查缓存
        cache_key = f"youpin:{search_name}:{max_float:.4f}"
        cached = _global_cache.get(cache_key)
        if cached is not None:
            return cached
        
        cid = self._find_commodity_id(search_name)
        if not cid:
            return None
        payload = {"pageIndex":1, "pageSize":limit, "appid":730, "commodityId": cid, "sortType": 8}  # 8:磨损升序（社区常用）
        try:
            resp = requests.post(self.cfg.list_api, json=payload, cookies=self.cookies, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None
        items = (data.get("Data") or data.get("data") or {}).get("List") or []
        best = None
        for it in items:
            fl = it.get("WearValue") or it.get("float") or it.get("PaintWear")
            pr = it.get("Price") or it.get("SellingPrice") or it.get("UnitPrice")
            try:
                if fl is None or pr is None:
                    continue
                if float(fl) <= max_float:
                    best = min(best, float(pr)) if best is not None else float(pr)
            except Exception:
                continue
        # 保存到缓存
        if best is not None:
            _global_cache.set(cache_key, best)
        return best

# ------------------------------
# 聚合抓取器：按优先级尝试，最后回退本地价格 + 溢价
# ------------------------------
@dataclass
class FetcherConfig:
    source: str = "hybrid"     # "local" / "csfloat" / "buff" / "youpin" / "hybrid"
    price_markup_lowfloat: float = 0.07  # 没有低漂样本时，对本地均价追加的保守溢价（7%）
    requests_limit_per_series: int = 3   # 每个 series 只对 Top‑N 候选做外部抓取

class MarketPriceFetcher:
    """
    统一入口：
    - get_lowfloat_price(market_hash_name, f_allow, local_price) -> float
    """
    def __init__(self, cfg: Optional[FetcherConfig]=None, csfloat: Optional[CSFloatClient]=None, buff: Optional[BuffClient]=None, youpin: Optional[YouPinClient]=None):
        self.cfg = cfg or FetcherConfig()
        self.csfloat = csfloat or CSFloatClient()
        self.buff = buff or BuffClient()
        self.youpin = youpin or YouPinClient()

    def _from_local(self, local_price: Optional[float]) -> Optional[float]:
        if local_price is None:
            return None
        return float(local_price) * (1.0 + max(0.0, self.cfg.price_markup_lowfloat))

    def get_lowfloat_price(self, market_hash_name: str, f_allow: float, local_price: Optional[float]) -> Optional[float]:
        src = (self.cfg.source or "hybrid").lower()
        # 单一源
        if src == "local":
            return self._from_local(local_price)
        if src == "csfloat":
            p = self.csfloat.get_lowfloat_listing_price(market_hash_name, f_allow)
            return p if p is not None else self._from_local(local_price)
        if src == "buff":
            p = self.buff.get_lowfloat_listing_price(market_hash_name, f_allow)
            return p if p is not None else self._from_local(local_price)
        if src == "youpin":
            p = self.youpin.get_lowfloat_listing_price(market_hash_name, f_allow)
            return p if p is not None else self._from_local(local_price)
        # 混合（推荐顺序：CSFloat → BUFF → YouPin → 本地）
        for fn in (
            lambda: self.csfloat.get_lowfloat_listing_price(market_hash_name, f_allow),
            lambda: self.buff.get_lowfloat_listing_price(market_hash_name, f_allow),
            lambda: self.youpin.get_lowfloat_listing_price(market_hash_name, f_allow),
        ):
            try:
                p = fn()
                if p is not None:
                    return p
            except Exception:
                continue
        return self._from_local(local_price)
