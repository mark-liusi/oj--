#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 https://api.steamdt.com/user/ranking/v1/page 拉数据，
解析出 marketHashName 和 当前卖价 sellPriceInfoVO.price，
然后放进一个 dict 里，后面可以喂给你的炼金公式。

注意：
1. 这里我假设 price 是接口里看到的整数，比如 102，如果你在网页上看到是 1.02，
   那就说明要 /100，一会儿下面有地方可以调。
2. 这个接口是分页的，我给你写了翻页参数 pageNum/pageSize，
   你看你需要多少条就拉多少条。
"""

import time
import requests
import datetime


class SteamDTClient:
    def __init__(self, timeout=5):
        self.base_url = "https://api.steamdt.com"
        self.timeout = timeout
        self.sess = requests.Session()
        # 这些头基本照你浏览器里来的
        self.base_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://steamdt.com",
            "Referer": "https://steamdt.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Language": "zh_CN",
            "X-App-Version": "1.0.0",
            "X-Currency": "CNY",
            "X-Device": "1",
            # 这个设备id你那边是一个uuid，我随便写了一个；你也可以直接用你浏览器那串
            "X-Device-Id": "73576c95-d446-4e71-9dcf-3f0cd7b9f749",
            # 浏览器里是 Access-Token: undefined，那我也照抄
            "Access-Token": "undefined",
        }

    def _now_ms(self):
        return int(time.time() * 1000)

    def fetch_page(self, page_num=1, page_size=50):
        """
        真正的请求：POST https://api.steamdt.com/user/ranking/v1/page?timestamp=xxxx
        body 里也要有同样的结构
        """
        ts = self._now_ms()
        url = f"{self.base_url}/user/ranking/v1/page"
        params = {
            "timestamp": ts
        }
        # 这个 payload 就是你截图里的那坨，我补了一点常识性的字段
        payload = {
            "page": page_num,
            "pageSize": page_size,
            "nextId": "",
            "dataField": "priceRate",      # 你截图就是这个
            "dataRange": "ONE_DAY",        # 一天的数据
            "folder": None,
            # 你截图里只看到一半：[{field: "price", ...}]，我先给你一个最简单的
            "rangeConditionList": [
                {"field": "price"}
            ],
            "sortType": "DESC",
            "timestamp": ts
        }

        resp = self.sess.post(
            url,
            params=params,
            json=payload,
            headers=self.base_headers,
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_all_once(self, max_pages=3, page_size=50):
        all_prices = {}
        for p in range(1, max_pages+1):
            data = self.fetch_page(page_num=p, page_size=page_size)
            print("DEBUG page", p, data)  # 先打印看看是不是这次有list了
            data_obj = data.get("data", {})
            items = data_obj.get("list", [])
            if not items:
                break
            for it in items:
                item_info = it.get("itemInfoVO", {})
                name = item_info.get("marketHashName")
                if not name:
                    continue
                price_info = it.get("sellPriceInfoVO", {})
                price = price_info.get("price")
                if price is None:
                    continue
                price = float(price)
                # 如果你发现网页上是 1.02 这里是 102，就开这一行
                # price = price / 100.0
                all_prices[name] = price
            time.sleep(0.2)
        return all_prices


# ======================== 炼金那块示意 ========================

# 你自己的炼金配方表，还跟之前一样
RECIPES = {
    "M4A1 | 暴怒野兽": {
        "inputs": [
            {"name": "M4A1 | 暴怒野兽下级A", "count": 10},
        ],
        "outputs": [
            {"name": "M4A1 | 暴怒野兽", "prob": 1.0}
        ]
    },
}

def calc_ev_for_recipe(recipe_name:str, recipe:dict, latest_prices:dict)->dict:
    missing = []
    total_input_cost = 0.0
    for inn in recipe["inputs"]:
        n = inn["name"]
        c = inn["count"]
        p = latest_prices.get(n, -1.0)
        if p < 0:
            missing.append(n)
        total_input_cost += max(p, 0) * c

    total_output_ev = 0.0
    for out in recipe["outputs"]:
        n = out["name"]
        prob = out["prob"]
        p = latest_prices.get(n, -1.0)
        if p < 0:
            missing.append(n)
        total_output_ev += max(p, 0) * prob

    return {
        "recipe": recipe_name,
        "input_cost": total_input_cost,
        "output_ev": total_output_ev,
        "profit": total_output_ev - total_input_cost,
        "ok": len(missing) == 0,
        "missing": list(set(missing)),
    }


def main():
    client = SteamDTClient()
    # 1. 先拉几页，把现在能看到的大部分皮都收了
    prices = client.fetch_all_once(max_pages=5, page_size=50)
    print(f"[{datetime.datetime.now()}] 共抓到 {len(prices)} 条价格")

    # 2. 打印几条看看格式
    i = 0
    for name, price in prices.items():
        print(name, price)
        i += 1
        if i >= 5:
            break

    # 3. 用这批价格去算我们的炼金配方
    print("\n=== 炼金EV ===")
    for rname, r in RECIPES.items():
        info = calc_ev_for_recipe(rname, r, prices)
        if not info["ok"]:
            print(f"{rname} 缺价格: {info['missing']}")
            continue

        # 给一点手续费/滑点
        safe_profit = info["profit"] - 0.03 * info["input_cost"]
        print(f"{rname}: 成本={info['input_cost']:.2f}, 期望={info['output_ev']:.2f}, "
              f"盈亏={info['profit']:.2f}, 扣安全垫后={safe_profit:.2f}")
        if safe_profit > 0:
            print("  >>> 现在能搞")

if __name__ == "__main__":
    main()
