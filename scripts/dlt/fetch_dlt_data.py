import os
import requests
import pandas as pd
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

name_path = {"dlt": {"name": "大乐透", "path": "./"}}
data_file_name = "dlt_history.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://static.sporttery.cn/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

def fetch_from_sporttery():
    """从体彩官方API抓取大乐透历史数据"""
    all_records = []
    page_no = 1
    page_size = 100
    max_pages = 70

    for page_no in range(1, max_pages + 1):
        url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
        params = {
            "gameNo": "85",
            "provinceId": "0",
            "pageSize": str(page_size),
            "isVerify": "1",
            "pageNo": str(page_no),
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logging.warning(f"API返回状态码: {resp.status_code}")
                break
            data = resp.json()
            if data.get("errorCode") != "0":
                logging.warning(f"API返回异常: {data.get('errorMsg', '未知错误')}")
                break
            items = data.get("value", {}).get("list", [])
            if not items:
                break
            for item in items:
                draw_num = item.get("lotteryDrawNum", "")
                result_str = item.get("lotteryDrawResult", "")
                nums = [int(x) for x in result_str.split() if x.isdigit()]
                if len(nums) == 7:
                    all_records.append({
                        "期数": int(draw_num),
                        "红球_1": nums[0], "红球_2": nums[1], "红球_3": nums[2],
                        "红球_4": nums[3], "红球_5": nums[4],
                        "蓝球_1": nums[5], "蓝球_2": nums[6],
                    })
            if len(items) < page_size:
                break
            page_no += 1
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            logging.warning(f"网络请求失败(第{page_no}页): {e}")
            break
        except Exception as e:
            logging.warning(f"解析失败(第{page_no}页): {e}")
            break

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    df = df.sort_values(by="期数", ascending=False).reset_index(drop=True)
    logging.info(f"体彩API共获取 {len(df)} 条大乐透数据")
    return df

def fetch_dlt_data():
    name = "dlt"
    logging.info(f"开始获取{name_path[name]['name']}数据...")
    df = fetch_from_sporttery()
    if df is None:
        logging.warning("所有数据源均失败，无法获取大乐透数据")
        return False
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, encoding="gbk", index=False)
    logging.info(f"数据已保存至 {save_path}，共 {len(df)} 条")
    return True

def main():
    return fetch_dlt_data()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
