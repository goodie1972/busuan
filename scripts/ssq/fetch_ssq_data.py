import os
import requests
import pandas as pd
import sys
import logging
import time
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

name_path = {"ssq": {"name": "双色球", "path": "./"}}
data_file_name = "ssq_history.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.cwl.gov.cn/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def get_cwl_session():
    """先访问首页获取cookies"""
    try:
        resp = SESSION.get("https://www.cwl.gov.cn/", timeout=15)
        logging.info(f"cwl.gov.cn 首页状态码: {resp.status_code}")
        return True
    except Exception as e:
        logging.warning(f"获取cwl.gov.cn会话失败: {e}")
        return False

def normalize_period(code):
    """将官方期号(如2026102)转换为本地格式(如26102)"""
    s = str(code).strip()
    if len(s) >= 7 and s.isdigit():
        return int(s[1:])
    return int(s)

def fetch_from_cwl():
    """从福彩官方API抓取双色球数据"""
    get_cwl_session()
    all_records = []
    page_no = 1
    page_size = 100

    for page_no in range(1, 70):
        url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        params = {
            "name": "ssq",
            "pageNo": str(page_no),
            "pageSize": str(page_size),
            "systemType": "PC",
        }
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logging.warning(f"cwl.gov.cn API返回: {resp.status_code}")
                break
            data = resp.json()
            if data.get("state") != 0:
                logging.warning(f"API异常: {data.get('message', '')}")
                break
            items = data.get("result", [])
            if not items:
                break
            for item in items:
                code = item.get("code", "")
                red_str = item.get("red", "")
                blue_str = item.get("blue", "")
                if not code and item.get("lotteryDrawNum"):
                    code = item["lotteryDrawNum"]
                if not red_str and item.get("lotteryDrawResult"):
                    parts = item["lotteryDrawResult"].split()
                    if len(parts) >= 7:
                        red_str = " ".join(parts[:6])
                        blue_str = parts[6]
                reds = re.findall(r"\d+", str(red_str))
                blues = re.findall(r"\d+", str(blue_str))
                if len(reds) == 6 and len(blues) >= 1 and code:
                    all_records.append({
                        "期数": normalize_period(code),
                        "红球_1": int(reds[0]), "红球_2": int(reds[1]),
                        "红球_3": int(reds[2]), "红球_4": int(reds[3]),
                        "红球_5": int(reds[4]), "红球_6": int(reds[5]),
                        "蓝球": int(blues[0]),
                    })
            if len(items) < page_size:
                break
            page_no += 1
            time.sleep(0.3)
        except requests.exceptions.RequestException as e:
            logging.warning(f"cwl.gov.cn网络错误(第{page_no}页): {e}")
            break
        except Exception as e:
            logging.warning(f"cwl.gov.cn解析错误(第{page_no}页): {e}")
            break

    if all_records:
        logging.info(f"cwl.gov.cn获取 {len(all_records)} 条数据")
        df = pd.DataFrame(all_records)
        return df.sort_values(by="期数", ascending=False).reset_index(drop=True)
    return None

def fetch_from_500com():
    """从500.com备用源抓取"""
    all_records = []
    limit = 2000
    step = 200

    for start in range(1, limit, step):
        end = start + step - 1
        url = "https://datachart.500.com/ssq/history/newinc/history.php"
        params = {"start": str(start), "end": str(end)}
        try:
            resp = requests.get(
                url, params=params,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=30, verify=False
            )
            if resp.status_code != 200:
                logging.warning(f"500.com返回: {resp.status_code}")
                continue
            encodings = ["gb2312", "gbk", "utf-8"]
            html = None
            for enc in encodings:
                try:
                    resp.encoding = enc
                    html = resp.text
                    break
                except UnicodeDecodeError:
                    continue
            if not html:
                continue
            trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
            for tr in trs:
                tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                cleaned = [re.sub(r"<[^>]+>", "", td).strip() for td in tds]
                if len(cleaned) >= 9 and cleaned[1].isdigit() and len(cleaned[1]) >= 5:
                    try:
                        record = {"期数": int(cleaned[1])}
                        for i in range(6):
                            record[f"红球_{i+1}"] = int(cleaned[i+2])
                        record["蓝球"] = int(cleaned[8])
                        valid = all(1 <= record[f"红球_{i+1}"] <= 33 for i in range(6))
                        valid = valid and 1 <= record["蓝球"] <= 16
                        if valid:
                            all_records.append(record)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logging.warning(f"500.com请求失败: {e}")
            continue

    if all_records:
        logging.info(f"500.com获取 {len(all_records)} 条数据")
        df = pd.DataFrame(all_records)
        return df.drop_duplicates(subset=["期数"]).sort_values(by="期数", ascending=False).reset_index(drop=True)
    return None

def has_sufficient_data():
    """检查已有数据是否足够"""
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_name)
    if os.path.exists(save_path):
        try:
            df = pd.read_csv(save_path, encoding="gbk")
            if len(df) > 500:
                logging.info(f"本地已有 {len(df)} 条数据，跳过抓取")
                return True
        except:
            pass
    return False

def fetch_ssq_data():
    name = "ssq"
    logging.info(f"开始获取{name_path[name]['name']}数据...")

    # 注意：这里不做"数据已足够则跳过"的判断，否则本地数据一旦超过500条就永远不会更新。
    # has_sufficient_data 仅用于所有数据源都失败时的兜底。

    df = fetch_from_cwl()
    if df is None:
        df = fetch_from_500com()
    if df is None:
        logging.warning("所有数据源均失败")
        if has_sufficient_data():
            return True
        return False

    # 与本地数据合并去重后保存：官方API仅保留2013年至今的数据，
    # 直接覆盖会丢失更早的历史记录，因此以新抓取数据为准、按"期数"去重合并
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        local_df = None
        if os.path.exists(save_path):
            try:
                local_df = pd.read_csv(save_path, encoding="gbk")
            except Exception as e:
                logging.warning(f"读取本地数据失败，将仅保存新抓取数据: {e}")
        if local_df is not None and not local_df.empty:
            merged = pd.concat([df, local_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["期数"], keep="first")
            merged = merged.sort_values(by="期数", ascending=False).reset_index(drop=True)
            logging.info(f"本地原有 {len(local_df)} 条，新抓取 {len(df)} 条，合并去重后共 {len(merged)} 条")
            df = merged
        df.to_csv(save_path, encoding="gbk", index=False)
        logging.info(f"数据已保存至 {save_path}，共 {len(df)} 条")
        return True
    except Exception as e:
        logging.warning(f"保存失败: {e}")
        return False

def main():
    return fetch_ssq_data()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
