"""
AIフォーム営業リサーチャー - バックエンドサーバー
（gBizINFO 公式APIを使った企業検索版）
"""
import os
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AIフォーム営業リサーチャー API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY", "")
GBIZINFO_TOKEN    = os.getenv("GBIZINFO_TOKEN", "")

GBIZINFO_BASE = "https://info.gbiz.go.jp/hojin/v1/hojin"

PREFECTURE_CODE = {
    "東京": "13", "大阪": "27", "名古屋": "23", "愛知": "23",
    "福岡": "40", "札幌": "01", "北海道": "01", "神奈川": "14",
    "埼玉": "11", "京都": "26", "兵庫": "28", "広島": "34", "仙台": "04", "宮城": "04",
}

INDUSTRY_KEYWORD_MAP = {
    "IT・ソフトウェア": "ソフトウェア",
    "製造業": "製造",
    "小売・EC": "小売",
    "不動産": "不動産",
    "医療・介護": "医療",
    "教育": "教育",
    "飲食": "飲食",
    "建設": "建設",
    "物流・運輸": "運輸",
    "金融・保険": "金融",
    "コンサルティング": "コンサルティング",
    "広告・マーケティング": "広告",
}


class SearchRequest(BaseModel):
    industry:  str = ""
    region:    str = ""
    employees: str = ""
    keyword:   str = ""

class CompanyUpdateRequest(BaseModel):
    company_id:   str
    sales_status: str
    memo:         str = ""


@app.post("/search")
async def search_companies(req: SearchRequest):
    if not GBIZINFO_TOKEN:
        return {
            "companies": [],
            "total": 0,
            "error": "GBIZINFO_TOKEN が設定されていません。Renderの環境変数を確認してください。",
        }

    candidates = await gbizinfo_search(req)
    print(f"[search_companies] gBizINFO candidates: {len(candidates)}")

    companies = []
    for c in candidates:
        try:
            company = build_company_record(c, req)

            # gBizINFOにURLが登録されていない場合は、会社名から検索して補う
            if not company.get("hp"):
                guessed_url = await guess_company_url(company["name"])
                if guessed_url:
                    company["hp"] = guessed_url
                    print(f"[search_companies] guessed url for {company['name']}: {guessed_url}")
                else:
                    print(f"[search_companies] skip (no url found): {company.get('name')}")
                    companies.append(company)  # URLなしでも基本情報だけは表示する
                    continue

            cached = await get_cached_company(company["name"])
            if cached:
                company = {**company, **cached, "_cached": True}
            else:
                form_info = await find_contact_form(company["hp"])
                company.update(form_info)
                score_result = await ai_score_company(company)
                company.update(score_result)
                await save_company(company)

            companies.append(company)
        except Exception as e:
            print(f"[search_companies] error processing {c.get('name')}: {type(e).__name__}: {e}")
            continue

    print(f"[search_companies] final company count: {len(companies)}")
    companies.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"companies": companies, "total": len(companies)}


@app.get("/companies")
async def get_companies():
    companies = await fetch_all_companies()
    return {"companies": companies}


@app.post("/update-status")
async def update_status(req: CompanyUpdateRequest):
    result = await update_company_status(req.company_id, req.sales_status, req.memo)
    return {"success": True, "updated": result}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gbizinfo_token_set": bool(GBIZINFO_TOKEN),
    }


@app.get("/debug-gbiz")
def debug_gbiz(name: str = "ソフトウェア", prefecture: str = "", limit: int = 5):
    params = {"name": name, "limit": limit}
    if prefecture:
        params["prefecture"] = prefecture
    headers = {"Accept": "application/json", "X-hojinInfo-api-token": GBIZINFO_TOKEN}
    resp = requests.get(GBIZINFO_BASE, headers=headers, params=params, timeout=15)
    return {
        "request_url": resp.url,
        "status_code": resp.status_code,
        "body": safe_json(resp),
    }


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return resp.text[:2000]


async def gbizinfo_search(req: SearchRequest) -> list:
    """
    gBizINFO APIで企業候補を検索する。
    同じ条件で再検索された場合は、前回の続きのページから取得することで、
    既に収集済みの企業との重複をできるだけ避ける。
    """
    name_query = ""
    if req.industry:
        name_query = INDUSTRY_KEYWORD_MAP.get(req.industry, req.industry)
    elif req.keyword:
        name_query = req.keyword
    else:
        name_query = "株式会社"

    prefecture_code = PREFECTURE_CODE.get(req.region, "")
    query_key = build_query_key(req)

    headers = {"Accept": "application/json", "X-hojinInfo-api-token": GBIZINFO_TOKEN}

    # 既存の検索済み企業名一覧を取得（重複除外用）
    existing_names = await get_existing_company_names()
    print(f"[gbizinfo] query_key='{query_key}' existing_names_count={len(existing_names)}")

    collected = []
    start_page = await get_next_page(query_key)
    print(f"[gbizinfo] start_page={start_page}")
    page = start_page
    max_pages_to_try = 5  # 無限ループ防止（5ページ分=最大100件まで探索）

    for _ in range(max_pages_to_try):
        params = {"name": name_query, "limit": 20, "page": page}
        if prefecture_code:
            params["prefecture"] = prefecture_code

        try:
            resp = requests.get(GBIZINFO_BASE, headers=headers, params=params, timeout=15)
            print(f"[gbizinfo] page={page} status={resp.status_code} url={resp.url}")
            if resp.status_code != 200:
                print(f"[gbizinfo] non-200 body: {resp.text[:500]}")
                break
            data = resp.json()
            infos = data.get("hojin-infos", [])
            print(f"[gbizinfo] page={page} got {len(infos)} results")

            if not infos:
                # これ以上データがない場合はページを1に戻して終了
                page = 1
                break

            # 既にDBにある企業名は除外
            new_infos = [i for i in infos if (i.get("name") or "") not in existing_names]
            collected.extend(new_infos)

            page += 1
            if len(collected) >= 20:
                break
        except Exception as e:
            print(f"[gbizinfo] error: {type(e).__name__}: {e}")
            break

    # 次回はこのページから再開する
    await set_next_page(query_key, page)
    print(f"[gbizinfo] collected {len(collected)} new companies (pages {start_page}..{page-1})")

    return collected[:20]


def build_query_key(req: SearchRequest) -> str:
    """検索条件から進捗管理用の一意なキーを作る"""
    parts = [req.industry or "-", req.region or "-", req.employees or "-", req.keyword or "-"]
    return "|".join(parts)


async def guess_company_url(company_name: str) -> str:
    """
    gBizINFOにURLが登録されていない場合、会社名でBing検索して
    公式サイトらしきURLを1件だけ推測する。
    検索結果一覧の取得ではなく単発の名寄せ用途のため、
    Bingのbot対策に引っかかってもアプリ全体は壊れない設計。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "ja-JP,ja;q=0.9",
    }
    skip_domains = [
        "bing.", "microsoft.", "google.", "wikipedia.", "youtube.", "twitter.", "x.com",
        "facebook.", "instagram.", "indeed.com", "houjin-bangou.nta.go.jp", "gbiz.go.jp",
        "mapfan.", "mapion.", "navitime.", "goo.ne.jp/map", "townpage.", "ecareernavi.",
    ]
    query = f"{company_name} 公式サイト"
    search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&mkt=ja-JP"

    try:
        resp = requests.get(search_url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "bing.com/ck/" in href:
                continue
            if not href.startswith("http"):
                continue
            if any(skip in href for skip in skip_domains):
                continue
            return href
    except Exception as e:
        print(f"[guess_company_url] {company_name} failed: {type(e).__name__}: {e}")

    return ""


def build_company_record(c: dict, req: SearchRequest) -> dict:
    name = c.get("name") or ""
    corp_no = c.get("corporate_number") or ""
    hp = c.get("company_url") or ""
    employees = c.get("employee_number") or parse_employee_range(req.employees)
    location = c.get("location") or ""

    company_id = hashlib.md5((corp_no or name).encode()).hexdigest()[:12]

    return {
        "id": company_id,
        "name": name,
        "corp_no": corp_no,
        "address": location,
        "tel": "",
        "hp": normalize_url(hp),
        "form_url": "",
        "employees": employees,
        "industry": req.industry or (c.get("business_summary") or "不明")[:20],
        "region": req.region or extract_region(location),
        "has_form": False,
        "form_type": "",
        "sales_ok": False,
        "no_sales_note": False,
        "score": 0,
        "reason": "",
        "form_fields": [],
        "required_fields": [],
        "has_recaptcha": False,
        "has_confirm_page": False,
        "has_checkbox": False,
        "sales_status": "未営業",
        "last_contact": None,
        "memo": "",
        "log": [],
        "updated_at": datetime.now().isoformat(),
        "_cached": False,
    }


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    return url


def extract_region(address: str) -> str:
    for pref in ["東京", "大阪", "名古屋", "愛知", "福岡", "北海道", "神奈川", "埼玉", "京都", "兵庫", "広島", "宮城"]:
        if pref in address:
            return pref
    return ""


def parse_employee_range(emp_str: str) -> int:
    ranges = {"1-10": 5, "11-50": 30, "51-200": 100, "201-500": 300, "501+": 600, "": 50}
    return ranges.get(emp_str, 50)


async def find_contact_form(base_url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesBot/1.0)"}
    form_keywords = ["contact", "inquiry", "お問い合わせ", "ご相談", "問い合わせ", "contact-us"]
    exclude_keywords = ["recruit", "採用", "ir-", "investor", "login", "signin"]

    result = {
        "has_form": False, "form_url": "", "form_type": "", "sales_ok": False,
        "no_sales_note": False, "form_fields": [], "required_fields": [],
        "has_recaptcha": False, "has_confirm_page": False, "has_checkbox": False,
    }

    candidate_paths = ["", "/contact", "/inquiry", "/about", "/company"]
    visited = set()

    for path in candidate_paths:
        url = base_url.rstrip("/") + path
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=headers, timeout=6)
            soup = BeautifulSoup(resp.text, "html.parser")

            text = soup.get_text()
            if any(kw in text for kw in ["営業はお断り", "勧誘お断り", "営業電話お断り", "営業禁止"]):
                result["no_sales_note"] = True

            for a in soup.find_all("a", href=True):
                href = a["href"]
                link_text = a.get_text()
                full_url = href if href.startswith("http") else base_url.rstrip("/") + "/" + href.lstrip("/")

                if any(kw in href.lower() or kw in link_text for kw in form_keywords):
                    if any(ex in href.lower() for ex in exclude_keywords):
                        continue
                    form_detail = await analyze_form_page(full_url)
                    if form_detail["has_form"]:
                        result.update(form_detail)
                        result["form_url"] = full_url
                        result["has_form"] = True
                        result["sales_ok"] = not result["no_sales_note"] and \
                            result["form_type"] not in ["採用", "IR・その他"]
                        return result

            if soup.find("form"):
                form_detail = await analyze_form_page(url)
                if form_detail["has_form"]:
                    result.update(form_detail)
                    result["form_url"] = url
                    result["has_form"] = True
                    result["sales_ok"] = not result["no_sales_note"] and \
                        result["form_type"] not in ["採用", "IR・その他"]
                    return result

        except Exception as e:
            print(f"[find_contact_form] {url} failed: {type(e).__name__}: {e}")
            continue

    return result


async def analyze_form_page(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesBot/1.0)"}
    result = {
        "has_form": False, "form_type": "", "form_fields": [], "required_fields": [],
        "has_recaptcha": False, "has_confirm_page": False, "has_checkbox": False,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form")
        if not form:
            return result

        result["has_form"] = True
        title = soup.find("title")
        page_text = (title.text if title else "") + soup.get_text()[:500]
        result["form_type"] = classify_form_type(page_text)

        fields, required = [], []
        for inp in form.find_all(["input", "textarea", "select"]):
            label = get_field_label(soup, inp)
            if label and label not in ["送信", "確認", "戻る", "リセット"]:
                fields.append(label)
                if inp.get("required") or inp.get("aria-required") == "true":
                    required.append(label)

        result["form_fields"] = fields[:10]
        result["required_fields"] = required
        result["has_recaptcha"] = bool(
            soup.find(class_=re.compile("recaptcha", re.I)) or
            soup.find("script", src=re.compile("recaptcha", re.I))
        )
        result["has_confirm_page"] = any(kw in page_text for kw in ["確認画面", "confirm", "内容確認"])
        result["has_checkbox"] = bool(form.find("input", {"type": "checkbox"}))
    except Exception as e:
        print(f"[analyze_form_page] {url} failed: {type(e).__name__}: {e}")

    return result


def classify_form_type(text: str) -> str:
    if any(kw in text for kw in ["採用", "エントリー", "応募", "求人"]):
        return "採用"
    if any(kw in text for kw in ["IR", "投資家", "株主"]):
        return "IR・その他"
    if any(kw in text for kw in ["資料請求", "カタログ"]):
        return "資料請求"
    if any(kw in text for kw in ["パートナー", "代理店", "加盟"]):
        return "パートナー募集"
    if any(kw in text for kw in ["相談", "ご相談"]):
        return "ご相談"
    return "お問い合わせ"


def get_field_label(soup, inp) -> str:
    inp_id = inp.get("id")
    if inp_id:
        label = soup.find("label", {"for": inp_id})
        if label:
            return label.get_text(strip=True)
    placeholder = inp.get("placeholder", "")
    if placeholder:
        return placeholder
    name = inp.get("name", "")
    label_map = {
        "name": "氏名", "company": "会社名", "email": "メールアドレス",
        "tel": "電話番号", "message": "お問い合わせ内容", "subject": "件名",
        "department": "部署名", "title": "役職",
    }
    for key, val in label_map.items():
        if key in name.lower():
            return val
    return ""


async def ai_score_company(company: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        return rule_based_score(company)

    prompt = f"""
以下の企業情報を分析し、フォーム営業の優先度をスコアリングしてください。

会社名: {company.get('name')}
業種: {company.get('industry')}
従業員数: {company.get('employees')}名
フォーム有無: {'あり' if company.get('has_form') else 'なし'}
フォーム種別: {company.get('form_type')}
営業禁止記載: {'あり' if company.get('no_sales_note') else 'なし'}
reCAPTCHA: {'あり' if company.get('has_recaptcha') else 'なし'}
入力項目数: {len(company.get('form_fields', []))}個

以下のJSON形式のみで返答してください（説明文不要）:
{{"score": 1〜5の整数, "reason": "判定理由を1〜2文で"}}
"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            data = resp.json()
            text = data["content"][0]["text"].strip()
            parsed = json.loads(text)
            return {"score": int(parsed.get("score", 1)), "reason": parsed.get("reason", "")}
    except Exception as e:
        print(f"[ai_score_company] error: {type(e).__name__}: {e}")
        return rule_based_score(company)


def rule_based_score(company: dict) -> dict:
    if not company.get("has_form"):
        return {"score": 0, "reason": "フォームが見つからないため対象外"}
    if company.get("no_sales_note"):
        return {"score": 0, "reason": "営業禁止の記載があるため対象外"}

    score = 1
    reasons = []
    form_type = company.get("form_type", "")
    if form_type in ["お問い合わせ", "ご相談"]:
        score += 2
        reasons.append(f"フォーム種別が「{form_type}」で営業適性高")
    elif form_type in ["資料請求", "パートナー募集"]:
        score += 1
        reasons.append(f"フォーム種別「{form_type}」（条件付き）")

    emp = company.get("employees", 0)
    if emp >= 200:
        score += 1
        reasons.append(f"従業員{emp}名の中堅企業")
    elif emp >= 50:
        reasons.append(f"従業員{emp}名の成長企業")

    if not company.get("has_recaptcha"):
        score = min(score + 1, 5)
        reasons.append("reCAPTCHAなしで送信しやすい")

    return {"score": min(score, 5), "reason": "・".join(reasons) if reasons else "基本条件を満たしています"}


def get_supabase_headers():
    """
    GET（データ取得）用のヘッダー。
    'Prefer: return=minimal' はPOST/PATCH専用のオプションのため、
    GETには付けない（付けるとSupabaseが結果を返さなくなる）。
    """
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


async def get_existing_company_names() -> set:
    """DBに既に保存されている全企業名を取得する（重複除外用）"""
    if not SUPABASE_URL:
        return set()
    try:
        url = f"{SUPABASE_URL}/rest/v1/companies?select=name"
        resp = requests.get(url, headers=get_supabase_headers(), timeout=8)
        data = resp.json()
        return {d.get("name") for d in data if d.get("name")}
    except Exception as e:
        print(f"[get_existing_company_names] error: {e}")
        return set()


async def get_next_page(query_key: str) -> int:
    """この検索条件で次に取得すべきページ番号を取得する（初回は1）"""
    if not SUPABASE_URL:
        return 1
    try:
        url = f"{SUPABASE_URL}/rest/v1/search_progress?query_key=eq.{requests.utils.quote(query_key)}"
        resp = requests.get(url, headers=get_supabase_headers(), timeout=5)
        data = resp.json()
        if data:
            return data[0].get("next_page", 1)
        return 1
    except Exception as e:
        print(f"[get_next_page] error: {e}")
        return 1


async def set_next_page(query_key: str, page: int) -> None:
    """この検索条件の次回開始ページ番号を保存する"""
    if not SUPABASE_URL:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/search_progress"
        headers = {**get_supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        payload = {"query_key": query_key, "next_page": page, "updated_at": datetime.now().isoformat()}
        requests.post(url, headers=headers, json=payload, timeout=5)
    except Exception as e:
        print(f"[set_next_page] error: {e}")


async def get_cached_company(name: str):
    if not SUPABASE_URL:
        return None
    try:
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"{SUPABASE_URL}/rest/v1/companies?name=eq.{requests.utils.quote(name)}&updated_at=gte.{cutoff}"
        resp = requests.get(url, headers=get_supabase_headers(), timeout=5)
        data = resp.json()
        return data[0] if data else None
    except Exception:
        return None


async def save_company(company: dict) -> bool:
    if not SUPABASE_URL:
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/companies"
        headers = {**get_supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        data = {k: v for k, v in company.items() if not k.startswith("_")}
        data["form_fields"] = json.dumps(data.get("form_fields", []))
        data["required_fields"] = json.dumps(data.get("required_fields", []))
        data["log"] = json.dumps(data.get("log", []))
        requests.post(url, headers=headers, json=data, timeout=5)
        return True
    except Exception as e:
        print(f"[save_company] error: {e}")
        return False


async def fetch_all_companies() -> list:
    if not SUPABASE_URL:
        return []
    try:
        url = f"{SUPABASE_URL}/rest/v1/companies?order=score.desc"
        resp = requests.get(url, headers=get_supabase_headers(), timeout=5)
        data = resp.json()
        for c in data:
            for key in ["form_fields", "required_fields", "log"]:
                if isinstance(c.get(key), str):
                    try:
                        c[key] = json.loads(c[key])
                    except Exception:
                        c[key] = []
        return data
    except Exception:
        return []


async def update_company_status(company_id: str, status: str, memo: str) -> bool:
    if not SUPABASE_URL:
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/companies?id=eq.{company_id}"
        headers = {**get_supabase_headers(), "Prefer": "return=minimal"}
        payload = {
            "sales_status": status,
            "memo": memo,
            "last_contact": datetime.now().date().isoformat() if status != "未営業" else None,
        }
        requests.patch(url, headers=headers, json=payload, timeout=5)
        return True
    except Exception:
        return False
