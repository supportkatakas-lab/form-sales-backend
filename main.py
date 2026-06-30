"""
AIフォーム営業リサーチャー - バックエンドサーバー
"""
import os
import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AIフォーム営業リサーチャー API")

# ─── CORS設定（フロントエンドからのアクセスを許可）───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では自分のURLに絞ること
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Anthropic Claude APIクライアント ───
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY", "")

# ─── リクエスト/レスポンスの型定義 ───
class SearchRequest(BaseModel):
    industry:  str = ""
    region:    str = ""
    employees: str = ""
    keyword:   str = ""

class CompanyUpdateRequest(BaseModel):
    company_id:   str
    sales_status: str
    memo:         str = ""


# ══════════════════════════════════════════════════════
#  ① 企業検索エンドポイント
# ══════════════════════════════════════════════════════
@app.post("/search")
async def search_companies(req: SearchRequest):
    """
    条件に合う企業を収集してスコアリングして返す
    """
    query_parts = []
    if req.industry:  query_parts.append(req.industry)
    if req.region:    query_parts.append(req.region)
    if req.keyword:   query_parts.append(req.keyword)
    if req.employees: query_parts.append(f"従業員{req.employees}名")
    query_parts.append("お問い合わせ 企業")
    query = " ".join(query_parts)
    print(f"[search_companies] query='{query}'")

    # 1. Google検索で企業URLを収集
    urls = await google_search(query, num=15)
    print(f"[search_companies] found {len(urls)} candidate urls")

    # 2. 各URLから企業情報を取得
    companies = []
    for url in urls:
        try:
            company = await extract_company_info(url, req)
            if company:
                # 3. Supabaseで30日キャッシュ確認
                cached = await get_cached_company(company["name"])
                if cached:
                    company = {**company, **cached, "_cached": True}
                else:
                    # 4. フォーム探索
                    form_info = await find_contact_form(url)
                    company.update(form_info)
                    # 5. AIスコアリング
                    score_result = await ai_score_company(company)
                    company.update(score_result)
                    # 6. Supabaseに保存
                    await save_company(company)
                companies.append(company)
            else:
                print(f"[search_companies] skipped (no name extracted): {url}")
        except Exception as e:
            print(f"[search_companies] error processing {url}: {type(e).__name__}: {e}")
            continue

    print(f"[search_companies] final company count: {len(companies)}")
    # スコア順に並べて返す
    companies.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"companies": companies, "total": len(companies)}


# ══════════════════════════════════════════════════════
#  ② 企業DB取得エンドポイント
# ══════════════════════════════════════════════════════
@app.get("/companies")
async def get_companies():
    """保存済み企業一覧を返す"""
    companies = await fetch_all_companies()
    return {"companies": companies}


# ══════════════════════════════════════════════════════
#  ③ 営業ステータス更新エンドポイント
# ══════════════════════════════════════════════════════
@app.post("/update-status")
async def update_status(req: CompanyUpdateRequest):
    """営業ステータスとメモを更新する"""
    result = await update_company_status(
        req.company_id,
        req.sales_status,
        req.memo
    )
    return {"success": True, "updated": result}


# ══════════════════════════════════════════════════════
#  ④ 法人番号検索エンドポイント（国税庁API）
# ══════════════════════════════════════════════════════
@app.get("/corp-number/{company_name}")
async def get_corp_number(company_name: str):
    """国税庁APIで法人番号を取得する"""
    corp_no = await fetch_corp_number(company_name)
    return {"corp_number": corp_no}


# ══════════════════════════════════════════════════════
#  ⑤ ヘルスチェック（サーバーが動いてるか確認用）
# ══════════════════════════════════════════════════════
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ══════════════════════════════════════════════════════
#  内部関数群
# ══════════════════════════════════════════════════════

async def google_search(query: str, num: int = 15) -> list[str]:
    """
    企業URLリストを取得する。
    DuckDuckGoはクラウドサーバーのIPからブロックされやすいため、
    Bing検索（HTML版）をメインに使用し、失敗時はDuckDuckGoにフォールバックする。
    """
    urls = []

    # ── ① Bing検索を試す ──
    try:
        urls = bing_search(query, num)
        print(f"[search] Bing returned {len(urls)} urls for query='{query}'")
        if urls:
            return urls[:num]
    except Exception as e:
        print(f"[search] Bing search error: {type(e).__name__}: {e}")

    # ── ② ダメならDuckDuckGoにフォールバック ──
    try:
        urls = duckduckgo_search(query, num)
        print(f"[search] DuckDuckGo returned {len(urls)} urls for query='{query}'")
    except Exception as e:
        print(f"[search] DuckDuckGo search error: {type(e).__name__}: {e}")

    return urls[:num]


def bing_search(query: str, num: int) -> list[str]:
    """Bing検索のHTML結果からURLを抽出する"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "ja-JP,ja;q=0.9",
    }
    search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&count=30"
    resp = requests.get(search_url, headers=headers, timeout=12)
    print(f"[bing] status={resp.status_code} length={len(resp.text)}")
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    skip_domains = ["bing.", "microsoft.", "google.", "wikipedia.", "youtube.",
                     "twitter.", "x.com", "facebook.", "instagram.", "indeed.com"]

    # Bingの検索結果リンクは <li class="b_algo"> の中の <h2><a href="...">
    for li in soup.select("li.b_algo"):
        a = li.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if href.startswith("http") and not any(skip in href for skip in skip_domains):
            urls.append(href)

    return urls[:num]


def duckduckgo_search(query: str, num: int) -> list[str]:
    """DuckDuckGo検索のHTML結果からURLを抽出する（フォールバック用）"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    resp = requests.get(search_url, headers=headers, timeout=10)
    print(f"[ddg] status={resp.status_code} length={len(resp.text)}")
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    skip_domains = ["google.", "wikipedia.", "youtube.", "twitter.", "facebook."]

    for a in soup.select(".result__a, .result__url"):
        href = a.get("href", "")
        if href.startswith("http") and not any(skip in href for skip in skip_domains):
            urls.append(href)

    return urls[:num]


async def extract_company_info(url: str, req: SearchRequest) -> Optional[dict]:
    """
    企業サイトのトップページから基本情報を取得する
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesBot/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 会社名を取得（titleタグまたはh1）
        name = ""
        title_tag = soup.find("title")
        if title_tag:
            name = title_tag.text.strip().split("|")[0].split("｜")[0].split("-")[0].strip()
        if not name:
            h1 = soup.find("h1")
            if h1:
                name = h1.text.strip()
        if not name or len(name) > 50:
            return None

        # メタ情報・電話番号を取得
        tel = extract_phone(soup)
        address = extract_address(soup)

        # 会社IDを生成（URL のハッシュ）
        company_id = hashlib.md5(url.encode()).hexdigest()[:12]

        return {
            "id": company_id,
            "name": name,
            "hp": url,
            "tel": tel,
            "address": address,
            "industry": req.industry or "不明",
            "region": req.region or extract_region(address),
            "employees": parse_employee_range(req.employees),
            "corp_no": "",
            "score": 0,
            "has_form": False,
            "form_url": "",
            "form_type": "",
            "sales_ok": False,
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
    except Exception as e:
        print(f"[extract] failed for {url}: {type(e).__name__}: {e}")
        return None


async def find_contact_form(base_url: str) -> dict:
    """
    企業サイトのフォームURLを探索する
    トップ・会社概要・フッター・サイトマップを巡回
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesBot/1.0)"}
    form_keywords = ["contact", "inquiry", "お問い合わせ", "ご相談", "問い合わせ", "contact-us"]
    exclude_keywords = ["recruit", "採用", "ir-", "investor", "login", "signin"]

    result = {
        "has_form": False,
        "form_url": "",
        "form_type": "",
        "sales_ok": False,
        "no_sales_note": False,
        "form_fields": [],
        "required_fields": [],
        "has_recaptcha": False,
        "has_confirm_page": False,
        "has_checkbox": False,
    }

    # 巡回するページ候補
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

            # 営業禁止表記チェック
            text = soup.get_text()
            if any(kw in text for kw in ["営業はお断り", "勧誘お断り", "営業電話お断り", "営業禁止"]):
                result["no_sales_note"] = True

            # フォームを含むリンクを探す
            for a in soup.find_all("a", href=True):
                href = a["href"]
                link_text = a.get_text()
                full_url = href if href.startswith("http") else base_url.rstrip("/") + "/" + href.lstrip("/")

                if any(kw in href.lower() or kw in link_text for kw in form_keywords):
                    if any(ex in href.lower() for ex in exclude_keywords):
                        continue
                    # フォームページを解析
                    form_detail = await analyze_form_page(full_url)
                    if form_detail["has_form"]:
                        result.update(form_detail)
                        result["form_url"] = full_url
                        result["has_form"] = True
                        result["sales_ok"] = not result["no_sales_note"] and \
                            result["form_type"] not in ["採用", "IR・その他"]
                        return result

            # ページ内にフォームタグがあるか確認
            if soup.find("form"):
                form_detail = await analyze_form_page(url)
                if form_detail["has_form"]:
                    result.update(form_detail)
                    result["form_url"] = url
                    result["has_form"] = True
                    result["sales_ok"] = not result["no_sales_note"] and \
                        result["form_type"] not in ["採用", "IR・その他"]
                    return result

        except Exception:
            continue

    return result


async def analyze_form_page(url: str) -> dict:
    """
    フォームページの構造を解析する
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesBot/1.0)"}
    result = {
        "has_form": False,
        "form_type": "",
        "form_fields": [],
        "required_fields": [],
        "has_recaptcha": False,
        "has_confirm_page": False,
        "has_checkbox": False,
    }

    try:
        resp = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form")
        if not form:
            return result

        result["has_form"] = True

        # ページタイトルからフォーム種別を判定
        title = soup.find("title")
        page_text = (title.text if title else "") + soup.get_text()[:500]
        result["form_type"] = classify_form_type(page_text)

        # 入力フィールドを取得
        fields = []
        required = []
        for inp in form.find_all(["input", "textarea", "select"]):
            label = get_field_label(soup, inp)
            if label and label not in ["送信", "確認", "戻る", "リセット"]:
                fields.append(label)
                if inp.get("required") or inp.get("aria-required") == "true":
                    required.append(label)

        result["form_fields"] = fields[:10]
        result["required_fields"] = required

        # reCAPTCHA チェック
        result["has_recaptcha"] = bool(
            soup.find(class_=re.compile("recaptcha", re.I)) or
            soup.find("script", src=re.compile("recaptcha", re.I))
        )

        # 確認画面チェック
        result["has_confirm_page"] = any(
            kw in page_text for kw in ["確認画面", "confirm", "内容確認"]
        )

        # チェックボックスチェック
        result["has_checkbox"] = bool(form.find("input", {"type": "checkbox"}))

    except Exception:
        pass

    return result


def classify_form_type(text: str) -> str:
    """テキストからフォーム種別を判定"""
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
    """inputに対応するラベルテキストを取得"""
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


def extract_phone(soup) -> str:
    """電話番号を抽出"""
    text = soup.get_text()
    pattern = r"0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_address(soup) -> str:
    """住所を抽出"""
    text = soup.get_text()
    pattern = r"[〒\d]{3}-?\d{4}[\s\S]{0,50}[都道府県][\s\S]{0,100}[0-9１-９一二三四五六七八九十]+[丁目番地号]"
    match = re.search(pattern, text)
    return match.group(0).strip()[:80] if match else ""


def extract_region(address: str) -> str:
    """住所から都道府県を抽出"""
    for pref in ["東京", "大阪", "名古屋", "福岡", "神奈川", "埼玉", "京都", "兵庫", "広島", "仙台"]:
        if pref in address:
            return pref
    return ""


def parse_employee_range(emp_str: str) -> int:
    """従業員数レンジから代表値を返す"""
    ranges = {
        "1-10": 5, "11-50": 30, "51-200": 100,
        "201-500": 300, "501+": 600, "": 50
    }
    return ranges.get(emp_str, 50)


async def ai_score_company(company: dict) -> dict:
    """
    Claude APIで企業の営業優先度をスコアリングする
    """
    if not ANTHROPIC_API_KEY:
        # APIキーがない場合はルールベースでスコアリング
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
            return {
                "score": int(parsed.get("score", 1)),
                "reason": parsed.get("reason", ""),
            }
    except Exception as e:
        print(f"AI scoring error: {e}")
        return rule_based_score(company)


def rule_based_score(company: dict) -> dict:
    """AIなしのルールベーススコアリング（フォールバック）"""
    score = 0
    reasons = []

    if not company.get("has_form"):
        return {"score": 0, "reason": "フォームが見つからないため対象外"}

    if company.get("no_sales_note"):
        return {"score": 0, "reason": "営業禁止の記載があるため対象外"}

    score += 1
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

    return {
        "score": min(score, 5),
        "reason": "・".join(reasons) if reasons else "基本条件を満たしています",
    }


async def fetch_corp_number(company_name: str) -> str:
    """国税庁法人番号APIで法人番号を取得"""
    try:
        url = "https://api.houjin-bangou.nta.go.jp/4/name"
        params = {
            "id": os.getenv("NTA_API_KEY", ""),  # 国税庁APIキー（無料取得可）
            "name": company_name,
            "type": "12",
        }
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data.get("corporations"):
            return data["corporations"][0].get("corporateNumber", "")
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════
#  Supabase連携関数
# ══════════════════════════════════════════════════════

def get_supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def get_cached_company(name: str) -> Optional[dict]:
    """30日以内に取得済みの企業データを返す"""
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
    """企業データをSupabaseに保存（upsert）"""
    if not SUPABASE_URL:
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/companies"
        headers = {**get_supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        # JSON serializable に変換
        data = {k: v for k, v in company.items() if not k.startswith("_")}
        data["form_fields"] = json.dumps(data.get("form_fields", []))
        data["required_fields"] = json.dumps(data.get("required_fields", []))
        data["log"] = json.dumps(data.get("log", []))
        requests.post(url, headers=headers, json=data, timeout=5)
        return True
    except Exception as e:
        print(f"Supabase save error: {e}")
        return False


async def fetch_all_companies() -> list:
    """Supabaseから全企業データを取得"""
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
    """営業ステータスを更新"""
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
