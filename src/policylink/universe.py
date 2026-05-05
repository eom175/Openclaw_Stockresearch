from typing import Dict, List


DEFAULT_UNIVERSE: List[Dict[str, str]] = [
    {"code": "005930", "stock_code": "005930", "name": "삼성전자", "stock_name": "삼성전자", "sector": "semiconductor_battery"},
    {"code": "000660", "stock_code": "000660", "name": "SK하이닉스", "stock_name": "SK하이닉스", "sector": "semiconductor_battery"},
    {"code": "005380", "stock_code": "005380", "name": "현대차", "stock_name": "현대차", "sector": "auto_ev"},
    {"code": "000270", "stock_code": "000270", "name": "기아", "stock_name": "기아", "sector": "auto_ev"},
    {"code": "105560", "stock_code": "105560", "name": "KB금융", "stock_name": "KB금융", "sector": "financial_value"},
    {"code": "055550", "stock_code": "055550", "name": "신한지주", "stock_name": "신한지주", "sector": "financial_value"},
    {"code": "035420", "stock_code": "035420", "name": "NAVER", "stock_name": "NAVER", "sector": "platform_internet"},
    {"code": "035720", "stock_code": "035720", "name": "카카오", "stock_name": "카카오", "sector": "platform_internet"},
    {"code": "012450", "stock_code": "012450", "name": "한화에어로스페이스", "stock_name": "한화에어로스페이스", "sector": "defense_shipbuilding"},
    {"code": "034020", "stock_code": "034020", "name": "두산에너빌리티", "stock_name": "두산에너빌리티", "sector": "energy_infra"},
]

KNOWN_STOCK_SECTOR: Dict[str, str] = {
    item["code"]: item["sector"]
    for item in DEFAULT_UNIVERSE
}


def universe_for_market_data() -> List[Dict[str, str]]:
    return [
        {"code": item["code"], "name": item["name"], "sector": item["sector"]}
        for item in DEFAULT_UNIVERSE
    ]


def universe_for_dataset() -> List[Dict[str, str]]:
    return [
        {"stock_code": item["stock_code"], "stock_name": item["stock_name"], "sector": item["sector"]}
        for item in DEFAULT_UNIVERSE
    ]
