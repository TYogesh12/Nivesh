# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import yfinance as yf
import google.auth

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

# Set Vertex AI defaults only if GOOGLE_GENAI_USE_VERTEXAI is not explicitly set to False/True in env/dotenv
if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") is None:
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


watchlist = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "WIPRO"]


# PURPOSE: Fetches the current live stock price for a given ticker symbol.
# DESIGN: Uses yfinance with a '.NS' (NSE) suffix to support free NSE data. Alpha Vantage free tier only supports BSE data via '.BSE' suffixes, whereas yfinance provides comprehensive NSE coverage.
# TRADEOFF: yfinance is an unofficial scraper-based library, which makes it subject to sudden rate-limits or format breakages.
# BEHAVIOR: Automatically converts symbols to uppercase and appends '.NS' if not present.
def get_stock_price(symbol: str) -> dict:
    """Fetches the live stock price for a given symbol from NSE India.

    Args:
        symbol: The stock ticker symbol (e.g., 'RELIANCE', 'INFY').

    Returns:
        A dictionary containing the stock symbol and its current price, or an error message.
    """
    if not symbol:
        return {"error": "Symbol cannot be empty"}

    ticker_symbol = symbol.upper()
    if not ticker_symbol.endswith(".NS"):
        ticker_symbol = f"{ticker_symbol}.NS"

    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            price = hist["Close"].iloc[-1]
            return {"symbol": symbol, "price": round(float(price), 2)}

        # Fallback to fast_info
        try:
            price = ticker.fast_info["lastPrice"]
            if price is not None:
                return {"symbol": symbol, "price": round(float(price), 2)}
        except Exception:
            pass

        return {"symbol": symbol, "error": f"No price data found for {ticker_symbol}"}
    except Exception as e:
        return {"symbol": symbol, "error": f"Failed to fetch price: {str(e)}"}


# PURPOSE: Fetches the top 3 headlines from Google News RSS feed for a given stock symbol.
# DESIGN: Parses an XML RSS feed via urllib and xml.etree to avoid heavy external RSS parsing dependencies.
# TRADEOFF: Google News RSS headlines are recent but may be days old, not strictly real-time.
# BEHAVIOR: Returns a list of dictionaries with title, link, and pubDate keys.
def get_stock_news(symbol: str) -> dict:
    """Fetches the top 3 headlines from Google News RSS for a given Indian stock.

    Args:
        symbol: The stock ticker symbol (e.g., 'RELIANCE', 'INFY').

    Returns:
        A dictionary containing the symbol and a list of the top 3 news headlines.
    """
    if not symbol:
        return {"error": "Symbol cannot be empty"}

    query = f"{symbol.upper()} India share price"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        req = urllib.request.Request(
            rss_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        headlines = []
        for item in root.findall(".//item")[:3]:
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")

            headlines.append(
                {
                    "title": title_elem.text if title_elem is not None else "",
                    "link": link_elem.text if link_elem is not None else "",
                    "pubDate": pub_date_elem.text if pub_date_elem is not None else "",
                }
            )
        return {"symbol": symbol, "news": headlines}
    except Exception as e:
        return {"symbol": symbol, "error": f"Failed to fetch news: {str(e)}"}


# PURPOSE: Returns the current list of watched stock symbols.
# DESIGN: Returns a dictionary wrapping the mutable in-memory watchlist. The docstring restricts tool invocation to prevent the LLM from calling it on general queries.
# TRADEOFF: In-memory storage is volatile and will reset upon process restarts.
# BEHAVIOR: The LLM should only call this when the user explicitly inquires about their watchlist or portfolio status.
def get_watchlist() -> dict:
    """Returns a list of watched stock symbols.

    Returns:
        A dictionary containing the list of watchlist symbols.
    """
    return {"watchlist": watchlist}


# PURPOSE: Adds a new stock symbol to the user's watchlist.
# DESIGN: Standardizes the symbol to uppercase and appends to the in-memory list if it does not already exist. The docstring restricts tool invocation to prevent accidental model actions.
# TRADEOFF: Does not validate symbol validity against exchanges during addition.
# BEHAVIOR: The LLM must only call this upon receiving an explicit instruction from the user to add a stock.
def add_to_watchlist(symbol: str) -> dict:
    """Adds a stock symbol to the user's watchlist.
        Appends symbol.upper() to watchlist if not already present.
        Only call this when the user EXPLICITLY asks to add a stock.
        Never call this based on inference or context alone.

    Args:
        symbol: The stock ticker symbol to add.
    """
    sym = symbol.upper()
    if sym not in watchlist:
        watchlist.append(sym)
        return {"added": sym, "watchlist": watchlist}
    return {"message": f"{sym} already in watchlist", "watchlist": watchlist}


# PURPOSE: Removes a stock symbol from the user's watchlist.
# DESIGN: Standardizes the symbol to uppercase and removes it from the in-memory list if present. The docstring restricts tool invocation to avoid accidental model actions.
# TRADEOFF: Silent failure response if the stock symbol is not in the watchlist.
# BEHAVIOR: The LLM must only call this upon receiving an explicit instruction from the user to remove a stock, or if a user confirms removal of an invalid stock.
def remove_from_watchlist(symbol: str) -> dict:
    """Removes symbol.upper() from the user's watchlist.
       Only call this when the user EXPLICITLY asks to remove a stock
       or confirms removal after an invalid symbol warning.

    Args:
        symbol: The stock ticker symbol to remove.
    """
    sym = symbol.upper()
    if sym in watchlist:
        watchlist.remove(sym)
        return {"removed": sym, "watchlist": watchlist}
    return {"message": f"{sym} not in watchlist", "watchlist": watchlist}


root_agent = Agent(
    name="paisa_agent",
    model=Gemini(
        model="gemini-3.1-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        """
        You are Nivesh, a personal concierge for Indian retail investors.
        Your personality: direct, data-driven, and conversational —
        like a knowledgeable friend who follows the markets, not a
        cautious financial advisor. You give real analysis, not disclaimers.
        TOOL CALLING RULES — follow these strictly:
        1. Any question about a specific stock → call get_stock_price() and get_stock_news()
        2. Any question about 'my portfolio', 'my watchlist', 'my stocks',
           'how am I doing', 'what do you think' → call get_watchlist() first,
           then get_stock_price() AND get_stock_news() for EVERY stock returned.
        3. Users can manage their watchlist by saying 'add X to my watchlist' or 'remove X from my watchlist'. Always call add_to_watchlist() or remove_from_watchlist() for these requests.
        4. NEVER answer price or news questions from memory. Always call the tool.
        5. If no tool is relevant, answer conversationally from knowledge.

        RESPONSE RULES:
        1. Lead with the data you fetched, then your analysis.
        2. News headlines are recent but may be days old — mention this naturally,
           not as a disclaimer at the end.
        3. After any watchlist change, confirm what was changed and show the updated watchlist.
        4. Never say 'I cannot provide financial advice'.
           Instead say 'here is what the data shows' and let the user decide.
        5. Keep responses concise and structured with headers for multi-stock answers.

        If get_stock_price returns no data for a symbol, immediately
        offer to remove it from the watchlist and call
        remove_from_watchlist() if the user confirms.
        """
    ),
    tools=[
        get_stock_price,
        get_stock_news,
        get_watchlist,
        add_to_watchlist,
        remove_from_watchlist,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)

# Set up process-wide session service
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="app",
    session_service=session_service,
    auto_create_session=True,
)
