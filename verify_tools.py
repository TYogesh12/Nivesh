import os
import sys

# Add app directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

from app.agent import get_stock_news, get_stock_price, get_watchlist


def main():
    print("=== Testing get_watchlist ===")
    watchlist = get_watchlist()
    print("Watchlist:", watchlist)
    assert "watchlist" in watchlist
    assert len(watchlist["watchlist"]) > 0

    print("\n=== Testing get_stock_price ===")
    for sym in watchlist["watchlist"][:2]:
        price_info = get_stock_price(sym)
        print(f"Price for {sym}:", price_info)
        assert "symbol" in price_info
        assert "price" in price_info or "error" in price_info

    print("\n=== Testing get_stock_news ===")
    for sym in watchlist["watchlist"][:2]:
        news_info = get_stock_news(sym)
        print(f"News for {sym}:", news_info)
        assert "symbol" in news_info
        assert "news" in news_info or "error" in news_info
        if "news" in news_info:
            print(f"Fetched {len(news_info['news'])} headlines.")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
