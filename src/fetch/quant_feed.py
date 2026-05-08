# fetch/quant_feed.py
import asyncio
import logging
import yfinance as yf
from web3 import AsyncWeb3
from typing import Dict, Any, Optional

logger = logging.getLogger("Sovereign.JuniorQuantFeed")

class FinancialKinematicsIntake:
    """
    JuniorAGI_SDK Bridge.
    Asynchronously fetches High-Frequency Financial Modeling (HFFM) data and Web3 topologies.
    """
    def __init__(self, rpc_url: str = "https://eth.public-rpc.com"):
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))

    async def fetch_market_matrix(self, ticker: str) -> Dict[str, Any]:
        """Fetches standard market data and formats for tensor ingestion."""
        try:
            # Offload synchronous yfinance call
            def _get_data():
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d", interval="1h")
                if hist.empty:
                    return None
                return {
                    "latest_close": float(hist['Close'].iloc[-1]),
                    "latest_volume": float(hist['Volume'].iloc[-1]),
                    "vwap_5d": float((hist['Close'] * hist['Volume']).sum() / hist['Volume'].sum()) if hist['Volume'].sum() > 0 else 0.0,
                    "volatility": float(hist['Close'].pct_change().std())
                }
            
            data = await asyncio.to_thread(_get_data)
            if not data:
                 return {"status": "ERROR", "message": f"No liquidity data for {ticker}"}
                 
            return {"status": "SUCCESS", "ticker": ticker, "metrics": data}
        except Exception as e:
            logger.error(f"[-] Quant Feed Collapse [{ticker}]: {e}")
            return {"status": "ERROR", "message": str(e)}

    async def fetch_web3_topology(self) -> Dict[str, Any]:
        """Polls current Web3 block metrics for structural gas/throughput data."""
        try:
            if not await self.w3.is_connected():
                return {"status": "ERROR", "message": "RPC disconnected."}
                
            block = await self.w3.eth.get_block('latest')
            gas_price = await self.w3.eth.gas_price
            
            return {
                "status": "SUCCESS",
                "metrics": {
                    "block_number": block['number'],
                    "base_fee_gwei": float(self.w3.from_wei(block.get('baseFeePerGas', gas_price), 'gwei')),
                    "gas_used_ratio": float(block['gasUsed'] / block['gasLimit'])
                }
            }
        except Exception as e:
            logger.error(f"[-] Web3 Node Collapse: {e}")
            return {"status": "ERROR", "message": str(e)}
