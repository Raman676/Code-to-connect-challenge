"""
Text preparation for the sentiment/impact model.

Masks everything that identifies the company, so the model has to learn from what the
text says rather than from who it is about (e.g. it can't learn "Nike = negative"):
  URLs                                   -> [URL]
  @mentions, $cashtags, "NYSE: KO" forms -> [COMPANY]
  pool company names, subsidiaries, CEOs -> [COMPANY]
  pool tickers (upper-case, whole word)  -> [COMPANY]

Product names (iPhone, Windows, Prime, Air Jordan, ...) are NOT masked: a known
limitation, listed in the model report.
"""
from __future__ import annotations

import re

import pandas as pd

COMPANY_TOKEN = "[COMPANY]"
URL_TOKEN = "[URL]"

# Masked in any letter case. Includes company names, subsidiaries/units and CEOs (current
# and those in office during 2009-2020) of the 47 pool companies.
NAMES_ANY_CASE = [
    # company names
    "Microsoft", "NVIDIA", "Amazon", "Alphabet", "Google", "Facebook", "Broadcom", "Tesla",
    "Berkshire Hathaway", "Berkshire", "JPMorgan Chase", "JPMorgan", "JP Morgan", "J.P. Morgan",
    "Walmart", "Wal-Mart", "Eli Lilly", "Mastercard", "Netflix", "Exxon Mobil", "ExxonMobil", "Exxon",
    "Costco", "Johnson & Johnson", "Johnson and Johnson", "J&J", "Home Depot", "Procter & Gamble",
    "Procter and Gamble", "P&G", "AbbVie", "Bank of America", "BofA", "Palantir", "UnitedHealth",
    "Coca-Cola", "Coca Cola", "CocaCola", "Cisco", "Wells Fargo", "Salesforce", "salesforce.com", "IBM",
    "Merck", "PepsiCo", "Pepsi", "Adobe", "Disney", "AT&T", "Verizon", "Pfizer", "Comcast", "Nike",
    "McDonald's", "McDonald’s", "McDonalds", "PayPal", "Boeing", "Citigroup", "Citibank", "Citi",
    "General Electric",
    # subsidiaries and units
    "Instagram", "WhatsApp", "Oculus", "YouTube", "Waymo", "AWS", "Whole Foods", "Zappos", "LinkedIn",
    "GitHub", "Xbox", "Pixar", "Lucasfilm", "ESPN", "Hulu", "NBCUniversal", "DirecTV", "WarnerMedia",
    "Time Warner", "HBO", "Venmo", "Gatorade", "Frito-Lay", "Sam's Club", "Jet.com", "Red Hat",
    "MuleSoft", "Tableau", "Allergan", "Optum", "Merrill Lynch", "Gillette", "Mobil",
    # CEOs (full names where the surname alone is a common word)
    "Tim Cook", "Satya Nadella", "Nadella", "Steve Ballmer", "Ballmer", "Bill Gates", "Jensen Huang",
    "Jeff Bezos", "Bezos", "Andy Jassy", "Jassy", "Sundar Pichai", "Pichai", "Larry Page", "Sergey Brin",
    "Mark Zuckerberg", "Zuckerberg", "Sheryl Sandberg", "Sandberg", "Hock Tan", "Elon Musk",
    "Warren Buffett", "Buffett", "Jamie Dimon", "Dimon", "Doug McMillon", "McMillon", "Larry Ellison",
    "Safra Catz", "Mark Hurd", "Reed Hastings", "Darren Woods", "Rex Tillerson", "Tillerson",
    "Craig Jelinek", "Alex Gorsky", "Gorsky", "Craig Menear", "Frank Blake", "David Taylor",
    "Rick Gonzalez", "Brian Moynihan", "Moynihan", "Alex Karp", "Lisa Su", "David Wichmann",
    "Stephen Hemsley", "James Quincey", "Muhtar Kent", "Chuck Robbins", "John Chambers", "Mike Wirth",
    "John Watson", "Tim Sloan", "Charles Scharf", "John Stumpf", "Stumpf", "Marc Benioff", "Benioff",
    "Ginni Rometty", "Rometty", "Arvind Krishna", "Ken Frazier", "Kenneth Frazier", "Indra Nooyi",
    "Nooyi", "Ramon Laguarta", "Shantanu Narayen", "Narayen", "Bob Iger", "Iger", "Bob Chapek", "Chapek",
    "Randall Stephenson", "John Stankey", "Hans Vestberg", "Lowell McAdam", "Ian Read", "Albert Bourla",
    "Bourla", "Brian Krzanich", "Krzanich", "Bob Swan", "Pat Gelsinger", "Gelsinger", "Brian Roberts",
    "Mark Parker", "John Donahoe", "Phil Knight", "Steve Easterbrook", "Easterbrook", "Chris Kempczinski",
    "Kempczinski", "Dan Schulman", "Dennis Muilenburg", "Muilenburg", "Dave Calhoun", "Michael Corbat",
    "Corbat", "Vikram Pandit", "Jeff Immelt", "Immelt", "John Flannery", "Larry Culp", "Al Kelly",
    "Charlie Scharf", "Ajay Banga", "Banga",
    # --- S&P 100 members beyond the original 47-company pool ---
    # company names (incl. former names used in 2009-2020 news)
    "Abbott Laboratories", "Abbott", "Accenture", "Applied Materials", "Amgen", "American Tower",
    "Arista Networks", "Arista", "American Express", "AmEx", "Booking Holdings", "Booking.com", "Priceline",
    "BlackRock", "Bristol-Myers Squibb", "Bristol Myers Squibb", "Bristol-Myers", "BNY Mellon",
    "Bank of New York Mellon", "Bank of New York", "Capital One", "ConocoPhillips", "Conoco", "CVS Health",
    "CVS", "Deere", "Danaher", "Duke Energy", "Emerson Electric", "FedEx", "General Dynamics", "GE Vernova",
    "Gilead Sciences", "Gilead", "General Motors", "Goldman Sachs", "Goldman", "Intuitive Surgical",
    "Linde", "Praxair", "Lockheed Martin", "Lockheed", "Lowe's", "Lowe’s", "Lowes", "Lam Research",
    "Mondelez", "Mondelēz", "Kraft Foods", "Medtronic", "3M Company", "3M Co", "Altria", "Morgan Stanley",
    "NextEra Energy", "NextEra", "Florida Power & Light", "ServiceNow", "Palo Alto Networks",
    "Philip Morris", "Qualcomm", "Raytheon", "United Technologies", "RTX Corp", "Starbucks",
    "Charles Schwab", "Schwab", "SanDisk", "Southern Company", "Southern Co", "Thermo Fisher Scientific",
    "Thermo Fisher", "T-Mobile", "MetroPCS", "Texas Instruments", "Union Pacific", "United Parcel Service",
    "U.S. Bancorp", "US Bancorp", "Accenture plc",
    # subsidiaries and units
    "OpenTable", "iShares", "Pratt & Whitney", "Collins Aerospace", "Gulfstream", "Sikorsky",
    "E*Trade", "Florida Power and Light", "Gulf Power",
    # CEOs 2009-2020
    "Miles White", "Robert Ford", "Pierre Nanterme", "Julie Sweet", "Gary Dickerson", "Robert Bradway",
    "Jim Taiclet", "Tom Bartlett", "Jayshree Ullal", "Ken Chenault", "Kenneth Chenault", "Chenault",
    "Stephen Squeri", "Squeri", "Glenn Fogel", "Jeffery Boyd", "Larry Fink", "Laurence Fink",
    "Giovanni Caforio", "Caforio", "Lamberto Andreotti", "Gerald Hassell", "Todd Gibbons",
    "Doug Oberhelman", "Oberhelman", "Jim Umpleby", "Umpleby", "Richard Fairbank", "Ryan Lance",
    "Larry Merlo", "Karen Lynch", "Samuel Allen", "John May", "Michael Dell", "Tom Joyce", "Lynn Good",
    "David Farr", "Fred Smith", "Phebe Novakovic", "Novakovic", "John Milligan", "Daniel O'Day",
    "Mary Barra", "Dan Akerson", "Lloyd Blankfein", "Blankfein", "David Solomon", "Sasan Goodarzi",
    "Gary Guthart", "Steve Angel", "Marillyn Hewson", "Hewson", "Robert Niblock", "Marvin Ellison",
    "Martin Anstice", "Tim Archer", "Irene Rosenfeld", "Dirk Van de Put", "Omar Ishrak", "Inge Thulin",
    "Mike Roman", "Marty Barrington", "Howard Willard", "James Gorman", "John Mack", "Sanjay Mehrotra",
    "Mark Durcan", "Jim Robo", "Frank Slootman", "Slootman", "Mark McLaughlin", "Nikesh Arora",
    "Andre Calantzopoulos", "Steve Mollenkopf", "Mollenkopf", "Paul Jacobs", "Greg Hayes",
    "Louis Chenevert", "Howard Schultz", "Kevin Johnson", "Walt Bettinger", "Tom Fanning", "Marc Casper",
    "John Legere", "Legere", "Mike Sievert", "Rich Templeton", "Dara Khosrowshahi", "Khosrowshahi",
    "Travis Kalanick", "Kalanick", "Lance Fritz", "David Abney", "Carol Tome", "Richard Davis", "Andy Cecere",
]

# Masked only when written as here (or in all capitals): these are also ordinary words.
NAMES_CAPITALISED_ONLY = [
    "Apple", "Meta", "Visa", "Oracle", "Intel", "Chevron", "Lilly", "Coke", "Chase", "Marvel",
    "Android", "Azure", "Converse", "Slack", "Musk", "Ellison", "Hastings", "Karp", "Merrill",
    # S&P 100 additions
    "Caterpillar", "Dell", "Intuit", "Micron", "Uber", "Emerson", "Lowe", "Sprint", "Kayak",
]

# Pool and S&P 100 tickers, plus historical symbols used in the news data (FB, PCLN, BK, UTX, ...).
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "FB", "AVGO", "TSLA", "BRK.A", "BRK.B",
    "BRK-A", "BRK-B", "BRK/A", "BRK/B", "JPM", "WMT", "LLY", "ORCL", "V", "MA", "NFLX", "XOM", "COST",
    "JNJ", "HD", "PG", "ABBV", "BAC", "PLTR", "AMD", "UNH", "KO", "CSCO", "CVX", "WFC", "CRM", "IBM",
    "MRK", "PEP", "ADBE", "DIS", "T", "VZ", "INTC", "PFE", "CMCSA", "NKE", "MCD", "PYPL", "BA", "C", "GE",
    # S&P 100 additions
    "ABT", "ACN", "AMAT", "AMGN", "AMT", "ANET", "AXP", "BKNG", "PCLN", "BLK", "BMY", "BNY", "BK", "CAT",
    "COF", "COP", "CVS", "DE", "DELL", "DVMT", "DHR", "DUK", "EMR", "FDX", "GD", "GEV", "GILD", "GM", "GS",
    "INTU", "ISRG", "LIN", "PX", "LMT", "LOW", "LRCX", "MDLZ", "KFT", "MDT", "MMM", "MO", "MS", "MU", "NEE",
    "FPL", "NOW", "PANW", "PM", "QCOM", "RTX", "UTX", "SBUX", "SCHW", "SNDK", "SO", "TMO", "TMUS", "PCS",
    "TXN", "UBER", "UNP", "UPS", "USB",
]
# Tickers that are also letters or ordinary words ("52-Week LOW", "BUY NOW", "SO"): masked only as
# "(TICKER)" (cashtags and "NYSE: TICKER" are handled separately), never as a bare word.
AMBIGUOUS_TICKERS = {"T", "V", "C", "SO", "NOW", "LOW", "MO", "MS", "DE", "GD", "USB", "AMT", "COST", "ALL", "PM"}


def _alternation(words) -> str:
    # Longest first, so "Bank of America" wins over shorter overlapping names.
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


_URL_RE = re.compile(r"\bhttps?\S*|\bwww\.\S+|\bt\.co/\S*", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_CASHTAG_RE = re.compile(r"\$[A-Za-z][A-Za-z.\-]{0,9}\b")
_EXCHANGE_RE = re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq|NYSEArca|NYSEARCA)\s*:\s*[A-Z][A-Z.\-]{0,5}\b")
_NAMES_ANY_RE = re.compile(rf"(?<!\w)(?:{_alternation(NAMES_ANY_CASE)})(?!\w)", re.IGNORECASE)
_NAMES_CAP_RE = re.compile(
    rf"(?<!\w)(?:{_alternation(NAMES_CAPITALISED_ONLY + [n.upper() for n in NAMES_CAPITALISED_ONLY])})(?!\w)")
_BARE_TICKERS = [t for t in TICKERS if t not in AMBIGUOUS_TICKERS]
_TICKER_RE = re.compile(rf"(?<![\w$\[])(?:{_alternation(_BARE_TICKERS)})(?![\w\]])")
# Ambiguous tickers only in an unmistakable form: "(T)", "(LOW)".
_SINGLE_LETTER_RE = re.compile(rf"\((?:{_alternation(AMBIGUOUS_TICKERS)})\)")
# "3M" the company, but not amounts such as "$3M" or "€3M"
_THREE_M_RE = re.compile(r"(?<![\w$€£.,])3M(?!\w)")
_SPACES_RE = re.compile(r"\s+")


def mask_text(text: str) -> str:
    """Replace URLs with [URL] and company identifiers with [COMPANY]."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(URL_TOKEN, text)          # first: URLs can contain @, $ and names
    text = _MENTION_RE.sub(COMPANY_TOKEN, text)
    text = _CASHTAG_RE.sub(COMPANY_TOKEN, text)
    text = _EXCHANGE_RE.sub(COMPANY_TOKEN, text)
    text = _NAMES_ANY_RE.sub(COMPANY_TOKEN, text)
    text = _NAMES_CAP_RE.sub(COMPANY_TOKEN, text)
    text = _THREE_M_RE.sub(COMPANY_TOKEN, text)
    text = _TICKER_RE.sub(COMPANY_TOKEN, text)
    text = _SINGLE_LETTER_RE.sub(f"({COMPANY_TOKEN})", text)
    return _SPACES_RE.sub(" ", text).strip()


def cap_per_company_day(df: pd.DataFrame, n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Keep at most n random rows per (ticker, anchor_date), with a fixed seed."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(["ticker", "anchor_date"]).head(n).sort_index()
