"""Task 11: documented provenance of the 13 event reference dates, and window spacing.

Reviewer 2 major 4(a)-(b). For each event this records the underlying documented event,
its date, the scoring reference date, what that date represents (the anchor), why it is
used, and the sources consulted, strongest first: regulator, government, central bank or
exchange; official chronology; published study; contemporaneous reporting only where no
official source dates the day. Each source row states the proposition it supports, in
paraphrase, and how it was verified (page fetched on the retrieval date, or text extracted
from the PDF). The selection rationale was not recorded when the panel was built, so every
rationale here is labelled as a revision rationale, not as source-documented.

The script also lists the distance between adjacent event windows under the active panel,
searches the repository and local archive for a candidate list or overlap-priority rule,
and compares the active panel with the submitted one.

Outputs: <out root>/t11_event_provenance/
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from _common import HALF_WINDOW, PANEL, ROOT, load_asset, outdir, spy_events  # noqa: E402
from src.event_panels import load_seed  # noqa: E402
from src.revision_analysis import event_spans  # noqa: E402

OUT = outdir("t11_event_provenance")
RETRIEVED = "2026-10-03"
FETCHED, PDF = "page fetched and read", "PDF retrieved and its text extracted"

# event_id: (underlying documented event, its date(s), anchor type, revision rationale, [sources])
# source: (tier, citation, url, proposition supported, verification)
EVENTS = {
    "E01": ("Record one-day declines in the Nasdaq Composite and the Dow after the March 2000 consumer price report",
            "2000-04-14", "crash day: same-day reaction to the CPI release",
            "Documented crash day of the April 2000 selloff. No official source located dates the selloff to the day; the NBER "
            "chronology cited in the submitted Table 2 dates recessions by month only.",
            [(4, "UPI, 'Nasdaq, Dow drop in record plunge', 14 April 2000",
              "https://www.upi.com/Archives/2000/04/14/Nasdaq-Dow-drop-in-record-plunge/4179955684800/",
              "On 14 Apr 2000 the Nasdaq fell 9.7% and the Dow 5.6%, their largest single-day point losses to that date, after a "
              "Labor Department report of a sharp rise in March consumer prices.", FETCHED)]),
    "E02": ("Terrorist attacks of 11 September 2001; U.S. equity markets closed 11 to 14 September",
            "2001-09-11", "market reopening: first day equity prices could react",
            "First trading day after the four-day closure.",
            [(1, "Federal Reserve History, 'The Federal Reserve's Response to the September 11 Terrorist Attacks'",
              "https://www.federalreservehistory.org/essays/september-11",
              "U.S. equity markets were closed for four days; NYSE trading did not resume until Monday 17 September 2001.", FETCHED)]),
    "E03": ("WorldCom Inc. and U.S. subsidiaries filed voluntary Chapter 11 petitions, U.S. Bankruptcy Court S.D.N.Y., "
            "Case No. 02-13533", "2002-07-21 (Sunday)", "first trading reaction after a weekend filing",
            "First trading day after the filing. Revision correction: the submitted panel used 2002-07-24, a rebound day with no "
            "recorded rationale.",
            [(1, "WorldCom Inc., Form 8-K filed with the SEC, report date 21 July 2002",
              "https://www.sec.gov/Archives/edgar/data/0000723527/000090951802000544/mv7-22_8k.txt",
              "WorldCom filed voluntary Chapter 11 petitions on 21 July 2002 in the U.S. Bankruptcy Court for the Southern District "
              "of New York, Case No. 02-13533.", FETCHED),
             (4, "UPI, 'Stocks end day in the red', 22 July 2002",
              "https://www.upi.com/Business_News/2002/07/22/Stocks-end-day-in-the-red/55251027372037/",
              "On 22 Jul 2002 the Dow fell 234.82 points (2.93%) and the S&P 500 27.92 points to 819.83; the report attributes the "
              "decline to weak earnings together with WorldCom's Sunday bankruptcy filing.", FETCHED),
             (4, "CNN Money, 'Dow surges nearly 500 points in 2nd best one-day gain ever', 24 July 2002",
              "https://money.cnn.com/2002/07/24/markets/markets_newyork/index.htm",
              "Corroborates the 24 July 2002 rebound (Dow up 488.95 points). Located through search but not retrievable; the "
              "rebound is verified from the archived SPY prices instead.", "not retrievable (HTTP 503); not relied on")]),
    "E04": ("Unwind of quantitative long/short equity portfolios in the week of 6 August 2007; dislocation of the euro money "
            "market on the morning of 9 August 2007 and an ECB fine-tuning operation",
            "2007-08-06 (start of the unwind); 2007-08-09 (money-market dislocation)",
            "central-bank intervention day inside the week of quant-fund losses",
            "Documented money-market dislocation day inside the documented quant-loss week. Why 9 rather than 6 August was "
            "chosen was not recorded.",
            [(1, "European Central Bank, Monthly Bulletin, September 2007, Box 3",
              "https://www.ecb.europa.eu/pub/pdf/other/mb200709_focus03.en.pdf",
              "On the morning of 9 August 2007 a dislocation of the euro money market occurred; the ECB conducted an overnight "
              "fine-tuning operation in which 49 banks bid EUR 94.8 billion.", PDF),
             (3, "Khandani and Lo, 'What Happened To The Quants In August 2007?', NBER Working Paper 14465",
              "https://www.nber.org/papers/w14465",
              "Quantitative hedge funds suffered losses during the week of 6 August 2007, with a sustained unwind starting at the "
              "open on 6 August.", FETCHED)]),
    "E05": ("Lehman Brothers Holdings Inc. filed for Chapter 11 on 15 September 2008; on 29 September 2008 the House rejected "
            "the Emergency Economic Stabilization Act vehicle, H.R. 3997, 205 to 228",
            "2008-09-15 (onset: Lehman filing); 2008-09-29 (House vote)",
            "policy-event day with same-day market reaction; not the onset",
            "Largest SPY fall of the crisis window, on the day of the House vote. The Lehman filing lies 10 trading days earlier "
            "inside the same window. Not the onset date.",
            [(1, "U.S. Securities and Exchange Commission, press release 2008-198, 15 September 2008",
              "https://www.sec.gov/news/press/2008/2008-198.htm",
              "Lehman Brothers Holdings Inc. filed for Chapter 11 bankruptcy protection on 15 September 2008.", FETCHED),
             (1, "Office of the Clerk, U.S. House of Representatives, Roll Call 674, 29 September 2008",
              "https://clerk.house.gov/Votes/2008674",
              "At 2:07 p.m. on 29 September 2008 the House failed to concur in the Senate amendment to H.R. 3997, 205 ayes to "
              "228 noes.", FETCHED)]),
    "E06": ("Flash Crash of 6 May 2010", "2010-05-06", "crash day (event occurrence)", "Day of the documented event.",
            [(1, "CFTC and SEC staffs, 'Findings Regarding the Market Events of May 6, 2010', 30 September 2010",
              "https://www.cftc.gov/MarketReports/StaffReportonMay6MarketEvents/staffreport050610marketevents.html",
              "The joint staff report examines the market events of 6 May 2010.", FETCHED)]),
    "E07": ("Standard & Poor's lowered the U.S. long-term sovereign rating from AAA to AA+ on Friday 5 August 2011",
            "2011-08-05", "first trading reaction after the announcement",
            "First trading day after the downgrade.",
            [(1, "Board of Governors of the Federal Reserve System, press release 'Agencies issue guidance on federal debt', 5 August 2011",
              "https://www.federalreserve.gov/newsevents/pressreleases/bcreg20110805a.htm",
              "On 5 August 2011 Standard & Poor's lowered the long-term rating of the U.S. government and federal agencies from "
              "AAA to AA+.", FETCHED),
             (4, "CBS News, 'Dow plummets 634 points amid downgrade backlash', 8 August 2011",
              "https://www.cbsnews.com/news/dow-plummets-634-points-amid-downgrade-backlash/",
              "On 8 Aug 2011 the S&P 500 fell 79.92 points (6.7%), on the first trading day after the Friday downgrade.", FETCHED)]),
    "E08": ("People's Bank of China changed the RMB central-parity quotation, effective 11 August 2015; extreme U.S. equity "
            "volatility at the open on Monday 24 August 2015",
            "2015-08-11 (policy announcement); 2015-08-24 (market event)", "market-reaction (crash) day documented by the SEC",
            "Documented U.S. market-reaction day. The attribution to the China devaluation follows the event label and is not "
            "taken from the SEC note.",
            [(1, "People's Bank of China, announcement on improving quotation of the RMB central parity, 11 August 2015",
              "https://www.pbc.gov.cn/english/130721/2025080815064585842/index.html",
              "Effective 11 August 2015, central-parity quotes refer to the previous day's interbank closing rate.", FETCHED),
             (1, "SEC Division of Trading and Markets, Office of Analytics and Research, 'Research Note: Equity Market Volatility "
                 "on August 24, 2015', December 2015",
              "https://www.sec.gov/marketstructure/research/equity_market_volatility.pdf",
              "On Monday 24 August 2015 U.S. equity markets had unusual volatility around the open; SPY opened 5.2% below the "
              "previous close, was 7.8% down by 9:35 and closed down 4.2%.", PDF)]),
    "E09": ("Equity market turbulence of Monday 5 February 2018 and the role of exchange-traded volatility products",
            "2018-02-05", "crash day (event occurrence)", "Day of the documented event.",
            [(1, "BIS Quarterly Review, March 2018, 'The equity market turbulence of 5 February: the role of exchange-traded "
                 "volatility products'",
              "https://www.bis.org/publ/qtrpdf/r_qt1803t.htm",
              "On Monday 5 February 2018 the S&P 500 fell about 4% and the VIX rose 20 points, its largest daily rise since 1987.", FETCHED)]),
    "E10": ("Fourth-quarter 2018 equity decline; on 24 December 2018 the S&P 500 fell 2.7% into bear-market territory",
            "2018-12-24", "trough: lowest close of the decline",
            "Trough day of the Q4 2018 decline, confirmed in the archived prices.",
            [(4, "UPI, 'Dow plummets 653 points in worst Christmas Eve ever; S&P, Nasdaq also fall', 24 December 2018",
              "https://www.upi.com/Top_News/US/2018/12/24/Dow-plummets-653-points-in-worst-Christmas-Eve-ever-SP-Nasdaq-also-fall/9921545675241/",
              "On 24 Dec 2018 the Dow fell 653.17 points (2.91%) and the S&P 500 2.7%, bringing it just into bear-market territory.", FETCHED),
             (1, "BIS press release on the March 2019 Quarterly Review, 5 March 2019",
              "https://www.bis.org/press/p190305.htm",
              "U.S. stock prices fell sharply in December 2018 on concern that monetary policy would keep tightening as the outlook "
              "weakened.", FETCHED)]),
    "E11": ("FOMC cut the federal funds target range to 0 to 1/4 percent on Sunday 15 March 2020; on Monday 16 March 2020 a "
            "Level 1 market-wide circuit breaker halt was triggered at the open",
            "2020-03-15 (policy announcement); 2020-03-16 (market event)",
            "first trading reaction after a weekend policy announcement (crash day)",
            "First trading day after the emergency rate cut.",
            [(1, "Board of Governors of the Federal Reserve System, FOMC statement, 15 March 2020",
              "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm",
              "Released 5:00 p.m. EDT on 15 March 2020; the target range was lowered to 0 to 1/4 percent.", FETCHED),
             (1, "Report of the Market-Wide Circuit Breaker Working Group (NYSE; filed with the SEC)",
              "https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf",
              "On Monday 16 March 2020 the Level 1 (7%) threshold was crossed at 9:30:01 a.m. and the day ended 12% below the "
              "previous close; the Federal Reserve had cut rates to nearly zero on the Sunday.", PDF)]),
    "E12": ("BLS reported an 8.6% 12-month CPI increase for May 2022 on 10 June 2022; on Monday 13 June 2022 the S&P 500 "
            "closed in bear-market territory", "2022-06-10 (CPI release); 2022-06-13 (market event)",
            "bear-market close, second trading day after the CPI release",
            "Day the S&P 500 closed in bear-market territory after the CPI release. Not the release day (10 June 2022, itself a "
            "trading day).",
            [(1, "U.S. Bureau of Labor Statistics, The Economics Daily, 'Consumer prices up 8.6 percent over year ended May 2022'",
              "https://www.bls.gov/opub/ted/2022/consumer-prices-up-8-6-percent-over-year-ended-may-2022.htm",
              "The CPI-U rose 8.6% over the 12 months ending May 2022; the news release is dated 10 June 2022.", FETCHED),
             (4, "UPI, 'S&P 500 closes in bear market territory, reaching lowest level since 2021', 13 June 2022",
              "https://www.upi.com/Top_News/US/2022/06/13/SP-500-closes-bear-market-territory-lowest-level-2021/3721655151493/",
              "On 13 Jun 2022 the S&P 500 fell 3.88% to 3,749.63 and closed in bear-market territory.", FETCHED)]),
    "E13": ("Silicon Valley Bank closed on Friday 10 March 2023 with the FDIC as receiver; joint Treasury, Federal Reserve and "
            "FDIC statement on Sunday 12 March 2023", "2023-03-10 (bank failure); 2023-03-12 (policy statement)",
            "first trading day after the weekend policy announcement",
            "First trading day after the joint statement. The larger SPY fall within five days (9 March) preceded the failure.",
            [(1, "Federal Deposit Insurance Corporation, Silicon Valley Bank failed-bank page",
              "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/silicon-valley.html",
              "SVB was closed on Friday 10 March 2023 by the California Department of Financial Protection and Innovation; the "
              "FDIC was named receiver.", FETCHED),
             (1, "Board of Governors of the Federal Reserve System, Joint Statement by Treasury, Federal Reserve, and FDIC, 12 March 2023",
              "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm",
              "Released 6:15 p.m. EDT on 12 March 2023; depositors would have access to all their money starting Monday 13 March.", FETCHED)]),
}


def keyword_search() -> pd.DataFrame:
    pattern = re.compile(r"candidate|exclu|overlap|priorit|shortlist|screen", re.IGNORECASE)
    hits = []
    for base in [ROOT / "data", ROOT / "docs", ROOT / "config", ROOT / "archive", ROOT / "README.md", ROOT / "paper" / "manuscript.tex"]:
        if not base.exists():
            continue
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for path in files:
            if path.suffix.lower() not in {".csv", ".md", ".txt", ".json", ".tex", ".py"} or "node_modules" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line) and re.search(r"event", line, re.IGNORECASE):
                    hits.append({"file": path.relative_to(ROOT).as_posix(), "line": i, "text": line.strip()[:240]})
    return pd.DataFrame(hits)


def main() -> None:
    df, events = load_asset("SPY")
    table = spy_events().set_index("event_id")
    submitted = load_seed("submitted").set_index("event_id")
    spans = event_spans(df, events, HALF_WINDOW)
    r, price = df["log_return"].to_numpy(float), df["price"].to_numpy(float)
    rows, source_rows = [], []
    for _, s in spans.iterrows():
        eid, c = s["event_id"], int(s["center"])
        lo5, hi5 = max(0, c - 5), min(len(df) - 1, c + 5)
        lo20, hi20 = max(0, c - 20), min(len(df) - 1, c + 20)
        worst5 = lo5 + int(np.argmin(r[lo5: hi5 + 1]))
        p = price[lo20: hi20 + 1]
        trough = lo20 + int(np.argmin(p / np.maximum.accumulate(p)))
        underlying, underlying_date, anchor, rationale, sources = EVENTS[eid]
        rows.append({
            "event_id": eid, "event_label_submitted": submitted.loc[eid, "event_name"], "event_label": table.loc[eid, "event_name"],
            "underlying_documented_event": underlying, "underlying_event_date": underlying_date,
            "reference_date_submitted": submitted.loc[eid, "calendar_date"], "reference_date": table.loc[eid, "calendar_date"],
            "spy_trading_date": table.loc[eid, "mapped_trading_date"],
            "reference_date_changed_in_revision": submitted.loc[eid, "calendar_date"] != table.loc[eid, "calendar_date"],
            "anchor_type": anchor, "revision_rationale": rationale,
            "rationale_status": "revision rationale; the selection rationale was not recorded when the panel was built",
            "strongest_source_tier": min(src[0] for src in sources if src[4] != "not retrievable (HTTP 503); not relied on"),
            "n_sources_relied_on": sum(1 for src in sources if src[4] != "not retrievable (HTTP 503); not relied on"),
            "spy_return_on_reference_date_pct": round(100 * (np.exp(r[c]) - 1), 2),
            "largest_spy_fall_within_5_days_date": df["date"].iloc[worst5].date().isoformat(),
            "largest_spy_fall_within_5_days_pct": round(100 * (np.exp(r[worst5]) - 1), 2),
            "severity_window_trough_date": df["date"].iloc[trough].date().isoformat(),
            "trough_minus_reference_trading_days": trough - c,
            "window_start": df["date"].iloc[s["lo"]].date().isoformat(), "window_end": df["date"].iloc[s["hi"]].date().isoformat(),
        })
        for tier, citation, url, prop, how in sources:
            source_rows.append({"event_id": eid, "tier": tier, "citation": citation, "url": url, "proposition_supported": prop,
                                "verification": how, "retrieved": RETRIEVED})
    prov = pd.DataFrame(rows)
    prov.insert(0, "event_panel", PANEL)
    prov.to_csv(OUT / "event_provenance.csv", index=False)
    pd.DataFrame(source_rows).to_csv(OUT / "event_sources.csv", index=False)

    gaps = []
    for i in range(len(spans) - 1):
        a, b = spans.iloc[i], spans.iloc[i + 1]
        gaps.append({"earlier_event": a["event_id"], "later_event": b["event_id"],
                     "earlier_window_end": df["date"].iloc[a["hi"]].date().isoformat(),
                     "later_window_start": df["date"].iloc[b["lo"]].date().isoformat(),
                     "trading_days_between_windows": int(b["lo"] - a["hi"] - 1), "windows_overlap": bool(b["lo"] <= a["hi"])})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(OUT / "adjacent_window_gaps.csv", index=False)

    hits = keyword_search()
    hits.to_csv(OUT / "keyword_search_candidate_or_overlap_rules.csv", index=False)

    # Asset-level check: every observable window on the other assets is also disjoint.
    asset_overlaps = []
    for ticker in ["QQQ", "IWM", "HYG", "TLT"]:
        adf, aev = load_asset(ticker)
        asp = event_spans(adf, aev, HALF_WINDOW)
        asset_overlaps.append({"ticker": ticker, "observable_events": len(asp),
                               "min_trading_days_between_windows": int(min(asp["lo"].iloc[1:].to_numpy() - asp["hi"].iloc[:-1].to_numpy() - 1)),
                               "any_overlap": bool((asp["lo"].iloc[1:].to_numpy() <= asp["hi"].iloc[:-1].to_numpy()).any())})
    pd.DataFrame(asset_overlaps).to_csv(OUT / "adjacent_window_gaps_other_assets.csv", index=False)

    pd.DataFrame([{
        "event_panel": PANEL, "events": len(prov),
        "reference_dates_changed_from_submitted": ",".join(prov.loc[prov["reference_date_changed_in_revision"], "event_id"]),
        "events_with_official_source_tier1": int((prov["strongest_source_tier"] == 1).sum()),
        "events_with_academic_source_only_as_strongest_tier3": int((prov["strongest_source_tier"] == 3).sum()),
        "events_resting_on_contemporaneous_reporting_only_tier4": ",".join(prov.loc[prov["strongest_source_tier"] == 4, "event_id"]),
        "reference_dates_with_positive_spy_return": ",".join(prov.loc[prov["spy_return_on_reference_date_pct"] > 0, "event_id"]) or "none",
        "smallest_gap_between_spy_windows_trading_days": int(gaps["trading_days_between_windows"].min()),
        "any_spy_window_overlap": bool(gaps["windows_overlap"].any()),
        "keyword_hits_reviewed": len(hits),
        "candidate_list_exists": False, "overlap_priority_rule_exists": False,
    }]).to_csv(OUT / "provenance_summary.csv", index=False)

    pd.set_option("display.width", 260)
    print(prov[["event_id", "event_label", "underlying_event_date", "reference_date", "anchor_type", "strongest_source_tier",
                "spy_return_on_reference_date_pct"]].to_string(index=False))
    print(gaps.to_string(index=False))
    print(pd.read_csv(OUT / "provenance_summary.csv").T.to_string())


if __name__ == "__main__":
    main()
