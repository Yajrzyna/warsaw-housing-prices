import numpy as np
import pandas as pd

from real_estate.cleaning.rcn_clean import RULES, cleaning_log, flag


def _row(**kw):
    base = dict(
        transaction_id="t", doc_ref="d", date=pd.Timestamp("2020-06-01"), year=2020.0,
        transaction_type=1, market=2, seller_type=3, buyer_type=3, share=1.0, right_type=3,
        price_unit=500_000.0, price_transaction=500_000.0, area_m2=50.0, rooms=2, floor=3,
        ancillary_area_m2=np.nan, deed_n_units=1, deed_n_residential=1,
        street="ulica Testowa", house_no="1", district="Wola",
    )
    base.update(kw)
    return base


def _frame(special):
    # 60 normal peers with small price dispersion, then the rows under test
    rng = np.random.default_rng(0)
    peers = [_row(transaction_id=f"p{i}", house_no=str(100 + i),
                  price_unit=500_000.0 * float(np.exp(rng.normal(0, 0.1)))) for i in range(60)]
    df = pd.DataFrame(peers + special)
    for c in ["transaction_type", "market", "seller_type", "buyer_type", "right_type", "rooms", "floor"]:
        df[c] = df[c].astype("Int64")
    return df


def test_rules_flag_expected_rows():
    special = [
        _row(transaction_id="share", house_no="s1", share=0.5),
        _row(transaction_id="bonif", house_no="s2", transaction_type=6),
        _row(transaction_id="legacy", house_no="s3", transaction_type=pd.NA),  # pre-2016: kept
        _row(transaction_id="bundle", house_no="s4", price_unit=np.nan, deed_n_units=2),
        _row(transaction_id="cheap", house_no="s5", price_unit=100_000.0),
        _row(transaction_id="dup", house_no="100"),  # same as peer p0 except price -> not dup
    ]
    out = flag(_frame(special)).set_index("transaction_id")
    assert out.loc["share", "excl_partial_share"]
    assert out.loc["bonif", "excl_non_market_type"]
    assert not out.loc["legacy", "excluded"]
    assert out.loc["bundle", "excl_bundle_price"]
    assert out.loc["cheap", "excl_outlier_low"]
    assert not out.loc["p1", "excluded"]


def test_exact_duplicate_keeps_first():
    df = _frame([_row(transaction_id="a", house_no="x"), _row(transaction_id="b", house_no="x")])
    out = flag(df).set_index("transaction_id")
    assert not out.loc["a", "excl_duplicate"] and out.loc["b", "excl_duplicate"]


def test_log_is_consistent():
    out = flag(_frame([_row(transaction_id="s", house_no="s", share=0.25)]))
    log = cleaning_log(out)
    assert list(log["rule"]) == list(RULES)
    assert log["removed_sequential"].sum() == out["excluded"].sum()
