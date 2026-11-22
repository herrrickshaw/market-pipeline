"""bhavcopy_store.py -- LMDB OHLCV store: serialisation round-trip, build sources, readers, vacuum."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lmdb")
import bhavcopy_store as BS  # noqa: E402


def ohlcv(n=5, start="2024-01-01", base=100.0, vol=1000):
    idx = pd.date_range(start, periods=n, freq="B", name="Date")
    c = base + np.arange(n) * 0.5
    return pd.DataFrame(
        {"Open": c - 0.1, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.full(n, vol)},
        index=idx,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "CACHE", tmp_path)
    monkeypatch.setattr(BS, "STORE", tmp_path / "ohlcv.lmdb")
    monkeypatch.setattr(BS, "CLEANED", tmp_path / "cleaned_long.parquet")
    return tmp_path


# ── serialisation ─────────────────────────────────────────────────────────────
def test_roundtrip_restores_float64_prices_and_int_volume():
    df = ohlcv()
    out = BS._de(BS._ser(df))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert all(out[c].dtype == np.float64 for c in ("Open", "High", "Low", "Close"))
    assert out["Volume"].dtype == np.int64
    assert out.index.name == "Date" and list(out.index) == list(df.index)
    assert out["Close"].to_numpy() == pytest.approx(df["Close"].to_numpy(), rel=1e-6)


def test_prices_are_stored_as_float32_precision():
    df = ohlcv()
    df.loc[df.index[0], "Close"] = 123.456789012345
    out = BS._de(BS._ser(df))
    assert out["Close"].iloc[0] == float(np.float32(123.456789012345))
    assert out["Close"].iloc[0] != 123.456789012345


def test_volume_is_coerced_nan_to_zero_and_text_to_number():
    df = ohlcv(3)
    df["Volume"] = ["1500", None, "abc"]
    out = BS._de(BS._ser(df))
    assert list(out["Volume"]) == [1500, 0, 0]


def test_de_sorts_by_date_and_handles_frames_without_a_date_column():
    df = ohlcv(4).iloc[[2, 0, 3, 1]]
    out = BS._de(BS._ser(df))
    assert out.index.is_monotonic_increasing
    plain = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=3), "Close": [3.0, 1.0, 2.0]})
    import io

    import pyarrow as pa
    import pyarrow.feather as feather

    buf = io.BytesIO()
    feather.write_feather(pa.Table.from_pandas(plain, preserve_index=False), buf)
    out = BS._de(buf.getvalue())  # first column becomes the index
    assert out.index.name == "when" and list(out["Close"]) == [3.0, 1.0, 2.0]


def test_ser_tolerates_missing_price_columns():
    df = ohlcv(3)[["Close", "Volume"]]
    out = BS._de(BS._ser(df))
    assert list(out.columns) == ["Close", "Volume"]


# ── build from a dict ─────────────────────────────────────────────────────────
def test_build_from_dict_skips_empty_frames_and_records_meta(store, capsys):
    hist = {
        "AAA": ohlcv(5),
        "BBB": ohlcv(3, start="2024-03-01"),
        "EMPTY": ohlcv(3).iloc[0:0],
        "NONE": None,
    }
    assert BS.build(hist) == 2
    assert BS.symbols() == sorted(BS.symbols()) and set(BS.symbols()) == {"AAA", "BBB"}
    info = BS.info()
    assert info["symbols"] == 2 and info["latest_bar"].startswith("2024-03-05")
    assert info["size_mb"] >= 0
    assert "LMDB store built: 2 symbols" in capsys.readouterr().out


def test_build_latest_bar_is_the_max_across_symbols(store):
    BS.build(
        {
            "OLD": ohlcv(3, start="2024-06-03"),
            "NEW": ohlcv(3, start="2025-01-06"),
            "MID": ohlcv(3, start="2024-09-02"),
        },
        verbose=False,
    )
    assert BS.info()["latest_bar"].startswith("2025-01-08")


def test_build_is_a_fresh_rebuild_not_an_append(store):
    BS.build({"AAA": ohlcv(), "BBB": ohlcv()}, verbose=False)
    BS.build({"CCC": ohlcv()}, verbose=False)
    assert BS.symbols() == ["CCC"] and BS.get("AAA") is None


def test_build_quiet_prints_nothing(store, capsys):
    BS.build({"AAA": ohlcv()}, verbose=False)
    assert capsys.readouterr().out == ""


# ── build from parquet seeds ──────────────────────────────────────────────────
def write_seed(path, symbols_to_frames):
    rows = []
    for sym, df in symbols_to_frames.items():
        t = df.reset_index()
        t["Symbol"] = sym
        t["Date"] = t["Date"].astype(str)
        t["Extra"] = 1
        rows.append(t)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def test_build_ingests_cleaned_and_market_seeds(store, capsys):
    write_seed(store / "cleaned_long.parquet", {"RELIANCE": ohlcv(4)})
    write_seed(store / "cleaned_long_US.parquet", {"AAPL": ohlcv(6, base=200.0)})
    n = BS.build()
    assert n == 2 and set(BS.symbols()) == {"RELIANCE", "AAPL"}
    aapl = BS.get("AAPL")
    assert list(aapl.columns) == ["Open", "High", "Low", "Close", "Volume"]  # Extra dropped
    assert len(aapl) == 6 and aapl["Close"].iloc[0] == pytest.approx(200.0)
    assert (
        "ingesting 2 market seed(s): cleaned_long.parquet, cleaned_long_US.parquet"
        in capsys.readouterr().out
    )


def test_build_without_any_seed_raises(store):
    with pytest.raises(FileNotFoundError, match="no cleaned_long"):
        BS.build()


def test_build_seed_dates_are_parsed_and_sorted(store):
    df = ohlcv(5).iloc[[3, 1, 4, 0, 2]]
    write_seed(store / "cleaned_long_JP.parquet", {"7203.T": df})
    BS.build(verbose=False)
    out = BS.get("7203.T")
    assert out.index.is_monotonic_increasing and out.index.dtype.kind == "M"


# ── readers ───────────────────────────────────────────────────────────────────
def test_readers_return_empty_when_store_missing(store):
    assert BS.get("X") is None and BS.get_many(["X"]) == {} and BS.load_all() == {}
    assert BS.symbols() == [] and BS.info() == {}


def test_get_and_get_many(store):
    BS.build({"AAA": ohlcv(5), "BBB": ohlcv(3), "CCC": ohlcv(4)}, verbose=False)
    assert len(BS.get("AAA")) == 5 and BS.get("NOPE") is None
    many = BS.get_many(["CCC", "NOPE", "AAA"])
    assert set(many) == {"CCC", "AAA"} and len(many["CCC"]) == 4
    assert BS.get_many([]) == {}


def test_load_all_excludes_meta_and_symbols_lists_keys(store):
    BS.build({"AAA": ohlcv(2), "BBB": ohlcv(3)}, verbose=False)
    out = BS.load_all()
    assert set(out) == {"AAA", "BBB"} and "__meta__" not in out
    assert len(out["BBB"]) == 3
    assert BS.symbols() == ["AAA", "BBB"]


def test_info_for_store_without_meta(store):
    import lmdb

    env = lmdb.open(str(BS.STORE), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"AAA", BS._ser(ohlcv(2)))
    env.close()
    info = BS.info()
    assert info["symbols"] == 0 and info["latest_bar"] is None


# ── vacuum ────────────────────────────────────────────────────────────────────
def test_vacuum_deletes_raw_csvs_only(store, capsys):
    (store / "nse").mkdir()
    (store / "bse").mkdir()
    (store / "nse" / "a.csv").write_bytes(b"x" * 1000)
    (store / "nse" / "keep.parquet").write_bytes(b"y" * 10)
    (store / "bse" / "b.csv").write_bytes(b"z" * 500)
    assert BS.vacuum() == 1500
    assert not (store / "nse" / "a.csv").exists() and not (store / "bse" / "b.csv").exists()
    assert (store / "nse" / "keep.parquet").exists()
    assert "reclaimed 0.0 MB" in capsys.readouterr().out
    assert BS.vacuum(verbose=False) == 0


def test_vacuum_without_directories(store):
    assert BS.vacuum(verbose=False) == 0
