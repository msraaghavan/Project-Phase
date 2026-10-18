"""The web app: pages, the static site data, headline numbers and the live demo."""

from pathlib import Path

import pytest

pytest.importorskip("httpx")
if not Path("results/results.json").exists():
    pytest.skip("needs a completed run in results/", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from rcm.report import site_data  # noqa: E402
from rcm.report.api import app  # noqa: E402

client = TestClient(app)


def test_pages_and_assets_are_served():
    page = client.get("/").text
    assert "Try it yourself" in page and '<script id="rcm-data">' in page
    assert client.get("/figures/fig1_bias_vs_pressure.png").status_code == 200
    assert client.get("/figures/..%2Fresults.json").status_code in (400, 404)


def test_summary_matches_results_json():
    import json

    res = json.loads(Path("results/results.json").read_text(encoding="utf-8"))
    s = client.get("/api/summary").json()
    assert s["contagion"]["memoryContagion"] == res["module3"]["contagion"]["memoryContagion"]
    assert s["best"]["mode"] in res["module4"] and len(s["figures"]) == 6
    assert any(r["backend"] == "tiny" for r in client.get("/api/runs").json())
    topics = client.get("/api/demo/options").json()["topics"]
    assert len(topics) == 5 and topics[0]["group"] == "Vorlan" and topics[0]["unfair"] == "hostile"


def test_demo_replays_the_sweep_exactly_and_edits_memory():
    import json

    res = json.loads(Path("results/results.json").read_text(encoding="utf-8"))
    ex = res["example_conversation"]                    # shared-kv, pi = 1.0, seed 0
    j = client.post("/api/demo", json={"arm": "shared-kv", "pressure": 1.0, "seed": 0,
                                       "intervention": "ablate"}).json()
    assert j["receiverBias"] == pytest.approx(ex["receiver_bias"])
    assert j["seqLen"] == len(ex["tokens"]) and len(j["turns"]) == 8
    assert j["edited"] is not None and j["edited"]["finalBias"] != j["finalBias"]
    assert {c["ch"] for c in j["cache"]} >= {"system", "probe", "agent", "peer"}
    assert j["nUnfair"] == 16 and j["nStatements"] == 16 and j["topic"]["group"] == "Vorlan"
    iso = client.post("/api/demo", json={"arm": "isolated", "pressure": 0.0, "seed": 0}).json()
    assert j["baselineBias"] == pytest.approx(iso["finalBias"])


def test_demo_rejects_bad_input():
    assert client.post("/api/demo", json={"arm": "telepathy"}).status_code == 400
    assert client.post("/api/demo", json={"pressure": 2}).status_code == 400


def test_site_data_matches_results_and_live_demo():
    import json

    if not site_data.is_built():
        pytest.skip("site data not built (python -m rcm.cli site)")
    data = site_data.load()
    res = json.loads(Path("results/results.json").read_text(encoding="utf-8"))
    ex = res["example_conversation"]                    # shared-kv, pi = 1.0, seed 0
    d = data["demos"]["0|shared-kv|16"]
    assert d["receiverBias"] == pytest.approx(ex["receiver_bias"], abs=1e-5)
    assert len(d["tok"]) == len(d["ch"]) == len(d["turn"]) == len(ex["tokens"])
    assert len(data["demos"]) == 5 * 9 * 3 and set(d["edits"]) == {"ablate", "gate"}
    tiny = next(r for r in data["runs"] if r["backend"] == "tiny")
    assert tiny["contagion"]["memoryContagion"] == res["module3"]["contagion"]["memoryContagion"]
    iso = data["demos"]["0|isolated|0"]
    assert data["baseline"]["0"] == pytest.approx(iso["finalBias"], abs=1e-5)
    # every combination the page can ask for exists and is consistent
    assert data["levels"] == [0, 2, 4, 6, 8, 10, 12, 14, 16]
    for t in data["topics"]:
        for arm in ("shared-kv", "text-only", "isolated"):
            for n in data["levels"]:
                rec = data["demos"][f"{t['seed']}|{arm}|{n}"]
                assert rec["nUnfair"] == n and len(rec["turns"]) == 8
                assert len(rec["tok"]) == len(rec["ch"]) == len(rec["turn"])
                assert set(rec["edits"]) == {"ablate", "gate"}
                assert ("e" in rec["ch"]) == (arm == "shared-kv")      # copied notes only when shared
                if arm == "isolated":                                  # hears nothing: same every time
                    assert rec["finalBias"] == pytest.approx(data["baseline"][str(t["seed"])], abs=1e-6)


def test_page_script_is_consistent_and_has_no_emoji():
    import re

    html = (site_data.PAGE).read_text(encoding="utf-8")
    page = re.sub(r'<script id="rcm-data">.*?</script>', "", html, flags=re.S)
    ids = set(re.findall(r'id="([\w-]+)"', page))
    used = set(re.findall(r'\$\("#([\w-]+)"', page)) | set(re.findall(r'getElementById\("([\w-]+)"\)', page)) \
        | set(re.findall(r'chart\("([\w-]+)"', page))
    assert used and used <= ids, used - ids
    slides = re.findall(r'<section class="slide"[^>]*id="([\w-]+)"', page)
    enter = re.search(r"const ENTER = \{(.*?)\n\};", page, re.S).group(1)
    keys = {k.strip('"') for k in re.findall(r'("?[\w-]+"?):\s', enter)}
    assert len(slides) == 21 and keys <= set(slides), keys - set(slides)
    assert not re.search("[\U0001F300-\U0001FAFF\u2600-\u27BF]", page)
    assert not re.search(r"&#x(1F|26|27)[0-9A-Fa-f]{2,3};", page)
