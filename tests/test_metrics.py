import pytest

from benchmarks.client import RequestRecord, summarise_records


@pytest.fixture
def mixed_records():
    # A: 1 token, ok. B: 2 tokens, ok. C: failed.
    return [
        RequestRecord(t_send=0.0, t_end=0.3, t_tokens=[0.2], completion_tokens=1),
        RequestRecord(t_send=0.1, t_end=0.5, t_tokens=[0.3, 0.32], completion_tokens=2),
        RequestRecord(t_send=0.2, t_end=0.4, t_tokens=[], completion_tokens=0, error="Some error"),
    ]


def test_record_metrics_from_known_timestamps():
    # 11 tokens: first at 200 ms, then one every 20 ms (the mock server's behavior).
    t_tokens = [0.2 + 0.02 * i for i in range(11)]
    r = RequestRecord(t_send=0.0, t_end=0.4, t_tokens=t_tokens, completion_tokens=11)

    assert r.ttft == pytest.approx(0.2)
    assert r.tpot == pytest.approx(0.02)
    assert r.e2e == pytest.approx(0.4)
    assert len(r.itls) == 10


def test_single_token_responses():
    r = RequestRecord(t_send=0.0, t_end=0.3, t_tokens=[0.2], completion_tokens=1)

    assert r.ttft == pytest.approx(0.2)
    assert r.tpot is None
    assert r.e2e == pytest.approx(0.3)
    assert len(r.itls) == 0


def test_request_with_no_tokens():
    r = RequestRecord(t_send=0.0, t_end=0.3, t_tokens=[], completion_tokens=0)

    assert r.ttft is None
    assert r.tpot is None
    assert r.e2e == pytest.approx(0.3)
    assert len(r.itls) == 0


def test_failure_recording():
    r = RequestRecord(t_send=0.0, t_end=0.3, t_tokens=[0.2], completion_tokens=1, error="Some error")

    assert r.ttft == pytest.approx(0.2)
    assert r.tpot is None
    assert r.error == "Some error"


def test_summary_percentiles_use_only_successful_requests(mixed_records):
    results, _, _ = summarise_records(mixed_records, slo_ttft=0.25, slo_tpot=0.05)

    # E2E is t_end - t_send: A = 0.3, B = 0.4 (C failed, excluded).
    # With two values, percentiles interpolate: p50 = midpoint, p95 = 0.3 + 0.95 * 0.1, p99 = 0.3 + 0.99 * 0.1.
    assert results["e2e"]["p50"] == pytest.approx(0.35)
    assert results["e2e"]["p95"] == pytest.approx(0.395)
    assert results["e2e"]["p99"] == pytest.approx(0.399)


def test_failed_request_does_not_change_latency_percentiles(mixed_records):
    slow_failure = RequestRecord(t_send=0.0, t_end=5.0, t_tokens=[], completion_tokens=0, error="timeout")

    without, _, _ = summarise_records(mixed_records)
    with_failure, _, _ = summarise_records(mixed_records + [slow_failure])

    # A 5 s failure would drag the E2E tail up massively if it were counted.
    assert with_failure["e2e"]["p99"] == pytest.approx(without["e2e"]["p99"])


def test_error_rate(mixed_records):
    _, throughput, _ = summarise_records(mixed_records)

    assert throughput["error_rate"] == pytest.approx(1 / 3)


def test_goodput_and_slo_attainment_on_mixed_records(mixed_records):
    _, throughput, goodput = summarise_records(mixed_records, slo_ttft=0.25, slo_tpot=0.05)

    # Only B meets the SLO (A has no TPOT, C failed). Wall time = 0.5 - 0.0 = 0.5 s.
    assert goodput["requests_per_second"] == pytest.approx(1 / 0.5)
    assert throughput["requests_per_second"] == pytest.approx(2 / 0.5)
    assert goodput["slo_attainment"] == pytest.approx(1 / 3)


ALL_GOOD = [
    RequestRecord(t_send=0.0, t_end=0.4, t_tokens=[0.2, 0.22, 0.24], completion_tokens=3),
    RequestRecord(t_send=0.1, t_end=0.5, t_tokens=[0.3, 0.32, 0.34], completion_tokens=3),
]
ALL_FAILED = [
    RequestRecord(t_send=0.0, t_end=0.1, error="boom"),
    RequestRecord(t_send=0.1, t_end=0.2, error="boom"),
]
ALL_SLOW = [
    RequestRecord(t_send=0.0, t_end=3.0, t_tokens=[2.0, 2.02, 2.04], completion_tokens=3),
    RequestRecord(t_send=0.1, t_end=3.1, t_tokens=[2.1, 2.12, 2.14], completion_tokens=3),
]


@pytest.mark.parametrize("records", [ALL_GOOD, ALL_FAILED, ALL_SLOW], ids=["all_good", "all_failed", "all_slow"])
def test_goodput_never_exceeds_throughput(records):
    _, throughput, goodput = summarise_records(records, slo_ttft=0.5, slo_tpot=0.03)

    assert goodput["requests_per_second"] <= throughput["requests_per_second"]
    assert 0.0 <= goodput["slo_attainment"] <= 1.0
