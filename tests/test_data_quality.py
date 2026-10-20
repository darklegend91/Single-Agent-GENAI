import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from app.config import Settings
from app.data.dictionary import FIELD_DICTIONARY
from app.data.ingestion import stage_dataset
from app.data.quality import profile_csv
from app.data.repository import SourceRepository
from app.errors import ValidationFailure
from app.security import Principal


FIXTURE = Path(__file__).parent / "fixtures/customers.csv"
OWNER = Principal(subject="tester", customer_ids={"U0000000001", "U0000000002"})


def configured(tmp_path, source=FIXTURE, **kwargs):
    return Settings(credential_mapping={"test-only": OWNER}, dataset_path=source,
                    staging_dir=tmp_path / "staging", database_path=tmp_path / "active.sqlite3",
                    active_manifest_path=tmp_path / "active.json", **kwargs)


def write_csv(tmp_path, rows, headers=None):
    path = tmp_path / "input.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_profile_is_aggregate_only_and_preserves_numeric_date_uncertainty():
    report = profile_csv(FIXTURE, chunk_size=1)
    assert report.row_count == 2
    assert report.missing_counts["PlanPricePaise"] == 1
    assert report.unresolved_timestamp_counts["RegistrationAtUTC"] == 1
    assert report.timestamp_shapes["RegistrationAtUTC"] == {"digits_14": 1}
    assert report.invalid_month_counts["LatestRewardMonth"] == 1
    assert report.categorical_distinct_counts["CustomerSegment"] == 2
    assert "Synthetic Alpha" not in report.model_dump_json()
    assert "U0000000001" not in report.model_dump_json()
    assert "UNEXPLAINED" not in report.model_dump_json()


def test_staging_keeps_exact_large_paise_and_does_not_publish(tmp_path):
    settings = configured(tmp_path)
    staged = stage_dataset(settings)
    record = SourceRepository(staged.sqlite_path).lookup(OWNER, "U0000000001")
    assert record.values["TotalSettledPaise"] == 9007199254740993
    assert record.values["RegistrationAtUTC"] == "20251012130405"
    assert record.normalized_timestamps == {}
    assert record.provenance["TotalSettledPaise"].source_row == 2
    assert record.provenance["TotalSettledPaise"].dataset_version == staged.version
    assert staged.publication_state == "staged"
    assert not settings.active_manifest_path.exists()
    assert not settings.database_path.exists()
    with sqlite3.connect(staged.sqlite_path) as connection:
        assert connection.execute("PRAGMA table_info(customers)").fetchone()[5] == 1


def test_explicit_timestamp_format_is_required_and_invalid_dates_are_counted(tmp_path):
    source = write_csv(tmp_path, [
        {"UserID": "U0000000001", "RegistrationAtUTC": "20251012130405"},
        {"UserID": "U0000000002", "RegistrationAtUTC": "20251312130405"},
    ])
    report = profile_csv(source, 1, timestamp_formats={"RegistrationAtUTC": "%Y%m%d%H%M%S"})
    assert report.invalid_timestamp_counts["RegistrationAtUTC"] == 1
    staged = stage_dataset(configured(tmp_path, source, timestamp_formats={"RegistrationAtUTC": "%Y%m%d%H%M%S"}))
    repository = SourceRepository(staged.sqlite_path)
    assert repository.lookup(OWNER, "U0000000001").normalized_timestamps["RegistrationAtUTC"] == "2025-10-12T13:04:05+00:00"
    assert repository.lookup(OWNER, "U0000000002").values["RegistrationAtUTC"] == "20251312130405"
    assert "unresolved_timestamp" in repository.lookup(OWNER, "U0000000002").limitations


@pytest.mark.parametrize("value", ["1.5", "1e3", "nan", "Infinity", "-2", "+3", "１２"])
def test_malformed_financial_values_reported_and_rejected(tmp_path, value):
    source = write_csv(tmp_path, [{"UserID": "U0000000001", "TotalSettledPaise": value}])
    assert profile_csv(source, 1).invalid_integer_counts["TotalSettledPaise"] == 1
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path, source))
    assert not list((tmp_path / "staging").glob("*.sqlite3"))


@pytest.mark.parametrize("user_id", ["", "U1", "U0000000001' OR 1=1--", " U0000000001"])
def test_invalid_required_keys_never_stage(tmp_path, user_id):
    source = write_csv(tmp_path, [{"UserID": user_id}])
    assert profile_csv(source, 1).invalid_user_id_count == 1
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path, source))


def test_duplicate_ids_detected_across_chunks_and_staging_rolls_back(tmp_path):
    source = write_csv(tmp_path, [{"UserID": "U0000000001"}] * 2)
    assert profile_csv(source, 1).duplicate_user_id_count == 1
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path, source, ingestion_batch_size=1))
    assert not list((tmp_path / "staging").glob("*.sqlite3"))


def test_missing_user_id_column_and_invalid_month_header_fail_closed(tmp_path):
    source = write_csv(tmp_path, [{"SyntheticName": "Synthetic"}])
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path, source))
    source = write_csv(tmp_path, [{"UserID": "U0000000001", "SettledRevenue202613Paise": "3"}])
    assert profile_csv(source, 1).invalid_month_columns == ["SettledRevenue202613Paise"]
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path, source))


def test_quality_checks_distinct_count_consistency_without_business_inference(tmp_path):
    source = write_csv(tmp_path, [{"UserID": "U0000000001", "PaymentAttemptCount": "1", "SuccessfulPaymentCount": "2", "FailedPaymentCount": "0", "RiskEventCount": "1", "ConfirmedRiskEvents": "1", "PendingRiskEvents": "1"}])
    report = profile_csv(source, 1)
    assert report.consistency_violation_counts["payment_outcomes_exceed_attempts"] == 1
    assert report.consistency_violation_counts["risk_outcomes_exceed_events"] == 1


def test_profile_cli_emits_only_sanitized_aggregate_json(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-m", "scripts.profile_data", "--csv", str(FIXTURE), "--output", str(output), "--chunk-size", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["row_count"] == 2
    assert "Synthetic Alpha" not in output.read_text() + result.stdout + result.stderr


def test_all_dictionary_fields_and_fixed_domains_are_known():
    assert len(FIELD_DICTIONARY) == 77
    assert FIELD_DICTIONARY["TotalSettledPaise"].unit == "paise"
    assert FIELD_DICTIONARY["ConfirmedRiskEvents"].unit == "count"
    assert FIELD_DICTIONARY["LastCancellationReasonCode"].code_meanings_known is False


def test_staging_bound_and_version_do_not_allow_path_escape(tmp_path):
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path), limit=0)
    with pytest.raises(ValidationFailure):
        stage_dataset(configured(tmp_path), version="../../active")
    staged = stage_dataset(configured(tmp_path), limit=1)
    assert staged.row_count == 1
    assert staged.sample_limit == 1


def test_quality_count_comparisons_do_not_overflow_fixed_width_integers(tmp_path):
    source = write_csv(tmp_path, [{"UserID": "U0000000001", "PaymentAttemptCount": "9223372036854775807", "SuccessfulPaymentCount": "9223372036854775807", "FailedPaymentCount": "1"}])
    assert profile_csv(source, 1).consistency_violation_counts["payment_outcomes_exceed_attempts"] == 1

