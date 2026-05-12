import pandas as pd
import pytest
from ts_advisor.config import TSAdvisorConfig
from ts_advisor.data import prepare_supervised_frame, split_frame
from ts_advisor.features import add_engineered_features, build_feature_matrices
from ts_advisor.logging import ProgressLogger

@pytest.fixture
def tiny_raw_df():
    rows = []
    for i in range(16):
        cls = "TeamA" if i < 8 else "TeamB"
        obs = """1 EFFECT
=====
Cell disabled alarm and traffic stopped.
2 TROUBLE DESCRIPTION
=====
2.2 Configuration Data
---
BB6630 21.Q3
2.3 Logs
3 MEASURES
=====
Restart node.
4 CSR
=====
None""" if cls == "TeamA" else """1 EFFECT
=====
Core dump crash after upgrade.
2 TROUBLE DESCRIPTION
=====
2.2 Configuration Data
---
AIR6449 22.Q1
2.3 Logs
3 MEASURES
=====
Collect pmd.
4 CSR
=====
None"""
        rows.append({
            "General.Eriref": f"H{i:04d}",
            "Faulty product.Design Responsible MHO": cls,
            "General.Heading": f"Heading {cls} {i}",
            "Observation.Observation": obs,
            "Answer.Answer": "CAUSE OF FAULT\n----------\nsoftware issue\n----------",
            "General.Submitter priority": "B",
            "General.Superior MHO": "LTE-MS-RBS" if cls == "TeamA" else "ERS-RADIO",
            "Observation.Fault type": "Alarm" if cls == "TeamA" else "Crash",
            "Registered": "2023-01-01 01:00:00",
            "routing_time": 1,
            "routing_waiting_time": 1,
            "total_routing_time": 2,
            "routed": True,
            "General.Is duplicate TR": False,
            "General.Duplicate TRs": "",
            "Faulty product.Product no & R-State": "KDU137848 R24A",
            "Faulty product.Product no": "KDU137848",
            "Node level.Product no": "KDU137848",
        })
    return pd.DataFrame(rows)

@pytest.fixture
def tiny_config(tmp_path):
    return TSAdvisorConfig(run_mode="fast", min_samples=1, model_path=str(tmp_path / "bundle.joblib"))

@pytest.fixture
def tiny_matrices(tiny_raw_df, tiny_config):
    logger = ProgressLogger(enabled=False)
    df = prepare_supervised_frame(tiny_raw_df, tiny_config)
    df = add_engineered_features(df, tiny_config)
    train, val, test = split_frame(df, tiny_config)
    return build_feature_matrices(train, val, test, tiny_config, logger=logger)
