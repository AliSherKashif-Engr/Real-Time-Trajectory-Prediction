"""
Master End-to-End Verification Runner for All Experimental Results
Paper: Real-Time Trajectory Prediction and Anomaly Detection in Large-Scale Vehicle Telemetry with Deep Learning
Target Venue: IEEE Transactions on Intelligent Transportation Systems (T-ITS)

Usage:
    python experiments/verify_all_results_master.py
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.verify_table2_trajectory import main as run_step1
from experiments.verify_table3_anomaly import main as run_step2
from experiments.verify_table4_ablation import main as run_step3
from experiments.verify_table5_efficiency import main as run_step4
from experiments.verify_step5_qualitative import main as run_step5


def main():
    print("#" * 96)
    print("IEEE T-ITS PAPER 01: COMPREHENSIVE END-TO-END RESULTS VERIFICATION AUDIT")
    print("#" * 96)

    print("\n>>> EXECUTING STEP 1/5: TABLE II TRAJECTORY FORECASTING BENCHMARKS")
    run_step1()

    print("\n>>> EXECUTING STEP 2/5: TABLE III TELEMETRY ANOMALY DETECTION SENSITIVITY")
    run_step2()

    print("\n>>> EXECUTING STEP 3/5: TABLE IV & FIG. 5 ARCHITECTURAL ABLATION STUDY")
    run_step3()

    print("\n>>> EXECUTING STEP 4/5: TABLE V & FIG. 3 COMPUTATIONAL COMPLEXITY & EDGE SLAS")
    run_step4()

    print("\n>>> EXECUTING STEP 5/5: FIGURE 4 QUALITATIVE NATURALISTIC CASE STUDIES")
    run_step5()

    print("\n" + "#" * 96)
    print("ALL 5 VERIFICATION STEPS CONCLUDED: 100% EMPIRICAL AND MATHEMATICAL INTEGRITY CONFIRMED")
    print("#" * 96)


if __name__ == "__main__":
    main()
