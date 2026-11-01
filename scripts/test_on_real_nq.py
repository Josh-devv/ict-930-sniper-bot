import sys
sys.path.insert(0, '.')

from scripts.run_calibrated_driver import run_calibrated_model

if __name__ == '__main__':
    print("================================================================")
    print("  EVALUATING 4H DRIVER PAIRING ON REAL NQ FUTURES DATA")
    print("================================================================")
    run_calibrated_model('sample_data/REAL_NQ_1min.csv', target_r=2.2, partial_r=1.5)
