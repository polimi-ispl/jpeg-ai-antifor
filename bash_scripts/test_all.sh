#!/usr/bin/env bash
# Simple script to test all the detectors considered in the paper

# USER PARAMETERS (put your device configuration params here)
DEVICE=0  # PUT THERE YOUR GPU ID
DATA_DIR='/your/absolute/path/to/data'  # PUT THERE THE PATH TO THE TEST SET
RESULTS_DIR='/your/absolute/path/to/results'  # PUT THERE THE PATH TO THE RESULTS FOLDER

echo ""
echo "-------------------------------------------------"
echo "| Testing Wang2020-A for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Wang2020-A --weights_path ../utils/third_party/Wang2020CNNDetection/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Wang2020-A |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Wang2020-A
echo ""
echo "-------------------------------------------------"
echo "| Wang2020-A done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Wang2020-B for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Wang2020-B --weights_path ../utils/third_party/Wang2020CNNDetection/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Wang2020-B |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Wang2020-B
echo ""
echo "-------------------------------------------------"
echo "| Wang2020-B done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Gragnaniello2021 for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Grag2021_progan --weights_path ../utils/third_party/DMImageDetection_test_code/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Gragnaniello2021 |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Grag2021_progan
echo ""
echo "-------------------------------------------------"
echo "| Gragnaniello2021 done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Corvi2023 for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Corvi2023 --weights_path ../utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Corvi2023 |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Corvi2023
echo ""
echo "-------------------------------------------------"
echo "| Corvi2023 done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Ojha2023 for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Ojha2023 --weights_path ../utils/third_party/UniversalFakeDetect_test_code/pretrained_weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Ojha2023 |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Ojha2023
echo ""
echo "-------------------------------------------------"
echo "| Ojha2023 done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Cozzolino2024-A for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Cozzolino2024-A --weights_path ../utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Cozzolino2024-A |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Cozzolino2024-A
echo ""
echo "-------------------------------------------------"
echo "| Cozzolino2024-A done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing Cozzolino2024-B for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Cozzolino2024-B --weights_path ../utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for Cozzolino2024-B |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector Cozzolino2024-B
echo ""
echo "-------------------------------------------------"
echo "| Cozzolino2024-B done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing NPR for all tasks |"
echo "-------------------------------------------------"
python ../test_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector NPR --weights_path ../utils/third_party/NPR/weights --batch_size 128 --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for NPR |"
echo "-------------------------------------------------"
python ../compute_metrics.py --results_dir=${RESULTS_DIR} --detector NPR
echo ""
echo "-------------------------------------------------"
echo "| NPR done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing TruFor for all tasks |"
echo "-------------------------------------------------"
python ../test_splicing_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector TruFor --weights_path ../utils/third_party/TruFor/test_docker --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for TruFor |"
echo "-------------------------------------------------"
python ../compute_splicing_metrics.py --results_dir=${RESULTS_DIR} --detector TruFor
echo ""
echo "-------------------------------------------------"
echo "| TruFor done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing MMFusion for all tasks |"
echo "-------------------------------------------------"
python ../test_splicing_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector MMFusion --weights_path ../utils/third_party/MMFusion/weights --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for MMFusion |"
echo "-------------------------------------------------"
python ../compute_splicing_metrics.py --results_dir=${RESULTS_DIR} --detector MMFusion
echo ""
echo "-------------------------------------------------"
echo "| MMFusion done! |"
echo "-------------------------------------------------"

echo ""
echo "-------------------------------------------------"
echo "| Testing ImageForensicsOSN for all tasks |"
echo "-------------------------------------------------"
python ../test_splicing_detector.py --input_dir=${DATA_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector ImageForensicsOSN --weights_path ../utils/third_party/ImageForensicsOSN/weights --test_all
echo ""
echo "-------------------------------------------------"
echo "| Computing metrics for ImageForensicsOSN |"
echo "-------------------------------------------------"
python ../compute_splicing_metrics.py --results_dir=${RESULTS_DIR} --detector ImageForensicsOSN
echo ""
echo "-------------------------------------------------"
echo "| MMFusion done! |"
echo "-------------------------------------------------"
