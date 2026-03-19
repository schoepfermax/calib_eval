from setuptools import setup, find_packages
import os
from glob import glob

package_name = "calib_eval"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config/reference", glob("config/reference/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="srijan",
    maintainer_email="www.srijangupta@gmail.com",
    description="Camera-LiDAR calibration evaluation pipeline",
    license="MIT",
    entry_points={
        "console_scripts": [
            # Preprocessor
            "data_preprocessor_node = calib_eval.data_preprocessor_node:main",

            # Reference publisher
            "reference_extrinsics_publisher = calib_eval.reference_extrinsics_publisher:main",

            # Model wrappers
            "supervised_model_node = calib_eval.supervised_model_node:main",
            "calib_model_node = calib_eval.calib_model_node:main",

            # Evaluators + pipeline
            "reprojection_evaluator_node = calib_eval.reprojection_evaluator_node:main",
            "edge_alignment_evaluator_node = calib_eval.edge_alignment_evaluator_node:main",
            "baseline_deviation_evaluator_node = calib_eval.baseline_deviation_evaluator_node:main",
            "evaluation_pipeline_node = calib_eval.evaluation_pipeline_node:main",

            # Utilities / checks (optional)
            "check_dataset_loader = calib_eval.check_dataset_loader:main",

            # 2D pipeline
            "offline2d = calib_eval.offline2d:main",
            "online2d = calib_eval.online2d:main",
        ],
    },
)