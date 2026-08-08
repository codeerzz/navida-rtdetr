PYTHON ?= python3

.PHONY: help test test-color test-ros lint build clean

help:
	@echo "Targets:"
	@echo "  make test        Run the ROS-free unit test suite (no ROS/ZED needed -- works on a laptop)"
	@echo "  make test-color  Run only the color-classifier + label-vote unit tests"
	@echo "  make test-ros    Run the ROS/ZED-dependent tests (needs a sourced ROS 2 env with cv_bridge)"
	@echo "  make lint        Run ruff over the Python packages"
	@echo "  make build       colcon build the ROS 2 workspace (needs a sourced ROS 2 env)"
	@echo "  make clean       Remove Python/pytest caches"

# ROS-dependent tests self-skip via pytest.importorskip when rclpy isn't installed,
# so `make test` is safe to run as-is on a laptop with no ROS.
#
# fusion_pkg/test/ is intentionally NOT run here: as of this writing it only has the
# ament boilerplate tests (test_copyright.py, test_flake8.py, test_pep257.py), which
# import ament_copyright/ament_flake8/ament_pep257 -- only available inside a sourced
# ROS 2 install, not via pip. That's pre-existing, unrelated to this change; run those
# with `colcon test` from a ROS 2 environment instead.
test:
	cd rtdetr_zed_tracker && $(PYTHON) -m pytest test/ -v --ignore=test/test_yolo_world_mock.py

test-color:
	cd rtdetr_zed_tracker && $(PYTHON) -m pytest test/test_color_classifier.py test/test_label_vote.py -v

test-ros:
	cd rtdetr_zed_tracker && $(PYTHON) -m pytest test/test_color_classification_node.py -v

lint:
	ruff check rtdetr_zed_tracker/rtdetr_zed_tracker fusion_pkg/fusion_pkg

build:
	colcon build --symlink-install

clean:
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} +
