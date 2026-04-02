# AutoDrive Geometry Lane

一个面向智能驾驶岗位展示的 OpenCV 车道感知项目。该项目将传统“画两条线”的入门代码，升级为**计算几何驱动的车道状态估计系统**。

## 项目亮点
- 计算几何核心：齐次直线表示、线交点/距离、消失点估计。
- 相机标定链路：支持棋盘格标定、去畸变、相机位姿参数化。
- 标定质量报告：自动生成重投影误差分布、坏帧剔除结果、可视化图。
- 真实物理尺度：基于内参 + 相机安装高度/俯仰角投影到地平面，输出米制车道宽度/偏移/曲率。
- 曲线车道与变道：二次曲线模型 + 时序状态机(FSM) + 置信度评分。
- 失效检测与降级：夜晚/雨天/逆光场景识别，异常时自动降级并保持输出连续。
- 工程化能力：模块化架构、CLI、可视化看板、自动评估脚本、单元测试。

## 目录结构

```text
autodrive_geometry_lane/
  src/autodrive_lane/
    config.py
    calibration/
      camera_model.py
      quality_report.py
    geometry/
      primitives.py
      homography.py
      metrics.py
    perception/
      feature_scene.py
      lane_modeling.py
      temporal_smoother.py
      lane_change_fsm.py
      overlay_renderer.py
      lane_pipeline.py
    visualization.py
  tools/
    calibrate_camera.py
    synthetic_benchmark.py
    video_metrics.py
  tests/
    test_geometry.py
    test_advanced_features.py
  run_demo.py
```

## 关键算法流程
1. 颜色-梯度联合预处理，提取稳定车道候选。
2. 动态 ROI（可由消失点自适应更新）限制搜索区域。
3. 概率霍夫提取线段，按几何约束分左右车道。
4. 霍夫线段做鲁棒拟合作为几何先验，估计消失点并动态调整 ROI。
5. 在 ROI 二值图中拟合左右车道二次曲线 $x(y)=ay^2+by+c$。
6. 相机去畸变后，使用相机模型将像素点投影到地平面坐标系。
7. 计算真实物理指标：
  - 车道宽度（像素 + 米）
  - 曲率半径（米）
  - 横向偏移（米）
8. 使用变道状态机(FSM)和置信度评分识别 prepare/changing/recovering 过程。
9. 在夜晚/雨天/逆光触发降级策略，复用时序状态保证输出连续。

## 运行方式

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行演示
```bash
python run_demo.py --show
```
默认会读取你现有工程中的样例视频并输出结果视频到 `outputs/demo_output.mp4`。

### 3. 先做相机标定（推荐）
```bash
python tools/calibrate_camera.py --images "data/calibration/*.jpg" --cols 9 --rows 6 --square-size 0.025
```
会输出：
- `outputs/camera_calibration.json`（标定参数）
- `outputs/calibration_quality_report.json`（误差分布与坏帧剔除）
- `outputs/calibration_quality.png`（自动可视化图）

### 4. 使用标定参数运行演示
```bash
python run_demo.py --show --calibration outputs/camera_calibration.json --camera-height 1.45 --pitch-deg 3.0
```

### 5. 批量指标评估
```bash
python tools/video_metrics.py --max-frames 500 --calibration outputs/camera_calibration.json
```
输出 JSON 统计报告到 `outputs/metrics.json`。

### 6. 合成数据鲁棒性测试
```bash
python tools/synthetic_benchmark.py
```

### 7. 单元测试
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 你在简历中可以这样写
- 设计并实现计算几何驱动的车道感知系统，支持消失点、车道宽度、曲率和偏移估计。
- 构建了完整相机标定与去畸变链路，自动输出误差分布报告并剔除坏帧。
- 基于二次曲线模型 + 时序状态机处理曲线车道与复杂变道场景，输出变道置信度。
- 引入夜晚/雨天/逆光失效检测与降级策略，提升复杂环境稳定性。
- 搭建了数据评估脚本与单元测试体系，实现模型稳定性与可复现性的量化验证。

## 推荐面试讲解文档
请结合 [docs/INTERVIEW_STORY.md](docs/INTERVIEW_STORY.md) 进行项目讲述。
