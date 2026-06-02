# 百度 生成式推荐广告排序推理性能优化

## 赛道概要背景
传统广告排序模型已难以满足个性化推荐需求，生成式广告排序模型凭借强大的序列建模与语义理解能力成为行业趋势。该类模型依托 Transformer 架构，能深度挖掘用户点击、转化等超长行为序列中的长距离依赖关系，精准捕捉用户兴趣演化规律，从而生成更具吸引力的个性化广告内容，提升广告点击率与用户体验。但在实际应用中，存在很多挑战，如模型参数规模大、注意力计算复杂、存在超长历史序列、量化影响推理精度等。

## 任务说明
本次任务提供百度商业真实的用户行为数据、广告信息，选手需要在保证模型推理效果的前提下，极致优化推理性能。

## 数据集介绍
- 用户行为数据：包括全局唯一的日志ID和用户ID、广告曝光时间、广告点击时间等信息；
- 广告内容：包括广告的文本描述、图片信息、广告主信息等；
- 上下文信息：包括用户的地理位置、职业、性别、设备类型等；
- 用户统计信息：包括用户的活跃度、兴趣标签、历史点击率等统计数据。

## Baseline 运行结果
- 推理时间：229.1826s
- AUC：0.759232
- PCOC：1.110063
- score_latency：0.236058
- score_model：0.310817
- score_all：25.848547

## 评估指标
### 推理效率评估
参赛者提交inference脚本后，会通过统计inference脚本的运行时间，来计算在测试集上单条样本的平均推理时间。推理效率打分采用特定公式，如平均推理时间超过定义的时间限制，则本项和最终得分为0。

### 策略效果评估
综合考虑AUC及PCOC指标，PCOC需满足[0.85, 1.15]，AUC需满足[0.65, 1]，方可进入榜单排序，否则本项和最终得分为0。得分由pcoc和auc组合而成。

### 指标说明
- AUC：ROC曲线下的面积，越接近与1越好
- PCOC：预估转化率 / 真实转化率，越接近于1越好

### 计分规则
综合考虑推理性能和策略效果两个指标。

### 警告⚠️
- 推理效率和策略效果任何一项得分为0，整体得分为0。
- 评估容器有整体运行时间限制（纯推理最长5分钟），如果超出则无法计入成绩；（build_env.sh等要在20分钟内）
- 任何作弊行为将会取消队伍成绩。

## 提交要求
参赛选手需要提交一个命名为【xxx】.zip的压缩包，压缩包内需要包含以下内容:
- 程序入口infer.py脚本，以及环境构建脚本build_env.sh、requirements.txt。
- 额外的python包环境，选手可以通过将python环境打包放在当前工作目录
- 优化过的模型文件，如量化后的模型等

### PS
- 打包不要包含 eval 文件夹 和 dataset 文件夹
- 权重若使用原版，无需修改权重参数且无需上传权重
- 若需要使用自定义权重，请自行完善和修改 infer.py相关逻辑，系统测评后台默认调用赛事官方权重无需上传
- 若需要进行编译等其他复杂操作，请在 build_env.sh 中完成

## 下载资源

### 方式1：使用 aistudio-sdk（推荐）

```bash
# 安装 aistudio-sdk
pip install --upgrade aistudio-sdk

# 下载数据集
aistudio download --dataset gump/2026_cti_data --local_dir ./dataset

# 下载模型权重
aistudio download --model gump/2026_cti_model --local_dir ./weights
```

### 方式2：使用 wget 下载预编译依赖包

```bash
# 下载预编译依赖包（5.3GB）
wget https://studio-package.bj.bcebos.com/2026-shangye-python-package/external-libraries.tar -O libraries.tar

# 解压到 libraries 目录
tar -xf libraries.tar --strip-components=1 -C libraries
```

### 方式3：在 AI Studio 平台使用

在 AI Studio 项目中，数据集和模型已绑定，可直接使用：

```bash
# 链接数据集
ln -s /home/aistudio/data/datasets/375013/2026_cti_data/dataset ./dataset

# 合并模型权重文件
cat /home/aistudio/data/models/45703/2026_cti_model/ckpt.part.0* > ./ckpt.pt
```

## 参考资料
- GRAB-百度推荐广告生成式排序模型技术实践: https://arxiv.org/pdf/2602.01865
- HSTU：Meta提出的用于长序列行为建模的高效模型: https://arxiv.org/abs/2402.17152
- 模型权重: https://aistudio.baidu.com/modelsdetail/45703/space
- 数据集: https://aistudio.baidu.com/dataset/detail/375013/file
- Baseline: https://aistudio.baidu.com/projectdetail/10186630

## 项目结构
```
Baidu GRAB/
├── dataset/          # 存放数据集（提交时不包含）
├── weights/          # 存放模型权重（提交时不包含）
├── docs/            # 项目文档
│   └── project_info.md
├── infer.py         # 推理脚本入口
├── build_env.sh     # 环境构建脚本
├── requirements.txt # 依赖列表
└── download_data.sh # 数据下载脚本
```
