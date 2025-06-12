基于VR遥操与模仿学习的机器人抓取系统

功能：

1、使用Genesis自动采集lerobot机械臂抓取物体的数据（数据格式与lerobot标准相同）


https://github.com/user-attachments/assets/ce12e647-d2ff-4716-90dd-3c3a77c1f644



2、使用VR遥操作控制lerobot机械臂



https://github.com/user-attachments/assets/559b8d76-2ab8-46dc-8e25-c2fc924fb6cf



3、借助lerobot官方代码实现数据训练

4、可以在Genesis仿真环境中进行推理评估

环境配置：

1、克隆项目

```
git clone https://github.com/nonsugarcokecola/lerobot_VR.git
cd lerobot_VR
```

可以选择更新里面的lerobot与lerobot-kinematics

```
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/box2ai-robotics/lerobot-kinematics.git
```

2、配置环境

```
#创建anaconda虚拟环境
conda create -y -n lerobot_genesis python=3.10
conda activate lerobot_genesis
conda install ffmpeg=7.1.1 -c conda-forge
#下载pyotrch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
#安装lerobot
cd lerobot && pip install -e ".[feetech]"
#安装lerobot-kinematics
cd lerobot-kinematics && pip install -e .
#安装Genesis
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
#假设anaconda环境在/home/{user}/anaconda3/envs/lerobot_genesis/lib/python3.10/site-packages，先删除其中的genesis_world-0.2.1.dist-info和genesis
rm -rf /home/{user}/anaconda3/envs/lerobot_genesis/lib/python3.10/site-packages/genesis_world-0.2.1.dist-info
rm -rf /home/{user}/anaconda3/envs/lerobot_genesis/lib/python3.10/site-packages/genesis
#将代码仓库中的genesis_world-0.2.1.dist-info和genesis来两个文件复制到环境中
cp -r ./genesis_world-0.2.1.dist-info /home/{user}/anaconda3/envs/lerobot_genesis/lib/python3.10/site-packages/
cp -r ./genesis /home/{user}/anaconda3/envs/lerobot_genesis/lib/python3.10/site-packages/

```

使用：

1、Genesis数据采集

在Genesis仿真环境中进行数据采集，--start控制采集初始位置索引，--last控制采集最后位置索引，--save_dir选择数据保存位置

```
python example/lerobot_sim_collect.py --start 0 --last 1 --save_dir='./data'
```

2、VR遥操作仿真环境中的lerobot机械臂(手柄数据获取是解耦的，VR也可以操控其他机械臂)

在Mujoco仿真环境中进行遥操作，基于WebXR实现，调试使用的是quest2，quest3应该也可以，首先开启后端接受VR手柄的数据

```
python example/serve_https.py
```

将VR与PC置于同一局域网下（可以用手机热点或者wifi），VR设备在网页中访问https://{电脑IP}:8000/webxr_quest_input.html"，即可进入操作界面，操作界面显示如下：



https://github.com/user-attachments/assets/abbd6e25-a0fa-4248-9e9e-b125903100d4



手柄数据会更新到example/controller_data.json中，包括手柄位置，姿态，按键信息等，通过下面命令遥操作lerobot机械臂

```
python example/lerobot_vr.py
```

3、数据训练

生成的数据缺少部分文件，需要进行编写，具体编写和训练过程可查看[doc/Train.md](./doc/Train.md)

4、推理评估

运行下面命令可将训练好的模型在仿真环境中进行推理，我通过50组仿真数据训练的模型权重以及训练数据在[这里](https://pan.baidu.com/s/1NJVjD33-rWkM-ubhmUyi-w?pwd=istr)可以看到

```
python example/lerobot_sim_infer.py
```

由于真实的机械臂和仿真环境中的机械臂会存在差异，sim2real时需要为六个关节加一定偏移量，可以参考lerobot_sim_collect_real.py的内容

本代码参考并使用了[lerobot](https://github.com/huggingface/lerobot)以及[lerobot-kinematics](https://github.com/box2ai-robotics/lerobot-kinematics)，感谢代码的作者
