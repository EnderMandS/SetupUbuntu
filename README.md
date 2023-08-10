# 配置Ubuntu环境

这是一个配置[Ubuntu](https://cn.ubuntu.com/download)环境的文档，包括[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)、[OpenCV](https://opencv.org/)、[Tensorflow2](https://tensorflow.google.cn/?hl=zh-cn)、[PyTorch2](https://pytorch.org/)、[ROS](https://www.ros.org/)、[Docker](https://www.docker.com/)、SLAM基本环境等。文档适用于新安装的Ubuntu系统。文章制作于2023年8月。

------

## 前提

多数软件包之间存在依赖关系，请务必按照文档顺序进行配置。软件存放在`~/pkg`文件夹下

### Ubuntu版本选择

CUDA最新版本已经停止对[Ubuntu18.04](https://releases.ubuntu.com/18.04.6/)的更新，故不推荐使用18版本。如果需要使用ROS1，建议选择[Ubuntu20.04](https://releases.ubuntu.com/20.04.6/)，如果不需要使用ROS1选择[Ubuntu22.04](https://cn.ubuntu.com/download/desktop)版本。本文档安装的软件不存在Ubuntu版本匹配问题。系统最少分配80GB，建议150GB。

### NVIDIA显卡驱动版本选择

选择530+版本的驱动，建议使用最新版本的[驱动](https://www.nvidia.cn/drivers/unix/)。

### CUDA版本选择

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)版本的选择取决于Tensorflow和PyTorch的匹配。优先考虑PyTorch匹配，[Tensorflow的Docker镜像](https://hub.docker.com/r/tensorflow/tensorflow/)可以解决匹配问题，yolov8虽然也有Docker镜像但是体积较大。PyTorch可以选择*Preview (Nightly)*标签选择最新CUDA匹配版本。点击查看[Tensorflow-CUDA对应关系](https://tensorflow.google.cn/install/source?hl=zh-cn#gpu)、[PyTorch-CUDA对应关系](https://pytorch.org/get-started/locally/)。

### Python版本选择

Python版本小于3.8的必须更新。版本的选择取决于[Tensorflow匹配版本](https://tensorflow.google.cn/install/source?hl=zh-cn#gpu)，如果在Docker中使用Tensorflow不需要考虑匹配问题，在[Python官网](https://www.python.org/downloads/)中建议选择*Maintenance status*标签为*security*的Python最新版本。

### 其它软件版本

文档中的`wget`等下载的是创建文档时的最新版本。建议使用者访问软件官网来获取最新版本。

### 换源

不建议换源，任何时候都不要`apt upgrade`。软件包下载需要访问海外资源。[八仙过海](https://www.q1cloud.me/)，[各显神通](https://github.com/shadowsocksrr/electron-ssr)。

### 本机环境及版本信息

时间：2023年8月，Ubuntu:18.04.6，NVIDIA驱动:530.41.03，CUDA:12.1.1，cuDNN:8.9.2，TensorRT:8.6.1，Python更新:3.10.12，CMake:3.27.0，Ninja:1.11.1，Docker:24.0.2，Tensorflow:2.6.0，PyTorch:Nightly CUDA 12.1，Clang-LLVM:16.0.0，LAPCAK:3.11.0，GMP:6.2.1，MPFR:4.2.0，SuiteSparse:7.1.0，fmt:10.0.0，Eigen:3.4.0，Sophus:61f9a98，Ceres:2.1.0，g2o:672aa7a，OpenCV:4.X，Pangolin:d484494，VTK:9.2.6，metslib:0.5.3，PCL:1.13.1，glog:0.6.0，gtest:1.13.0

------

## START

### 安装依赖

```shell
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install git vim wget openssh-server net-tools tree pkg-config curl dkms rename -y
sudo apt install build-essential cmake g++ gcc unzip python3-pip -y
sudo apt install ninja-build clang clang-format clang-tidy libboost-all-dev libssl-dev -y
sudo apt install libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev libwayland-dev libxkbcommon-dev wayland-protocols libeigen3-dev -y
sudo apt install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev libcanberra-gtk-module -y
sudo apt install libgtk2.0-dev libjasper-dev libtbb-dev zlib1g -y
sudo apt install libatlas-base-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev libmetis-dev -y
sudo apt install libspdlog-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 -y
sudo apt install meshlab libpcl-dev pcl-tools libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-utils -y
sudo apt install liboctomap-dev octovis libcoarrays-dev libopenblas-dev -y
mkdir ~/pkg
mkdir ~/dataset
```

### 安装软件包

```shell
sudo apt install openjdk-17-jdk -y
sudo apt install terminator htop -y
```

### GIT 设置

```shell
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

### 配置中文拼音输入法

```shell
sudo apt install ibus ibus-gtk ibus-gtk3 ibus-pinyin -y
```

安装完成后重启

### 配置SSH

开启SSH服务

```shell
sudo systemctl status ssh
sudo ufw allow ssh
```

生成密钥并赋予权限。其中`id_rsa.pub`为公钥，存放在服务器端。另一个为私钥

```shell
ssh-keygen
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
sudo chmod 600 ~/.ssh/authorized_keys
sudo chmod 700 ~/.ssh
```

编辑配置文档

```shell
sudo gedit etc/ssh/sshd_config
```

在文件末尾添加以下行

```shell
RSAAuthentication yes
PubkeyAuthentication yes
PasswordAuthentication no
```

保存退出，重启SSH服务

```shell
sudo service sshd restart
```

(可选) 在VSCode中可以配置SSH文件

```shell
Host your-host-name
	HostName remote_ip_address
	User user_name
	Port 22
```

### [Oh My Zsh](https://ohmyz.sh/)

Oh My Zsh是一个zsh终端，美化终端提供输入历史记录和高亮等，还有许多终端主题可以更换

```shell
sudo apt update
sudo apt install zsh -y
chsh -s $(which zsh)
```

登出再登入

```shell
sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

编译`~/.zshrc`文件，搜索*plugin*，添加`zsh-autosuggestions zsh-syntax-highlighting`到`{}`内

```shell
gedit ~/.zshrc
```

### [CMake](https://cmake.org/)

```shell
cd ~/pkg
mkdir cmake && cd cmake
wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc4/cmake-3.27.0-rc4-linux-x86_64.sh
sudo sh cmake-3.27.0-rc4-linux-x86_64.sh --prefix=/usr
```

询问License输入`y`，询问安装目录输入`n`，安装完成后输入`cmake --version`查询版本。安装成功后删除安装脚本

```shell
rm cmake-3.27.0-rc4-linux-x86_64.sh
```

### [Ninja](https://ninja-build.org/)

```shell
cd ~/pkg
mkdir ninja && cd ninja
wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
unzip ninja-linux.zip && rm ninja-linux.zip
sudo cp ninja /usr/bin
```

### Python

安装其他版本请自行更换版本关键字

```shell
cd ~/pkg
mkdir python && cd python
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz
tar xvf Python-3.10.12.tar.xz && rm Python-3.10.12.tar.xz
cd Python-3.10.12
```

选择安装路径为`/usr/local/python310`，也可以不指定安装路径

```shell
./configure --enable-optimizations --prefix=/usr/local/python310
make -j
make test
```

在`make test`中会有网络相关的测试项报错，无需理会。若有其他报错项，请查找相关文档，重新下载软件包并编译

```shell
sudo make install
make clean
sudo rm /usr/bin/python3
sudo ln -s /usr/local/python310/bin/python3.10 /usr/bin/python3
sudo ln -s /usr/local/python310/bin/pip3.10 /usr/bin/pip310
/usr/local/python310/bin/python3.10 -m pip install --upgrade pip
```

安装一些软件包

```shell
pip install launchpadlib
pip install numpy
pip install opencv-python
pip install pillow
pip install matplotlib
pip install g2o-python
pip install notebook
pip install scipy
```

### [NVIDIA显卡驱动](https://www.nvidia.cn/drivers/unix/)

```shell
cd ~/pkg
mkdir nvidia && cd nvidia
mkdir driver && cd driver
wget https://cn.download.nvidia.com/XFree86/Linux-x86_64/470.199.02/NVIDIA-Linux-x86_64-530.41.03.run
sudo sh NVIDIA-Linux-x86_64-530.41.03.run
rm NVIDIA-Linux-x86_64-530.41.03.run
```

驱动程序的安装可能需要无GUI环境，请在开机启动GNU GRUB界面配置无GUI启动

### [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)

```shell
cd ~/pkg/nvidia
mkdir cuda && cd cuda
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

由于显卡驱动已经安装，安装CUDA时不要选择安装驱动。安装结束后会提示安装未完全结束，是因为驱动已经在之前另外安装，无需理会。

```shell
gedit ~/.zshrc
```

编辑`~/.zshrc`文件，添加以下行，配置CUDA路径

```shell
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)

```shell
cd ~/pkg/nvidia
mkdir cudnn && cd cudnn
```

去官网下载对以CUDA版本的cuDNN到`~/pkg/nvidia/cudnn`路径下

```shell
tar -xvf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
rm cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### [TensorRT](https://developer.nvidia.com/zh-cn/tensorrt)

请在官网自行下载对应版本CUDA和系统版本的安装包

```shell
cd ~/pkg/nvidia
mkdir tensorRT && cd tensorRT
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu1804-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu1804-8.6.1-cuda-12.0_1.0-1_amd64.deb
rm nv-tensorrt-local-repo-ubuntu1804-8.6.1-cuda-12.0_1.0-1_amd64.deb*
```

### [Docker](https://www.docker.com/)

一种容器技术，可以理解为一种多平台通用、几乎没有性能损失的“虚拟机”

```shell
sudo apt update
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

### NVIDIA Container Toolkit

可以让Docker调用GPU

```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit-base
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

执行以下指令，如果有输出，删除所指示文件`sudo rm`

```shell
grep -l "nvidia.github.io" /etc/apt/sources.list.d/* | grep -vE "/nvidia-container-toolkit.list\$"
```

删除后，`apt update`应不会报错

```shell
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker $USER
newgrp docker
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu18.04 nvidia-smi
```

`nvidia/cuda:12.1.0-base-ubuntu18.04`词条应更换为使用者所对应环境，词条可以在[Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags)中搜索。可以不执行最后一行指令，其只是为了验证安装。

### Tensorflow2

使用tensorflow官方提供的镜像

```shell
docker pull tensorflow/tensorflow:latest-gpu
docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

或 使用我从官方镜像中添加了zsh的构建

```shell
docker run --rm -p 10022:22 -p 8888:8888 -itd --runtime=nvidia --gpus all endermands/tensorflow-gpu-zsh:latest nvidia-smi
```

由于网络原因在Docker构建镜像时下载慢，以后使用容器时安装以下包

```shell
pip install notebook
pip install scipy
pip install pillow
pip install opencv-python
```

### [PyTorch](https://pytorch.org/get-started/locally/)

请自行选择版本更换词条

```shell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

验证安装

```shell
python3 -c "import torch;print(torch.cuda.is_available())"
```

### [yolo](https://github.com/ultralytics/ultralytics)

```shell
docker pull ultralytics/ultralytics:latest
```

### [Clang-llvm](https://clang.llvm.org/)

```shell
cd ~/pkg
mkdir llvm && cd llvm
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar xvf clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
rm clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
cd clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04
sudo cp -r bin/ /usr
sudo cp -r include/ /usr
sudo cp -r lib/ /usr
sudo cp -r libexec/ /usr
sudo cp -r share/ /usr
```

### (可选) LAPACK

```shell
cd ~/pkg
mkdir lapack && cd lapack
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.0.tar.gz
tar xvf v3.11.0.tar.gz && rm v3.11.0.tar.gz
cd lapack-3.11.0
mkdir -p build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
```

### (可选) GMP

```shell
cd ~/pkg
mkdir GMP && cd GMP
wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
tar xvf gmp-6.2.1.tar.xz && rm gmp-6.2.1.tar.xz
cd gmp-6.2.1
./configure --enable-cxx
make -j
make check
sudo make install
make clean
```

### (可选) MPFR

```shell
mkdir MPFR && cd MPFR
wget https://www.mpfr.org/mpfr-current/mpfr-4.2.0.tar.xz
tar xvf mpfr-4.2.0.tar.xz && rm mpfr-4.2.0.tar.xz
cd mpfr-4.2.0
./configure
make -j
make check
sudo make install
make clean
```

### (可选) SuitSparse

```shell
cd ~/pkg
mkdir SuiteSparse && cd SuiteSparse
wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.1.0.tar.gz
tar xvf v7.1.0.tar.gz
rm v7.1.0.tar.gz
cd SuiteSparse-7.1.0
make -j
sudo make install
make clean
```

### [fmt](https://github.com/fmtlib/fmt)

```shell
cd ~/pkg
mkdir fmt && cd fmt
wget https://github.com/fmtlib/fmt/releases/download/10.0.0/fmt-10.0.0.zip
unzip fmt-10.0.0.zip
rm fmt-10.0.0.zip
cd fmt-10.0.0
mkdir -p build && cd build
cmake -GNinja -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
ninja
sudo ninja install
ninja clean
```

### Eigen3

```shell
cd ~/pkg
mkdir eigen && cd eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvf eigen-3.4.0.tar.gz
rm eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir -p build && cd build
cmake -GNinja ..
sudo ninja install
ninja clean
```

### Sophus

```shell
cd ~/pkg
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir -p build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
```

### Ceres

```shell
cd ~/pkg
mkdir ceres && cd ceres
wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.zip
unzip ceres-solver-2.1.0.zip
rm ceres-solver-2.1.0.zip
cd ceres-solver-2.1.0
mkdir -p build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
```

### g2o

```shell
cd ~/pkg
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
mkdir -p build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
```

如果编译报错添加c++17标准

```shell
gedit ~/pkg/g2o/CMakeLists.txt
```

添加`add_definitions(-std=c++17)`

### [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

```shell
mkdir -p ~/pkg/opencv && cd ~/pkg/opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
rm -rf opencv.zip opencv_contrib.zip
mkdir -p build && cd build
cmake -GNinja -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
ninja
sudo ninja install
ninja clean
cd ~
```

### Pangolin

```shell
cd ~/pkg
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
./scripts/install_prerequisites.sh --dry-run recommended
./scripts/install_prerequisites.sh -m apt all
mkdir -p build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
cd
```

### (可选) FBoW

```shell
cd ~/pkg
git clone https://github.com/rmsalinas/fbow.git
cd fbow
mkdir build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
cd
```

### VTK

```shell
cd ~/pkg
mkdir vtk && cd vtk
wget https://www.vtk.org/files/release/9.2/VTK-9.2.6.tar.gz
tar xvf VTK-9.2.6.tar.gz
rm VTK-9.2.6.tar.gz
cd VTK-9.2.6
mkdir build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
cd
```

### metslib

```shell
cd ~/pkg
mkdir metslib && cd metslib
wget https://github.com/coin-or/metslib/archive/refs/tags/releases/0.5.3.tar.gz
tar xvf 0.5.3.tar.gz
rm 0.5.3.tar.gz
cd metslib-releases-0.5.3
./configure
make -j`nproc`
sudo make install
make clean
cd
```

### PCL

```shell
cd ~/pkg
mkdir pcl && cd pcl
wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.13.1/source.tar.gz
tar xvf source.tar.gz
rm source.tar.gz
cd pcl
mkdir build && cd build
sudo ln -s /usr/bin/vtk6 /usr/bin/vtk
sudo ln -s /usr/lib/python2.7/dist-packages/vtk/libvtkRenderingPythonTkWidgets.x86_64-linux-gnu.so /usr/lib/x86_64-linux-gnu/libvtkRenderingPythonTkWidgets.so
gedit ../cmake/pcl_find_vtk.cmake
```

编辑文件，在第31行添加VTK版本指定`find_package(VTK 9)`

```shell
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja
sudo ninja install
ninja clean
cd ~
```

### glog 和 gtest

```shell
cd ~/pkg
mkdir -p google && cd google
wget https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz
wget https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
tar xvf v0.6.0.tar.gz && tar xvf v1.13.0.tar.gz
rm v0.6.0.tar.gz && rm v1.13.0.tar.gz
cd glog-0.6.0/
mkdir build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
cd ../../googletest-1.13.0
mkdir build && cd build
cmake -GNinja ..
ninja
sudo ninja install
ninja clean
```

### [ORB SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

```shell
cd ~/pkg
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
cd ORB_SLAM3 && chmod +x build.sh
gedit CMakeLists.txt
```

添加C++14编译标准`add add_compile_options(-std=c++14)`

```shell
./build.sh
```

### ROS

使用脚本安装ROS，自行选择ROS版本，不要换源

```shell
wget http://fishros.com/install -O fishros && . fishros
echo "source /opt/ros/melodic/setup.zsh" >> ~/.zshrc
```

### [VSCode](https://code.visualstudio.com/)

安装以下扩展`cmake`, `clangd`, `ros`, `codelldb`, `docker`, `jupyter`, `black formatter`

使用Clangd作为代码提示更快更准确，所以使用ROS扩展时禁用C/C++扩展，在工程下`tasks.json`文件中，添加以下行

```json
-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

重新编译ROS工作区生效
