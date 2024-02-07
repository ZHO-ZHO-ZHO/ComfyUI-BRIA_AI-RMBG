
![BR_](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG/assets/140084057/c8167676-c347-468a-a719-aee3c4c69310)


# ComfyUI-BRIA_AI-RMBG

Unofficial [BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4) of BRIA RMBG Model for ComfyUI

![Dingtalk_20240207145631](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG/assets/140084057/f22fcbc4-f223-44be-bbcd-93e2e55937de)

## 项目介绍 | Info

- 对[BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4)的非官方实现

- BRIA Background Removal v1.4：由 BRIA AI 开发，可作为非商业用途的开源模型

- 版本：**V1.5** 支持批量处理（可去除视频背景）、新增输出 mask 功能

## 视频演示 

SVD1.1 + RMBG 1.4 = 

https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG/assets/140084057/fdae7655-bcd0-4250-9d9b-a88b59f80d43



## 安装 | Install

- 推荐使用管理器 ComfyUI Manager 安装（On the Way）

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG.git`
    3. 重启 ComfyUI


## 使用说明 | How to Use

- 将 [Removal v1.4](https://huggingface.co/briaai/RMBG-1.4) 模型下载至`/custom_nodes/ComfyUI-BRIA_AI-RMBG/RMBG-1.4`

- 节点：

  ![Dingtalk_20240207154339](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG/assets/140084057/70b9089e-81e8-4853-8564-83145f881175)

   - 🧹BRIA_RMBG Model Loader：自动加载 Removal v1.4 模型
   - 🧹BRIA RMBG：去除背景


## 更新日志

- 20240207

  V1.5 支持批量处理、新增输出 mask 功能

  创建项目 V1.0 


## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG&Date)


## Credits

[BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4)

代码参考了 [@camenduru](https://twitter.com/camenduru) 感谢！
