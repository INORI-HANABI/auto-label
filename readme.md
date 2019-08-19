### 针对VOC数据的自动标注流程

1.准备数据。

针对视频数据，如果是DVR录制的.h264格式，则先用h264toavi.bat脚本在windows环境下将视频格式进行转换，若是其他格式也同样处理，转换为opencv支持的格式。

针对图片格式，opencv支持即可。

2.自动标注。

如果为视频数据，则运行auto-label-video.py脚本。

如果为图片数据，则运行auto-label-image.py脚本。

3.筛选与整理

人工进行筛选需要的部分，整理时运行label-to-img.py脚本，保证标注与图片的统一。

