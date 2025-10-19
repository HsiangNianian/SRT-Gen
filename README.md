# SRT-Gen
> 一个基于 funASR 的干声音频字幕文件生成器, 由多个小模型耦合而成, 本地即可推理  
> A funASR-based dry audio subtitle generator, built by coupling multiple small models and capable of local inference

# SRT 生成思路(SRT Generation Strategy)
1. 一般而言, 拿到一段原始音频或者视频, 需要使用 ffmpeg 提取音频, 并使用 mst 或者其他工具/模型提取人的干声作为后续处理的音频文件
   > Generally speaking, after obtaining a raw audio or video clip, use ffmpeg to extract the audio, and then use mst or other tools/models to extract the dry voice of the person as an audio file for subsequent processing

3. 在拿到干声以后, 需要先进行语音活动检测, 即 VAD, 获取到精确的有效语音时间片段
   > After obtaining the dry voice, perform voice activity detection (VAD) to obtain accurate and valid speech time segments.

4. 将片段送入 ASR 语音识别模型获取连续文本
   > Feed the segments into the ASR speech recognition model to obtain continuous text.

6. 将连续文本送入 ct-punc 模型进行标点恢复
   > Feed the continuous text into the ct-punc model for punctuation recovery.

5. 对于较长的长音频片段, 需要额外进行基于 ct-punc 的标点句逗切分以及 fa-zh 对齐操作, 来获得更小的音频片段, 这样的小音频片段才是有效的
   > For longer audio segments, perform additional ct-punc-based punctuation and sentence segmentation and fa-zh alignment to obtain smaller audio segments. These smaller audio segments are
   
7. 最后根据每个片段的音频时间戳与音频识别文本生成 `.srt` 文件
   > Finally, generate a .srt file based on the audio timestamps and audio recognition text of each segment.

# 代码架构 - Architecture
(下次写...to be...
