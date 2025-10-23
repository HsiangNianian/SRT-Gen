"""
1. 预处理， 归一化
2. 分帧， 计算均方根 EQ
3. 语音+静音判断
4. 片段的边界判断， 从开始静音那一刻计算阈值
5. 换气点细分
    条件1: 谷底能量 < 前后峰值平均 × (1 - 0.2) EQ 显著下降
    条件2: 低能量区域持续 ≥ 0.4秒 ， 持续时间够长
    条件3: 前后都有峰值，确保是"说话-换气-说话"模式
6. 切分换气点
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt


class AudioSegmentDetector:
    def __init__(
        self,
        volume_threshold: float = 0.7,  # 音量大小阈值（相对于最大振幅）
        min_silence_duration: float = 0.3,  # 静音时间长度阈值（秒）
        min_speech_duration: float = 0.2,  # 连续高音量声音持续时间阈值（秒）
        frame_duration: float = 0.02,  # 帧长度（秒），默认20ms
        enable_breath_detection: bool = True,  # 是否启用换气点检测
        energy_drop_threshold: float = 0.3,  # 能量下降阈值（相对于峰值）
        min_breath_duration: float = 0.15,  # 最小换气时长（秒）
    ):
        self.volume_threshold = volume_threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.frame_duration = frame_duration
        self.enable_breath_detection = enable_breath_detection
        self.energy_drop_threshold = energy_drop_threshold
        self.min_breath_duration = min_breath_duration

    def calculate_energy(self, audio: np.ndarray) -> float:
        """计算音频片段的能量（RMS）"""
        return np.sqrt(np.mean(audio**2))

    def refine_segments_by_breath(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[tuple[float, float]],
        frame_energies: np.ndarray,
    ) -> list[tuple[float, float]]:
        """
        TODO: 根据音量变化趋势（换气点）细分语音片段

            识别"递减-递增"的模式：
            1. 说话时能量递减（声音变小）
            2. 换气时能量降到低点
            3. 再次说话时能量递增（重新开始）
        """
        refined_segments = []
        min_breath_frames = int(self.min_breath_duration / self.frame_duration)

        for start_time, end_time in segments:
            start_frame = int(start_time / self.frame_duration)
            end_frame = int(end_time / self.frame_duration)

            # 提取该片段的能量序列
            segment_energies = frame_energies[start_frame:end_frame]

            if len(segment_energies) < min_breath_frames * 3:
                # 片段太短，不进行细分
                refined_segments.append((start_time, end_time))
                continue

            # 使用滑动窗口平滑能量曲线，减少噪声影响
            window_size = max(3, min_breath_frames // 2)
            smoothed_energies = np.convolve(
                segment_energies, np.ones(window_size) / window_size, mode="same"
            )

            # 找到局部最大值（能量峰值）和局部最小值（能量谷底）
            peaks = []
            valleys = []

            for i in range(1, len(smoothed_energies) - 1):
                # 局部最大值
                if (
                    smoothed_energies[i] > smoothed_energies[i - 1]
                    and smoothed_energies[i] > smoothed_energies[i + 1]
                ):
                    peaks.append(i)
                # 局部最小值
                elif (
                    smoothed_energies[i] < smoothed_energies[i - 1]
                    and smoothed_energies[i] < smoothed_energies[i + 1]
                ):
                    valleys.append(i)

            # 找到显著的能量下降（换气点）
            split_points = []
            max_energy = np.max(smoothed_energies)

            for valley_idx in valleys:
                valley_energy = smoothed_energies[valley_idx]

                # 找到该谷底前后的峰值
                prev_peak_energy = max_energy
                next_peak_energy = max_energy

                # 找前一个峰值
                for peak_idx in reversed(peaks):
                    if peak_idx < valley_idx:
                        prev_peak_energy = smoothed_energies[peak_idx]
                        break

                # 找后一个峰值
                for peak_idx in peaks:
                    if peak_idx > valley_idx:
                        next_peak_energy = smoothed_energies[peak_idx]
                        break

                # 判断是否为显著的能量下降（换气点）
                # 条件1: 谷底能量显著低于前后峰值
                # 条件2: 前后都有足够的能量回升（说明是换气后继续说话）
                avg_peak_energy = (prev_peak_energy + next_peak_energy) / 2

                if valley_energy < avg_peak_energy * (1 - self.energy_drop_threshold):
                    # 检查谷底持续时长（确保不是瞬间的能量波动）
                    valley_start = valley_idx
                    valley_end = valley_idx

                    # 向前找低能量区域的起点
                    while valley_start > 0 and smoothed_energies[
                        valley_start
                    ] < avg_peak_energy * (1 - self.energy_drop_threshold * 0.7):
                        valley_start -= 1

                    # 向后找低能量区域的终点
                    while valley_end < len(smoothed_energies) - 1 and smoothed_energies[
                        valley_end
                    ] < avg_peak_energy * (1 - self.energy_drop_threshold * 0.7):
                        valley_end += 1

                    # 检查低能量区域的持续时长
                    if valley_end - valley_start >= min_breath_frames:
                        # 在谷底的中点进行分割
                        split_frame = (valley_start + valley_end) // 2
                        split_points.append(split_frame)

            # 根据分割点切分片段
            if not split_points:
                # 没有找到换气点，保持原片段
                refined_segments.append((start_time, end_time))
            else:
                # 按分割点切分
                split_points = sorted(set(split_points))
                prev_split = 0

                for split_idx in split_points:
                    # 计算子片段的时间
                    sub_start_time = (start_frame + prev_split) * self.frame_duration
                    sub_end_time = (start_frame + split_idx) * self.frame_duration

                    # 检查子片段是否满足最小时长要求
                    if sub_end_time - sub_start_time >= self.min_speech_duration:
                        refined_segments.append((sub_start_time, sub_end_time))

                    prev_split = split_idx

                # 添加最后一个子片段
                sub_start_time = (start_frame + prev_split) * self.frame_duration
                sub_end_time = end_time
                if sub_end_time - sub_start_time >= self.min_speech_duration:
                    refined_segments.append((sub_start_time, sub_end_time))

        return refined_segments

    def detect_segments(
        self, audio_path: str | Path
    ) -> tuple[list[tuple[float, float]], int, np.ndarray]:
        """
        检测音频中的有效说话片段
        """
        audio, sample_rate = sf.read(audio_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        frame_size = int(self.frame_duration * sample_rate)
        num_frames = len(audio) // frame_size
        frame_energies = []

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame = audio[start_idx:end_idx]
            energy = self.calculate_energy(frame)
            frame_energies.append(energy)

        frame_energies = np.array(frame_energies)

        # 根据阈值判断每帧是否为语音
        is_speech = frame_energies > self.volume_threshold

        # 找到语音片段的起止点
        segments = []
        in_speech = False
        speech_start = 0
        silence_frames = 0
        min_silence_frames = int(self.min_silence_duration / self.frame_duration)
        min_speech_frames = int(self.min_speech_duration / self.frame_duration)

        for i, speech_flag in enumerate(is_speech):
            if speech_flag:  # 当前帧是语音
                if not in_speech:  # 从静音转为语音
                    speech_start = i
                    in_speech = True
                silence_frames = 0  # 重置静音计数
            else:  # 当前帧是静音
                if in_speech:  # 在语音段中
                    silence_frames += 1
                    # 如果静音帧数超过阈值，结束当前语音段
                    if silence_frames >= min_silence_frames:
                        speech_end = i - silence_frames
                        # 检查语音段长度是否满足最小要求
                        if speech_end - speech_start >= min_speech_frames:
                            # 转换为时间（秒）
                            start_time = speech_start * self.frame_duration
                            end_time = speech_end * self.frame_duration
                            segments.append((start_time, end_time))
                        in_speech = False
                        silence_frames = 0

        # 处理最后一个语音段
        if in_speech:
            speech_end = len(is_speech) - silence_frames
            if speech_end - speech_start >= min_speech_frames:
                start_time = speech_start * self.frame_duration
                end_time = speech_end * self.frame_duration
                segments.append((start_time, end_time))

        # 根据换气点细分片段
        if self.enable_breath_detection and segments:
            segments = self.refine_segments_by_breath(
                audio, sample_rate, segments, frame_energies
            )

        return segments, sample_rate, audio

    def visualize_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[tuple[float, float]],
        save_path: str | Path | None = None,
        show_energy: bool = True,
    ):
        """
        可视化音频波形和检测到的语音片段
        """
        if show_energy:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # 绘制音频波形
            time_axis = np.arange(len(audio)) / sample_rate
            ax1.plot(time_axis, audio, alpha=0.5, linewidth=0.5)
            ax1.grid(True, alpha=0.3)

            # 标记语音片段
            for i, (start, end) in enumerate(segments):
                ax1.axvspan(
                    start,
                    end,
                    alpha=0.3,
                    color="green",
                )
            ax1.legend()

            # 绘制能量曲线
            frame_size = int(self.frame_duration * sample_rate)
            num_frames = len(audio) // frame_size
            frame_energies = []
            frame_times = []

            for i in range(num_frames):
                start_idx = i * frame_size
                end_idx = start_idx + frame_size
                frame = audio[start_idx:end_idx]
                energy = self.calculate_energy(frame)
                frame_energies.append(energy)
                frame_times.append(i * self.frame_duration)

            ax2.plot(
                frame_times,
                frame_energies,
                linewidth=1.5,
                color="blue",
            )
            ax2.axhline(
                y=self.volume_threshold,
                color="r",
                linestyle="--",
            )

            # 标记语音片段
            for i, (start, end) in enumerate(segments):
                ax2.axvspan(start, end, alpha=0.2, color="green")
                
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
            time_axis = np.arange(len(audio)) / sample_rate
            ax1.plot(time_axis, audio, alpha=0.5, linewidth=0.5)
            ax1.grid(True, alpha=0.3)

            # 标记语音片段
            for i, (start, end) in enumerate(segments):
                ax1.axvspan(
                    start,
                    end,
                    alpha=0.3,
                    color="green",
                )
            ax1.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def save_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[tuple[float, float]],
        output_dir: str | Path,
        prefix: str = "segment",
    ):
        """
        将检测到的语音片段保存为单独的音频文件
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            output_path = output_dir / f"{prefix}_{i + 1}_{start:.2f}s_{end:.2f}s.wav"
            sf.write(output_path, segment_audio, sample_rate)
            saved_files.append(output_path)

        return saved_files


def main():
    audio_file = Path(__file__).resolve().parents[1] / "audio" / "23_vocals.wav"
    detector = AudioSegmentDetector(
        volume_threshold=0.02,  # 音量阈值：2%的最大振幅
        min_silence_duration=0.2,  # 静音时长阈值：0.2秒
        min_speech_duration=0.05,  # 最小说话时长：0.3秒
        frame_duration=0.02,  # 帧长：20毫秒
        enable_breath_detection=True,
        energy_drop_threshold=0.2,  # 能量下降30%认为是换气
        min_breath_duration=0.4,  # 最小换气时长：0.4秒
    )
    
    segments, sample_rate, audio = detector.detect_segments(audio_file)
    total_duration = 0

    for i, (start, end) in enumerate(segments, 1):
        duration = end - start
        total_duration += duration
        print(f"片段 {i}: {start:6.2f}s - {end:6.2f}s (时长: {duration:5.2f}s)")

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "audio_segments"
    detector.save_segments(audio, sample_rate, segments, output_dir, prefix="speech")
    viz_path = output_dir / "segments_visualization.png"
    detector.visualize_segments(
        audio, sample_rate, segments, save_path=viz_path, show_energy=True
    )


if __name__ == "__main__":
    main()
