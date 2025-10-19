from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import soundfile as sf
from funasr import AutoModel
from funasr_onnx import CT_Transformer_VadRealtime, Fsmn_vad, Paraformer

TimestampMs = int
AudioSegment = np.ndarray
VadSegment = Tuple[TimestampMs, TimestampMs]
WordTimestamp = Tuple[str, List[TimestampMs]]
SentenceTimestamp = Tuple[str, List[TimestampMs]]


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    text_length_threshold_ms: int = 3000
    invalid_texts: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.invalid_texts is None:
            self.invalid_texts = ["啊", "哎"]


@dataclass
class ModelConfig:
    asr_model_path: str = (
        "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    vad_model_path: str = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model_path: str = (
        "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    )
    fazh_model_name: str = "fa-zh"


@dataclass
class ProcessingResult:
    segments: List[SentenceTimestamp]
    vad_segments: List[VadSegment]
    metadata: Dict[str, Any]


class AudioProcessor:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def ms_to_samples(self, ms: TimestampMs) -> int:
        return int(ms * self.sample_rate / 1000)

    def samples_to_ms(self, samples: int) -> TimestampMs:
        return int(samples * 1000 / self.sample_rate)

    def extract_segment(
        self, audio_data: AudioSegment, start_ms: TimestampMs, end_ms: TimestampMs
    ) -> AudioSegment:
        start_sample = self.ms_to_samples(start_ms)
        end_sample = self.ms_to_samples(end_ms)
        return audio_data[start_sample:end_sample]

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[AudioSegment, int]:
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        return sf.read(str(audio_path))

    @staticmethod
    def save_audio(
        audio_data: AudioSegment, output_path: Path, sample_rate: int
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, sample_rate)


class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.asr_model: Optional[Paraformer] = None
        self.vad_model: Optional[Fsmn_vad] = None
        self.punc_model: Optional[CT_Transformer_VadRealtime] = None
        self.fazh_model: Optional[AutoModel] = None
        self._loaded = False

    def load_models(self) -> None:
        if self._loaded:
            return

        self.asr_model = Paraformer(self.config.asr_model_path)
        self.vad_model = Fsmn_vad(self.config.vad_model_path)
        self.punc_model = CT_Transformer_VadRealtime(self.config.punc_model_path)
        self.fazh_model = AutoModel(model=self.config.fazh_model_name)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_models()


class SRTGenerator:
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        model_config: Optional[ModelConfig] = None,
        sentence_delimiters: Optional[List[str]] = None,
    ):
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig()
        self.model_manager = ModelManager(self.model_config)
        self.audio_processor = AudioProcessor(self.audio_config.sample_rate)
        self.sentence_delimiters = (
            sentence_delimiters
            if sentence_delimiters is not None
            else ["。", "！", "？", "；", "，", "、", "：", """, """, "'", "'"]
        )

    def load_models(self) -> None:
        self.model_manager.load_models()

    def get_vad_segments(self, audio_data: AudioSegment) -> List[VadSegment]:
        self.model_manager.ensure_loaded()
        vad_result = self.model_manager.vad_model(audio_in=audio_data)  # type: ignore
        return vad_result[0]

    def get_asr_text(self, audio_segment: AudioSegment) -> str:
        self.model_manager.ensure_loaded()
        result = self.model_manager.asr_model(audio_segment)  # type: ignore
        return result[0]["preds"][0]

    def get_punc_text(
        self, text: str, param_dict: Optional[Dict[str, Any]] = None
    ) -> str:
        self.model_manager.ensure_loaded()
        if param_dict is None:
            param_dict = {"cache": []}
        result = self.model_manager.punc_model(text, param_dict=param_dict)  # type: ignore
        return result[0]

    def get_fazh_result(
        self, audio_path: Path, text_path: Path
    ) -> List[Dict[str, Any]]:
        self.model_manager.ensure_loaded()
        result = self.model_manager.fazh_model.generate(
            input=(str(audio_path), str(text_path)), data_type=("sound", "text")
        )
        return result

    def process_vad_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
    ) -> str:
        """处理单个VAD片段，返回带标点的文本"""
        start_ms, end_ms = vad_segment
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        asr_text = self.get_asr_text(segment_audio)
        punc_text = self.get_punc_text(asr_text)
        return punc_text

    def fix_word_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        vad_segment: VadSegment,
    ) -> List[WordTimestamp]:
        """修正FA-ZH返回的单词时间戳

        TODO: 直接取平均值不能覆盖所有情况，需要更复杂的算法，考虑标点符号的影响和VAD时间戳的准确性
        """
        if not word_timestamps:
            return []

        vad_start, vad_end = vad_segment

        # 计算首尾时间差异的平均调整量
        first_word_start_diff = word_timestamps[0][1][0] - vad_start
        last_word_end_diff = word_timestamps[-1][1][1] - vad_end
        avg_adjustment = (
            first_word_start_diff + last_word_end_diff
        ) / 2 + 10  # 微调漂移，猜测可能和标点有关

        fixed_timestamps: List[WordTimestamp] = []
        for i, (word, timestamp) in enumerate(word_timestamps):
            start_time, end_time = timestamp

            if i == 0:
                # 首个单词使用VAD起始时间
                fixed_timestamps.append(
                    (word, [vad_start, int(end_time - avg_adjustment)])
                )
            elif i == len(word_timestamps) - 1:
                # 末尾单词使用VAD结束时间
                fixed_timestamps.append(
                    (word, [int(start_time - avg_adjustment), vad_end])
                )
            else:
                # 中间单词调整时间戳
                fixed_timestamps.append(
                    (
                        word,
                        [
                            int(start_time - avg_adjustment),
                            int(end_time - avg_adjustment),
                        ],
                    )
                )

        return fixed_timestamps

    def align_sentence_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        sentences: List[str],
    ) -> List[SentenceTimestamp]:
        """对齐句子时间戳

        将单词级别的时间戳对齐到句子级别
        """
        if len(word_timestamps) != len("".join(sentences)):
            raise ValueError(
                f"单词时间戳数量({len(word_timestamps)})与句子总字符数({len(''.join(sentences))})不匹配"
            )

        aligned_timestamps: List[SentenceTimestamp] = []
        char_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len == 0:
                continue

            # 获取句子首字符和末字符的时间戳
            start_time = word_timestamps[char_index][1][0]
            end_time = word_timestamps[char_index + sentence_len - 1][1][1]

            aligned_timestamps.append((sentence, [start_time, end_time]))
            char_index += sentence_len

        return aligned_timestamps

    def process_long_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
        punc_text: str,
        temp_dir: Path,
    ) -> List[SentenceTimestamp]:
        """处理长音频片段，使用FA-ZH进行细粒度对齐"""
        start_ms, end_ms = vad_segment

        # 保存临时音频文件
        temp_audio_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.wav"
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        self.audio_processor.save_audio(
            segment_audio, temp_audio_path, self.audio_config.sample_rate
        )

        # 保存临时文本文件（移除标点）
        temp_text_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.txt"
        plain_text = self.remove_punctuation(punc_text)
        temp_text_path.write_text(plain_text, encoding="utf-8")

        # 获取FA-ZH结果
        fazh_result = self.get_fazh_result(temp_audio_path, temp_text_path)
        word_list = "".join(fazh_result[0]["text"].split())

        #! FA-ZH 的时间戳是相对于片段的，需要转换为绝对时间戳
        absolute_timestamps: List[WordTimestamp] = []
        for char, timestamp_pair in zip(word_list, fazh_result[0]["timestamp"]):
            absolute_start = timestamp_pair[0] + start_ms
            absolute_end = timestamp_pair[1] + start_ms
            absolute_timestamps.append((char, [absolute_start, absolute_end]))

        # 修正标点
        fixed_timestamps = self.fix_word_timestamps(absolute_timestamps, vad_segment)

        # 强制对齐
        sentences = self.serialize_sentences(punc_text)
        aligned_timestamps = self.align_sentence_timestamps(fixed_timestamps, sentences)

        return aligned_timestamps

    def process_audio(
        self,
        audio_path: Path,
        output_dir: Path,
        save_segments: bool = True,
    ) -> ProcessingResult:
        """处理音频文件，生成SRT字幕"""
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载音频
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        if sample_rate != self.audio_config.sample_rate:
            print(
                f"音频采样率({sample_rate})与配置不一致({self.audio_config.sample_rate})"
            )
            self.audio_config.sample_rate = sample_rate
            self.audio_processor.sample_rate = sample_rate

        vad_segments = self.get_vad_segments(audio_data)

        # 处理VAD片段
        vad_punc_texts: List[str] = []
        for vad_segment in vad_segments:
            punc_text = self.process_vad_segment(audio_data, vad_segment)
            vad_punc_texts.append(punc_text)

        # 保存VAD片段信息
        self._save_vad_info(output_dir, vad_segments, vad_punc_texts)

        if save_segments:
            self._save_vad_audio_segments(output_dir, audio_data, vad_segments)

        all_sentence_timestamps: List[SentenceTimestamp] = []

        for vad_segment, punc_text in zip(vad_segments, vad_punc_texts):
            invalid_texts = self.audio_config.invalid_texts or []
            if not self.is_valid_text(punc_text, invalid_texts):
                # 空文本或无效文本跳过
                continue

            start_ms, end_ms = vad_segment
            duration_ms = end_ms - start_ms

            # 长片段使用FA-ZH细化
            if duration_ms > self.audio_config.text_length_threshold_ms:
                sentence_timestamps = self.process_long_segment(
                    audio_data, vad_segment, punc_text, output_dir
                )
                all_sentence_timestamps.extend(sentence_timestamps)
            else:
                # 短片段直接使用VAD时间戳
                sentences = self.serialize_sentences(punc_text)
                if sentences:
                    all_sentence_timestamps.append((sentences[0], [start_ms, end_ms]))

        # 生成 .srt
        srt_path = output_dir / "output.srt"
        self.generate_srt_file(all_sentence_timestamps, srt_path)

        if save_segments:
            self._save_final_audio_segments(
                output_dir, audio_data, all_sentence_timestamps
            )

        return ProcessingResult(
            segments=all_sentence_timestamps,
            vad_segments=vad_segments,
            metadata={
                "total_sentences": len(all_sentence_timestamps),
                "total_vad_segments": len(vad_segments),
                "audio_duration_ms": int(len(audio_data) * 1000 / sample_rate),
                "sample_rate": sample_rate,
            },
        )

    def generate_srt_file(
        self,
        sentence_timestamps: List[SentenceTimestamp],
        output_path: Path,
    ) -> None:
        """生成SRT字幕文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
                start_time_str = self.ms_to_srt_time(timestamp[0])
                end_time_str = self.ms_to_srt_time(timestamp[1])
                f.write(f"{idx}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{text}\n\n")

    def _save_vad_info(
        self,
        output_dir: Path,
        vad_segments: List[VadSegment],
        punc_texts: List[str],
    ) -> None:
        """保存VAD分段信息"""
        info_path = output_dir / "vad_segments.txt"
        with open(info_path, "w", encoding="utf-8") as f:
            for i, (vad_segment, punc_text) in enumerate(
                zip(vad_segments, punc_texts), 1
            ):
                start_ms, end_ms = vad_segment
                start_str = self.ms_to_srt_time(start_ms)
                end_str = self.ms_to_srt_time(end_ms)
                f.write(f"{i}. [{start_str} --> {end_str}] {punc_text}\n")

    def _save_vad_audio_segments(
        self,
        output_dir: Path,
        audio_data: AudioSegment,
        vad_segments: List[VadSegment],
    ) -> None:
        """保存VAD音频片段"""
        vad_dir = output_dir / "vad_segments"
        vad_dir.mkdir(exist_ok=True)

        for start_ms, end_ms in vad_segments:
            segment = self.audio_processor.extract_segment(audio_data, start_ms, end_ms)
            output_path = vad_dir / f"vad_{start_ms}_{end_ms}.wav"
            self.audio_processor.save_audio(
                segment, output_path, self.audio_config.sample_rate
            )

    def _save_final_audio_segments(
        self,
        output_dir: Path,
        audio_data: AudioSegment,
        sentence_timestamps: List[SentenceTimestamp],
    ) -> None:
        """保存最终句子音频片段"""
        final_dir = output_dir / "final_segments"
        final_dir.mkdir(exist_ok=True)

        for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
            start_ms, end_ms = timestamp
            segment = self.audio_processor.extract_segment(audio_data, start_ms, end_ms)

            safe_text = text[:20].replace(" ", "_").replace("/", "_")
            output_path = final_dir / f"segment_{idx:04d}_{safe_text}.wav"

            self.audio_processor.save_audio(
                segment, output_path, self.audio_config.sample_rate
            )

    @staticmethod
    def ms_to_srt_time(ms: TimestampMs) -> str:
        total_ms = int(round(ms))
        hours = total_ms // 3600000
        remaining = total_ms % 3600000
        minutes = remaining // 60000
        remaining = remaining % 60000
        seconds = remaining // 1000
        milliseconds = remaining % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def serialize_sentences(self, text: str) -> List[str]:
        """将带标点的文本切分为句子列表"""
        for delimiter in self.sentence_delimiters:
            text = text.replace(delimiter, "|")
        sentences = [s.strip() for s in text.split("|") if s.strip()]
        return sentences

    def remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        for delimiter in self.sentence_delimiters:
            text = text.replace(delimiter, "")
        return text

    @staticmethod
    def is_valid_text(text: str, invalid_list: List[str]) -> bool:
        """检查文本是否有效"""
        return len(text) > 0 and text not in invalid_list

    def analyze_punctuation(self, text: str) -> Dict[str, int]:
        """统计句子中对应标点符号总数"""
        punctuation_counts: Dict[str, int] = {}
        for char in text:
            if char in self.sentence_delimiters:
                if char not in punctuation_counts:
                    punctuation_counts[char] = 0
                punctuation_counts[char] += 1
        return punctuation_counts


def main():
    ROOT_DIR = Path(__file__).resolve().parent
    AUDIO_DIR = ROOT_DIR / "audio"
    TEST_AUDIO = AUDIO_DIR / "23_vocals.wav"
    OUTPUT_DIR = ROOT_DIR / ".output"

    model_config = ModelConfig()
    generator = SRTGenerator(model_config=model_config)
    generator.load_models()
    result = generator.process_audio(
        audio_path=TEST_AUDIO, output_dir=OUTPUT_DIR, save_segments=True
    )
    print(result)


if __name__ == "__main__":
    main()
