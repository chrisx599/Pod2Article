from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aliyun_asr_client  # noqa: E402
from aliyun_asr_client import AliyunAsrClient, AliyunAsrConfig  # noqa: E402


class StubAliyunAsrClient(AliyunAsrClient):
    def __init__(self, config: AliyunAsrConfig) -> None:
        super().__init__(config)
        self.upload_saw_existing_file = False
        self.uploaded_url = ""

    def download_youtube_audio(self, video_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"{video_id}.m4a"
        audio_path.write_bytes(b"fake audio")
        return audio_path

    def upload_file_and_get_url(self, file_path: Path) -> str:
        self.upload_saw_existing_file = file_path.exists()
        self.uploaded_url = f"oss://dashscope-test/{file_path.name}"
        return self.uploaded_url

    def submit_task(self, audio_url: str) -> str:
        self.assert_url = audio_url
        return "task-1"

    def wait_for_task(self, task_id: str) -> dict:
        return {
            "output": {
                "task_status": "SUCCEEDED",
                "results": [{"transcription_url": "https://example.test/result.json"}],
            }
        }

    def fetch_result(self, task_payload: dict) -> dict:
        return {
            "transcripts": [
                {
                    "sentences": [
                        {"begin_time": 0, "end_time": 1000, "text": "本地音频上传后转写成功。"}
                    ]
                }
            ]
        }


class AliyunAsrClientTestCase(unittest.TestCase):
    def test_transcribe_youtube_video_downloads_then_uploads_local_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = StubAliyunAsrClient(
                AliyunAsrConfig(api_key="test-key", audio_dir=Path(tmpdir), keep_audio=False)
            )
            payload = client.transcribe_youtube_video("abc123def45")

        self.assertTrue(client.upload_saw_existing_file)
        self.assertEqual(client.uploaded_url, "oss://dashscope-test/abc123def45.m4a")
        self.assertEqual(payload["uploaded_audio_url"], client.uploaded_url)
        self.assertEqual(payload["transcript"][0]["snippet"], "本地音频上传后转写成功。")

    def test_submit_task_enables_oss_resource_resolution(self) -> None:
        client = AliyunAsrClient(AliyunAsrConfig(api_key="test-key"))
        with mock.patch.object(
            aliyun_asr_client,
            "_request_json",
            return_value={"output": {"task_id": "task-1"}},
        ) as request_json:
            task_id = client.submit_task("oss://dashscope-test/input.m4a")

        self.assertEqual(task_id, "task-1")
        headers = request_json.call_args.kwargs["headers"]
        self.assertEqual(headers["X-DashScope-Async"], "enable")
        self.assertEqual(headers["X-DashScope-OssResourceResolve"], "enable")


if __name__ == "__main__":
    unittest.main()
