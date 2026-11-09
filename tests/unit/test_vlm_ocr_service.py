"""
Unit tests for GGUF Vision-Language Model OCR service.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from oprel.core.config import Config
from oprel.server.services import ocr_service


@pytest.fixture(autouse=True)
def reset_active_ocr_model():
    ocr_service._active_ocr_model_id = None
    yield
    ocr_service._active_ocr_model_id = None


def _create_test_image_bytes(width: int = 200, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_gpu_telemetry():
    with patch("oprel.telemetry.hardware.detect_gpu", return_value={"vendor": "NVIDIA", "vram_total_mb": 8192}):
        assert ocr_service.detect_gpu() is True

    with patch("oprel.telemetry.hardware.detect_gpu", return_value=None):
        assert ocr_service.detect_gpu() is False


def test_ocr_readiness_and_status(tmp_path: Path):
    with patch.object(ocr_service, "CONFIG", Config(cache_dir=tmp_path)), \
         patch.object(ocr_service, "is_rapidocr_available", return_value=False):
        # Initially empty cache without RapidOCR
        assert ocr_service.is_ocr_ready() is False
        status = ocr_service.get_ocr_status()
        assert status["ready"] is False
        assert status["installed"] is False
        assert status["size_mb"] == 0

        # Simulate presence of cached model and mmproj
        dummy_model = tmp_path / "model.gguf"
        dummy_mmproj = tmp_path / "mmproj-model.gguf"
        dummy_model.write_bytes(b"0" * (2 * 1024 * 1024))
        dummy_mmproj.write_bytes(b"0" * (1 * 1024 * 1024))

        with patch.object(ocr_service, "get_cached_vlm_files", return_value=(dummy_model, dummy_mmproj)):
            assert ocr_service.is_ocr_ready() is True
            status_ready = ocr_service.get_ocr_status()
            assert status_ready["ready"] is True
            assert status_ready["installed"] is True
            assert status_ready["size_mb"] == 3


def test_ocr_readiness_with_rapidocr():
    with patch.object(ocr_service, "is_rapidocr_available", return_value=True):
        assert ocr_service.is_ocr_ready() is True
        status = ocr_service.get_ocr_status()
        assert status["ready"] is True
        assert status["installed"] is True
        assert status["size_mb"] == 15


def test_parse_vlm_ocr_response_json():
    img_w, img_h = 1000, 500
    vlm_output = json.dumps([
        {"text": "Invoice Title", "bbox_2d": [50, 100, 150, 400], "confidence": 0.99},
        {"text": "Total $123.45", "bbox_2d": [300, 500, 400, 800], "confidence": 0.95},
    ])

    results = ocr_service.parse_vlm_ocr_response(vlm_output, img_w, img_h)
    assert len(results) == 2

    # First item: [ymin=50, xmin=100, ymax=150, xmax=400] on 0-1000 scale
    assert results[0]["text"] == "Invoice Title"
    assert results[0]["confidence"] == 0.99
    assert results[0]["bbox_norm"] == {
        "left": 0.1,
        "top": 0.05,
        "width": 0.3,
        "height": 0.1,
    }
    # Pixel bbox: x1=100, y1=25, x2=400, y2=75
    assert results[0]["bbox"] == [[100, 25], [400, 25], [400, 75], [100, 75]]

    assert results[1]["text"] == "Total $123.45"
    assert results[1]["bbox_norm"]["left"] == 0.5


def test_parse_vlm_ocr_response_markdown_json_wrap():
    img_w, img_h = 800, 600
    vlm_output = (
        "```json\n"
        '[\n  {"text": "Account Number", "bbox_2d": [100, 50, 200, 250]}\n]\n'
        "```"
    )
    results = ocr_service.parse_vlm_ocr_response(vlm_output, img_w, img_h)
    assert len(results) == 1
    assert results[0]["text"] == "Account Number"
    assert results[0]["bbox_norm"]["top"] == 0.1
    assert results[0]["bbox_norm"]["left"] == 0.05


def test_parse_vlm_ocr_response_grounding_tags():
    img_w, img_h = 1000, 1000
    vlm_output = "<|box_start|>(100,200),(300,600)<|box_end|> Header Text"

    results = ocr_service.parse_vlm_ocr_response(vlm_output, img_w, img_h)
    assert len(results) == 1
    assert results[0]["text"] == "Header Text"
    assert results[0]["bbox_norm"]["top"] == 0.1
    assert results[0]["bbox_norm"]["left"] == 0.2
    assert results[0]["bbox_norm"]["width"] == 0.4
    assert results[0]["bbox_norm"]["height"] == 0.2


def test_parse_vlm_ocr_response_plain_text_fallback():
    img_w, img_h = 1000, 1000
    vlm_output = "Line One\nLine Two\nLine Three"

    results = ocr_service.parse_vlm_ocr_response(vlm_output, img_w, img_h)
    assert len(results) == 3
    assert results[0]["text"] == "Line One"
    assert results[1]["text"] == "Line Two"
    assert results[2]["text"] == "Line Three"
    # Ensure vertical positions increment
    assert results[0]["bbox_norm"]["top"] < results[1]["bbox_norm"]["top"] < results[2]["bbox_norm"]["top"]


def test_download_progress_events():
    with patch("oprel.server.services.ocr_service.download_model") as mock_dl, \
         patch("oprel.server.services.ocr_service.get_cached_vlm_files") as mock_cached:
        
        mock_model = Path("/tmp/model.gguf")
        mock_mmproj = Path("/tmp/mmproj.gguf")
        mock_cached.return_value = (mock_model, mock_mmproj)

        ocr_service.start_download_async()
        events = []
        for chunk in ocr_service.iter_download_progress():
            assert chunk.startswith("data: ")
            payload = json.loads(chunk[len("data: "):])
            events.append(payload)

        steps = [e["step"] for e in events]
        assert "prep" in steps
        assert "models" in steps
        assert "verify" in steps
        assert "done" in steps
        assert any(e.get("done") is True for e in events)


@pytest.mark.asyncio
async def test_run_ocr_not_ready():
    image_bytes = _create_test_image_bytes()
    with patch.object(ocr_service, "is_ocr_ready", return_value=False):
        with pytest.raises(RuntimeError, match="not ready"):
            await ocr_service.run_ocr(image_bytes)


@pytest.mark.asyncio
async def test_run_vlm_ocr_success():
    image_bytes = _create_test_image_bytes(width=400, height=200)

    mock_gen_result = MagicMock()
    mock_gen_result.text = json.dumps([
        {"text": "Extracted Text", "bbox_2d": [100, 100, 300, 500]}
    ])

    with patch.object(ocr_service, "is_ocr_ready", return_value=True), \
         patch("oprel.server.services.generation.generate_text", new_callable=AsyncMock) as mock_gen:
        
        mock_gen.return_value = mock_gen_result

        result = await ocr_service.run_ocr(image_bytes, target_model_id="dots-ocr")
        assert result["width"] == 400
        assert result["height"] == 200
        assert result["engine"] == "dots-ocr"
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "Extracted Text"
        assert "bbox" in result["results"][0]
        assert "bbox_norm" in result["results"][0]


def test_run_rapidocr_processing():
    image_bytes = _create_test_image_bytes(width=1000, height=500)
    mock_engine = MagicMock()
    mock_engine.return_value = (
        [
            [[[10, 10], [200, 10], [200, 40], [10, 40]], "Title Line", 0.98],
            [[[50, 100], [450, 100], [450, 130], [50, 130]], "Paragraph Content", 0.95],
        ],
        0.05
    )
    with patch.object(ocr_service, "get_rapidocr_engine", return_value=mock_engine):
        res = ocr_service.run_rapidocr(image_bytes)
        assert res["width"] == 1000
        assert res["height"] == 500
        assert res["engine"] == "rapidocr"
        assert len(res["results"]) == 2
        r0 = res["results"][0]
        assert r0["text"] == "Title Line"
        assert r0["confidence"] == 0.98
        assert r0["grounded"] is True
        assert r0["engine"] == "rapidocr"
        assert r0["bbox_norm"]["left"] == 0.01
        assert r0["bbox_norm"]["top"] == 0.02
        assert r0["bbox_norm"]["width"] == 0.19
        assert r0["bbox_norm"]["height"] == 0.06


@pytest.mark.asyncio
async def test_run_ocr_defaults_to_rapidocr():
    image_bytes = _create_test_image_bytes(width=800, height=400)
    mock_engine = MagicMock()
    mock_engine.return_value = (
        [
            [[[20, 20], [100, 20], [100, 50], [20, 50]], "Default RapidOCR", 0.96],
        ],
        0.02
    )
    with patch.object(ocr_service, "get_rapidocr_engine", return_value=mock_engine), \
         patch.object(ocr_service, "get_active_ocr_model_id", return_value="rapidocr"):
        res = await ocr_service.run_ocr(image_bytes)
        assert res["engine"] == "rapidocr"
        assert len(res["results"]) == 1
        assert res["results"][0]["text"] == "Default RapidOCR"


@pytest.mark.asyncio
async def test_run_auto_ocr_fallback():
    image_bytes = _create_test_image_bytes(width=800, height=400)
    mock_engine = MagicMock()
    mock_engine.return_value = (
        [
            [[[20, 20], [100, 20], [100, 50], [20, 50]], "Hybrid Line", 0.99],
        ],
        0.02
    )
    with patch.object(ocr_service, "get_rapidocr_engine", return_value=mock_engine), \
         patch.object(ocr_service, "get_cached_vlm_files", return_value=(None, None)):
        res = await ocr_service.run_ocr(image_bytes, target_model_id="auto")
        assert res["engine"] == "auto"
        assert res["results"][0]["engine"] == "auto"
        assert res["results"][0]["text"] == "Hybrid Line"


@pytest.mark.asyncio
async def test_run_auto_ocr_with_vlm_fusion():
    image_bytes = _create_test_image_bytes(width=1000, height=500)
    mock_engine = MagicMock()
    mock_engine.return_value = (
        [
            # RapidOCR detected tight line at [x=50..200, y=20..40]
            [[[50, 20], [200, 20], [200, 40], [50, 40]], "Table Cell 1", 0.98],
            [[[50, 100], [250, 100], [250, 130], [50, 130]], "Main Document Title", 0.99],
        ],
        0.02
    )

    mock_vlm_res = {
        "results": [
            # VLM detected a coarse table region covering y=10..60
            {
                "category": "Table",
                "bbox_norm": {"left": 0.04, "top": 0.03, "width": 0.50, "height": 0.10},
                "text": "Table Cell 1",
            },
            # VLM detected Title region covering y=90..150
            {
                "category": "Title",
                "bbox_norm": {"left": 0.04, "top": 0.18, "width": 0.60, "height": 0.12},
                "text": "Main Document Title",
            },
        ]
    }

    dummy_model = Path("/tmp/dummy_model.gguf")
    dummy_proj = Path("/tmp/dummy_mmproj.gguf")

    with patch.object(ocr_service, "get_rapidocr_engine", return_value=mock_engine), \
         patch.object(ocr_service, "get_cached_vlm_files", return_value=(dummy_model, dummy_proj)), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(ocr_service, "run_vlm_ocr", new_callable=AsyncMock, return_value=mock_vlm_res):

        res = await ocr_service.run_ocr(image_bytes, target_model_id="auto")
        assert res["engine"] == "auto"
        assert len(res["results"]) == 2
        # Check that RapidOCR's tight boxes were enriched with VLM's semantic categories
        assert res["results"][0]["category"] == "Table"
        assert res["results"][1]["category"] == "Title"
        # Bounding boxes remain RapidOCR's pixel-accurate coordinates
        assert res["results"][0]["bbox_norm"]["left"] == 0.05


def test_ocr_model_selection():
    status = ocr_service.set_active_ocr_model("rapidocr")
    assert status["active_model_id"] == "rapidocr"

    status_dots = ocr_service.set_active_ocr_model("dots-ocr")
    assert status_dots["active_model_id"] == "dots-ocr"

    status_auto = ocr_service.set_active_ocr_model("auto")
    assert status_auto["active_model_id"] == "auto"

    with pytest.raises(ValueError, match="Unknown OCR model"):
        ocr_service.set_active_ocr_model("invalid-model-name")


def test_image_to_data_url():
    test_bytes = b"sample_bytes"
    data_url = ocr_service.image_to_data_url(test_bytes, "image/png")
    assert data_url.startswith("data:image/png;base64,")


def test_parse_vlm_ocr_response_orientation_invariant():
    img_w, img_h = 1000, 500

    # 1. Format [xmin, ymin, xmax, ymax]
    output_xy = json.dumps([
        {"text": "Ragul T", "bbox_2d": [250, 153, 378, 176]}
    ])
    res_xy = ocr_service.parse_vlm_ocr_response(output_xy, img_w, img_h)
    assert len(res_xy) == 1
    assert res_xy[0]["text"] == "Ragul T"
    assert res_xy[0]["bbox_norm"]["left"] == 0.25
    assert res_xy[0]["bbox_norm"]["top"] == 0.153
    assert res_xy[0]["bbox_norm"]["width"] == 0.128
    assert res_xy[0]["bbox_norm"]["height"] == 0.023

    # 2. Format [ymin, xmin, ymax, xmax] produces identical normalized bounding box
    output_yx = json.dumps([
        {"text": "Ragul T", "bbox_2d": [153, 250, 176, 378]}
    ])
    res_yx = ocr_service.parse_vlm_ocr_response(output_yx, img_w, img_h)
    assert len(res_yx) == 1
    assert res_yx[0]["text"] == "Ragul T"
    assert res_yx[0]["bbox_norm"]["left"] == 0.25
    assert res_yx[0]["bbox_norm"]["top"] == 0.153
    assert res_yx[0]["bbox_norm"]["width"] == 0.128
    assert res_yx[0]["bbox_norm"]["height"] == 0.023


def test_parse_vlm_ocr_response_multilingual():
    img_w, img_h = 1000, 500
    vlm_output = json.dumps([
        {"text": "இந்திய அரசாங்கம்", "bbox_2d": [180, 50, 350, 80]},
        {"text": "Government of India", "bbox_2d": [180, 90, 360, 120]},
        {"text": "ராகுல் த", "bbox_2d": [250, 130, 350, 155]},
        {"text": "Ragul T", "bbox_2d": [250, 160, 340, 185]},
    ])

    results = ocr_service.parse_vlm_ocr_response(vlm_output, img_w, img_h)
    assert len(results) == 4
    assert results[0]["text"] == "இந்திய அரசாங்கம்"
    assert results[2]["text"] == "ராகுல் த"
    # Verify non-empty bboxes
    for r in results:
        assert r["bbox_norm"]["width"] > 0
        assert r["bbox_norm"]["height"] > 0


def test_parse_vlm_ocr_response_dots_ocr_layout():
    img_w, img_h = 1000, 500
    dots_output = json.dumps([
        {
            "category": "Title",
            "bbox": [50, 20, 450, 60],
            "text": "Audience Build Job"
        },
        {
            "category": "Text",
            "bbox": [50, 80, 800, 140],
            "text": "Which email address is queued multiple times?"
        },
        {
            "category": "Table",
            "bbox": [50, 160, 600, 400],
            "text": "<table><tr><td>mkumar@healthlinkdimensions.com</td><td>17</td></tr></table>"
        },
        {
            "category": "List-item",
            "bbox": [10, 200, 250, 240],
            "text": "HLDRegister user count in development"
        }
    ])

    results = ocr_service.parse_vlm_ocr_response(dots_output, img_w, img_h)
    assert len(results) == 4

    # Item 0: Title [x1=50, y1=20, x2=450, y2=60]
    assert results[0]["text"] == "Audience Build Job"
    assert results[0]["category"] == "Title"
    assert results[0]["grounded"] is True
    assert results[0]["bbox_norm"]["left"] == 0.05
    assert results[0]["bbox_norm"]["top"] == 0.02
    assert results[0]["bbox_norm"]["width"] == 0.4
    assert results[0]["bbox_norm"]["height"] == 0.04
    # Pixel bbox: x1=50, y1=10, x2=450, y2=30
    assert results[0]["bbox"] == [[50, 10], [450, 10], [450, 30], [50, 30]]

    # Item 2: Table
    assert results[2]["category"] == "Table"
    assert results[2]["bbox_norm"]["left"] == 0.05
    assert results[2]["bbox_norm"]["top"] == 0.16
    assert results[2]["bbox_norm"]["width"] == 0.55
    assert results[2]["bbox_norm"]["height"] == 0.24

    # Item 3: List-item in left sidebar
    assert results[3]["category"] == "List-item"
    assert results[3]["bbox_norm"]["left"] == 0.01
    assert results[3]["bbox_norm"]["width"] == 0.24


def test_parse_vlm_ocr_response_dots_ocr_object_wrap():
    img_w, img_h = 800, 600
    dots_wrapped = json.dumps({
        "layout": [
            {
                "category": "Section-header",
                "bbox": [100, 50, 500, 100],
                "content": "Emails Queued Multiple Times"
            }
        ]
    })

    results = ocr_service.parse_vlm_ocr_response(dots_wrapped, img_w, img_h)
    assert len(results) == 1
    assert results[0]["text"] == "Emails Queued Multiple Times"
    assert results[0]["category"] == "Section-header"
    assert results[0]["grounded"] is True
    assert results[0]["bbox_norm"]["left"] == 0.1
    assert results[0]["bbox_norm"]["top"] == 0.05

