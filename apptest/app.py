"""
CancerGrading Assistant — Gradio Web Application
A doctor-friendly interface for classifying breast cancer histopathology images
into Grade 1, Grade 2, or Grade 3 using trained deep learning models.
Supports OOD detection to reject non-breast-cell or uncertain inputs.
"""
import os
import csv
import io
import tempfile
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from predict import (
    get_available_models,
    predict,
    predict_batch,
    CLASS_NAMES,
    CLASS_NAMES_EXTENDED,
    ENERGY_THRESHOLD,
    SOFTMAX_CONFIDENCE_HIGH,
)

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

GRADE_COLORS = {
    "GRADE1":     "#4CAF50",   # Green — low severity
    "GRADE2":     "#FF9800",   # Orange — moderate
    "GRADE3":     "#F44336",   # Red — high severity
    "UNKNOWN":    "#9E9E9E",   # Gray — uncertain, may be cell but can't grade
    "NOT_A_CELL": "#607D8B",   # Blue-gray — not a breast cell image
}

# User-friendly descriptions for OOD results
OOD_MESSAGES = {
    "UNKNOWN": (
        "⚠️ <strong>Uncertain</strong> — This image has some visual similarity to "
        "breast cells, but the model cannot confidently determine its grade. "
        "It may be normal tissue, benign cells, or an ambiguous sample. "
        "Consider further review by a pathologist."
    ),
    "NOT_A_CELL": (
        "🚫 <strong>Not a Breast Cell Image</strong> — This image does not appear "
        "to be a breast histopathology cell. The model was only trained on breast "
        "cancer cell images (Grade 1–3). Please upload a valid histopathology image."
    ),
}

# Get only models with checkpoints available
AVAILABLE_MODELS = get_available_models()
DEFAULT_MODEL = None
for preferred in ["Exp4_Fusion (Fusion+GELU+SE)", "ExpVGG19_Fusion_GELU_SE (VGG19 Fusion+GELU+SE)"]:
    if preferred in AVAILABLE_MODELS:
        DEFAULT_MODEL = preferred
        break
if DEFAULT_MODEL is None and AVAILABLE_MODELS:
    DEFAULT_MODEL = AVAILABLE_MODELS[0]


# ─────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────

def predict_single(
    image: Image.Image,
    model_name: str,
) -> Tuple[str, str, str]:
    """
    Run single image prediction and return components for Gradio UI updates.

    Returns:
        (result_html, confidence_bars_html, error_message)
    """
    if image is None:
        return ("", "", "⚠ Please upload an image first.")

    try:
        idx, label, conf, _ = predict(image, model_name)

        # ── Handle OOD results (UNKNOWN / NOT_A_CELL) ──────────────
        if label in OOD_MESSAGES:
            grade_color = GRADE_COLORS.get(label, "#333")
            ood_msg = OOD_MESSAGES[label]
            result_html = (
                f'<div style="text-align:center; padding:20px;">'
                f'  <div style="font-size:48px; font-weight:bold; color:{grade_color};'
                f'       margin:10px 0;">{label}</div>'
                f'  <div style="font-size:15px; color:#555; max-width:500px;'
                f'       margin:10px auto; line-height:1.5;">{ood_msg}</div>'
                f'</div>'
            )
            # No confidence bars for OOD results (avoid confusion)
            return (result_html, "", "")

        # ── Normal grade result ─────────────────────────────────────
        grade_color = GRADE_COLORS.get(label, "#333")
        result_html = (
            f'<div style="text-align:center; padding:20px;">'
            f'  <div style="font-size:16px; color:#666;">Predicted Grade</div>'
            f'  <div style="font-size:48px; font-weight:bold; color:{grade_color};'
            f'       margin:10px 0;">{label}</div>'
            f'  <div style="font-size:20px; color:#333;">'
            f'    Confidence: {conf[label]*100:.1f}%</div>'
            f'</div>'
        )

        bar_html = _build_confidence_bars(conf, label)
        return (result_html, bar_html, "")

    except Exception as e:
        return ("", "", f"⚠ Error: {str(e)}")


def _build_confidence_bars(
    conf: dict,
    label: str,
) -> str:
    """Build HTML confidence bar chart for the 3 grade classes."""
    bar_html = ""
    for cls in CLASS_NAMES:
        pct = conf.get(cls, 0) * 100
        color = GRADE_COLORS.get(cls, "#999")
        is_highest = cls == label
        bar_html += f"""
        <div style="margin:8px 0;">
          <div style="display:flex; justify-content:space-between; font-size:14px;">
            <span style="font-weight:{'bold' if is_highest else 'normal'};
                         color:{color};">{cls}</span>
            <span style="font-weight:{'bold' if is_highest else 'normal'};">
              {pct:.1f}%</span>
          </div>
          <div style="background:#e0e0e0; border-radius:8px; height:24px; overflow:hidden;">
            <div style="background:{color}; width:{pct}%; height:100%;
                        border-radius:8px; transition:width 0.5s;"></div>
          </div>
        </div>
        """
    # Show entropy if available
    if "entropy" in conf:
        entropy_val = conf["entropy"]
        bar_html += f"""
        <div style="margin-top:10px; font-size:12px; color:#999;
                    text-align:center;">
          Normalized Entropy: {entropy_val:.3f}
          &nbsp;(0=confident, 1=uniform)
        </div>
        """
    return bar_html


def _get_filename(file_obj) -> str:
    """Safely extract original filename from a Gradio FileData object.
    
    Handles:
      - Gradio 4.x: uses .orig_name or .name
      - Gradio 3.x: uses .name
      - Fallback: extract from temp path (only if it's a file)
    """
    # Gradio 4.x stores original name in .orig_name
    if hasattr(file_obj, "orig_name") and file_obj.orig_name:
        return os.path.basename(file_obj.orig_name)
    # Gradio 3.x / fallback
    if hasattr(file_obj, "name") and file_obj.name:
        return os.path.basename(file_obj.name)
    # Ultimate fallback: use temp file path only if it's an actual file
    return os.path.basename(file_obj.path)


def predict_multiple(
    files: List[gr.FileData],
    model_name: str,
) -> Tuple[str, Optional[str], str]:
    """
    Run batch prediction and return results table + CSV download + status.

    Returns:
        (results_html, csv_download_path or None, status_text)
    """
    if not files:
        return ("", None, "⚠ No files uploaded.")

    try:
        images = []
        filenames = []
        for f in files:
            # Validate that the path is an actual file (not a directory)
            file_path = f.path if hasattr(f, "path") else str(f)
            if not os.path.isfile(file_path):
                raise ValueError(f"'{_get_filename(f)}' is not a valid image file.")
            img = Image.open(file_path)
            images.append(img)
            filenames.append(_get_filename(f))

        results = predict_batch(images, model_name)

        # Build HTML table
        table_rows = ""
        csv_rows = [["Filename", "Predicted Grade", "Confidence",
                     "GRADE1 (%)", "GRADE2 (%)", "GRADE3 (%)", "Entropy"]]
        ood_count = 0

        for i, (idx, label, conf, _) in enumerate(results):
            fn = filenames[i] if i < len(filenames) else f"Image {i+1}"
            color = GRADE_COLORS.get(label, "#333")

            if label in OOD_MESSAGES:
                ood_count += 1
                # Truncate long filenames
                display_fn = fn if len(fn) <= 40 else fn[:37] + "..."
                table_rows += f"""
                <tr>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;"
                      title="{fn}">{display_fn}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             font-weight:bold; color:{color};">{label}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             color:#999;">—</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center; color:#ccc;">N/A</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center; color:#ccc;">N/A</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center; color:#ccc;">N/A</td>
                </tr>
                """
            else:
                pct = conf[label] * 100
                display_fn = fn if len(fn) <= 40 else fn[:37] + "..."
                table_rows += f"""
                <tr>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;"
                      title="{fn}">{display_fn}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             font-weight:bold; color:{color};">{label}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;">
                    {pct:.1f}%</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center;">{conf["GRADE1"]*100:.1f}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center;">{conf["GRADE2"]*100:.1f}</td>
                  <td style="padding:6px 10px; border-bottom:1px solid #ddd;
                             text-align:center;">{conf["GRADE3"]*100:.1f}</td>
                </tr>
                """

            csv_rows.append([
                fn,
                label,
                f"{conf.get(label, 0)*100:.1f}%" if label not in OOD_MESSAGES
                else f"{max(conf.get(c,0) for c in CLASS_NAMES)*100:.1f}% (OOD)",
                f"{conf.get('GRADE1', 0)*100:.1f}",
                f"{conf.get('GRADE2', 0)*100:.1f}",
                f"{conf.get('GRADE3', 0)*100:.1f}",
                f"{conf.get('entropy', 0):.3f}",
            ])

        warning_banner = ""
        if ood_count > 0:
            warning_banner = f"""
            <div style="padding:10px 15px; margin-bottom:12px;
                        background:#FFF3E0; border-left:4px solid #FF9800;
                        border-radius:4px; font-size:14px; color:#E65100;">
              ⚠ <strong>{ood_count} of {len(results)}</strong> image(s) were flagged as
              <strong>UNKNOWN</strong> or <strong>NOT_A_CELL</strong>.
              These are not valid breast cancer grade predictions.
            </div>
            """

        results_html = warning_banner + f"""
        <div style="max-height:400px; overflow-y:auto;">
          <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <thead style="position:sticky; top:0; background:#f5f5f5;">
              <tr>
                <th style="padding:8px 10px; text-align:left; border-bottom:2px solid #ddd;">
                  Filename</th>
                <th style="padding:8px 10px; text-align:left; border-bottom:2px solid #ddd;">
                  Grade</th>
                <th style="padding:8px 10px; text-align:left; border-bottom:2px solid #ddd;">
                  Confidence</th>
                <th style="padding:8px 10px; text-align:center; border-bottom:2px solid #ddd;
                           color:#4CAF50;">GRADE1%</th>
                <th style="padding:8px 10px; text-align:center; border-bottom:2px solid #ddd;
                           color:#FF9800;">GRADE2%</th>
                <th style="padding:8px 10px; text-align:center; border-bottom:2px solid #ddd;
                           color:#F44336;">GRADE3%</th>
              </tr>
            </thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>
        </div>
        <div style="margin-top:10px; font-size:13px; color:#666;">
          Total: {len(results)} images processed
          ({len(results) - ood_count} graded, {ood_count} rejected)
        </div>
        """

        # Generate CSV
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerows(csv_rows)
        csv_content = csv_buffer.getvalue()

        csv_path = os.path.join(tempfile.gettempdir(), "cancer_grading_results.csv")
        with open(csv_path, "w", newline="") as f:
            f.write(csv_content)

        return (results_html, csv_path,
                f"✅ Successfully processed {len(results)} images.")

    except Exception as e:
        return ("", None, f"⚠ Error: {str(e)}")


def run_all_models_comparison(
    image: Image.Image,
) -> Tuple[str, str]:
    """
    Run prediction through ALL available models and compare results.

    Returns:
        (comparison_html, status_text)
    """
    if image is None:
        return ("", "⚠ Please upload an image first.")

    rows = ""
    for model_name in AVAILABLE_MODELS:
        try:
            idx, label, conf, _ = predict(image, model_name)
            color = GRADE_COLORS.get(label, "#333")

            if label in OOD_MESSAGES:
                max_grade_conf = max(conf.get(c, 0) for c in CLASS_NAMES) * 100
                rows += f"""
                <tr>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-size:13px;">
                    {model_name.split("(")[0].strip()}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-weight:bold;
                             color:{color}; font-size:14px;">{label}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-size:13px;
                             color:#999;">Max grade conf: {max_grade_conf:.1f}%</td>
                </tr>
                """
            else:
                pct = conf[label] * 100
                bars = ""
                for cls in CLASS_NAMES:
                    c = GRADE_COLORS.get(cls, "#999")
                    p = conf[cls] * 100
                    bars += (
                        f'<div style="font-size:12px; color:{c}; display:inline-block; '
                        f'width:60px; text-align:center;">'
                        f'{cls.split("E")[1]}: {p:.0f}%</div>'
                    )
                rows += f"""
                <tr>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-size:13px;">
                    {model_name.split("(")[0].strip()}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-weight:bold;
                             color:{color}; font-size:14px;">{label}</td>
                  <td style="padding:8px; border-bottom:1px solid #eee; font-size:13px;">
                    {pct:.1f}%</td>
                </tr>
                """
        except Exception as e:
            rows += f"""
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee; font-size:13px;
                         color:#999;">{model_name.split("(")[0].strip()}</td>
              <td colspan="2" style="padding:8px; border-bottom:1px solid #eee;
                                     font-size:13px; color:#999;">Error: {str(e)}</td>
            </tr>
            """

    html = f"""
    <div style="max-height:400px; overflow-y:auto;">
      <table style="width:100%; border-collapse:collapse;">
        <thead style="position:sticky; top:0; background:#f5f5f5;">
          <tr>
            <th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">
              Model</th>
            <th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">
              Predicted Grade</th>
            <th style="padding:8px; text-align:left; border-bottom:2px solid #ddd;">
              Confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """
    return (html, f"✅ Compared {len(AVAILABLE_MODELS)} models.")


# ─────────────────────────────────────────────────────────────────────────
# Gradio UI Builder
# ─────────────────────────────────────────────────────────────────────────

def build_interface():
    """Build and return the Gradio Blocks interface."""

    # CSS for consistent styling
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', Arial, sans-serif; }
    h1 { margin-bottom: 5px !important; }
    .gr-tabs { border: none !important; }
    """

    with gr.Blocks(
        css=custom_css,
        title="CancerGrading Assistant",
        theme=gr.themes.Soft(),
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🔬 CancerGrading Assistant
            **Breast Cancer Histopathology Image Classifier** — Grade 1, 2, or 3
            """
        )

        # Model selector (shared across tabs)
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL,
            label="🧠 Select Model",
            info="Choose the trained model for classification. Fusion models are recommended.",
            interactive=True,
        )

        # ── Tabs ────────────────────────────────────────────────────────
        with gr.Tabs():
            # ──── TAB 1: Single Image ────────────────────────────────────
            with gr.TabItem("📷 Single Image", id="single"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Histopathology Image",
                            height=350,
                        )
                        with gr.Row():
                            classify_btn = gr.Button(
                                "🔍 Classify", variant="primary", size="lg"
                            )
                            clear_btn = gr.Button("🗑 Clear", size="lg")

                    with gr.Column(scale=1):
                        result_display = gr.HTML(
                            label="Prediction Result",
                            value=(
                                '<div style="text-align:center; padding:40px; '
                                'color:#999;">Upload an image and click '
                                '<strong>Classify</strong> to see results.</div>'
                            ),
                        )
                        confidence_bars = gr.HTML(label="Confidence Breakdown")
                        error_text = gr.Markdown(visible=False)

                # Wire up buttons
                classify_btn.click(
                    fn=predict_single,
                    inputs=[image_input, model_selector],
                    outputs=[result_display, confidence_bars, error_text],
                )
                clear_btn.click(
                    fn=lambda: (
                        None,
                        '<div style="text-align:center; padding:40px; '
                        'color:#999;">Upload an image and click '
                        '<strong>Classify</strong> to see results.</div>',
                        "",
                    ),
                    outputs=[image_input, result_display, confidence_bars],
                )

            # ──── TAB 2: Batch Processing ────────────────────────────────
            with gr.TabItem("📁 Batch Processing", id="batch"):
                gr.Markdown(
                    """
                    Upload **multiple** histopathology images for batch classification.
                    Results can be downloaded as a CSV report.
                    Images that don't look like breast cells will be flagged as
                    **UNKNOWN** or **NOT_A_CELL**.
                    """
                )
                batch_files = gr.File(
                    file_count="multiple",
                    label="Upload Images (JPG/PNG)",
                    file_types=[".jpg", ".jpeg", ".png"],
                )
                batch_btn = gr.Button(
                    "🚀 Process All", variant="primary", size="lg"
                )
                batch_results = gr.HTML(label="Batch Results")
                batch_status = gr.Markdown()
                csv_download = gr.File(label="📥 Download CSV Report")

                batch_btn.click(
                    fn=predict_multiple,
                    inputs=[batch_files, model_selector],
                    outputs=[batch_results, csv_download, batch_status],
                )

            # ──── TAB 3: Model Comparison ────────────────────────────────
            with gr.TabItem("⚖️ Model Comparison", id="compare"):
                gr.Markdown(
                    """
                    Upload a **single image** and compare predictions across
                    **all available models**. Useful for research or
                    cross-validating results.
                    """
                )
                compare_image = gr.Image(
                    type="pil",
                    label="Upload Histopathology Image",
                    height=300,
                )
                compare_btn = gr.Button(
                    "🔬 Compare All Models", variant="primary", size="lg"
                )
                compare_results = gr.HTML(label="Comparison Results")
                compare_status = gr.Markdown()

                compare_btn.click(
                    fn=run_all_models_comparison,
                    inputs=[compare_image],
                    outputs=[compare_results, compare_status],
                )

        # ── Footer ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **© CancerGrading Project** — Built with PyTorch & Gradio
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🏥 CancerGrading Assistant")
    print("=" * 60)
    print(f"  Available models: {len(AVAILABLE_MODELS)}")
    for m in AVAILABLE_MODELS:
        print(f"    • {m}")
    print(f"  Device: {'cuda' if not DEFAULT_MODEL is None else 'cpu (checking)'}")
    print(f"  Default model: {DEFAULT_MODEL}")
    print(f"  OOD Detection: ACTIVE (Energy-based)")
    print(f"    - Energy threshold: {ENERGY_THRESHOLD}")
    print(f"    - Softmax confidence threshold: {SOFTMAX_CONFIDENCE_HIGH}")
    print(f"    - Non-breast-cell images → flagged as NOT_A_CELL / UNKNOWN")
    print("=" * 60)

    if not AVAILABLE_MODELS:
        print("\n  ⚠ WARNING: No model checkpoints found!")
        print("  Please train models first using main.py (option 1-5).")
        print("  Checkpoints should be in 'outputs/' or 'newoutputs/' directories.\n")

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allows network access
        server_port=7860,
        share=False,              # Set True for public link via Gradio
        inbrowser=True,           # Auto-open browser
    )