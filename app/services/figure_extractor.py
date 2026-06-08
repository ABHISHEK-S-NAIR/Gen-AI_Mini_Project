"""
Vision-based figure extraction service.
Extracts images from PDFs and uses vision models to extract structured data
from charts, graphs, and scientific figures.
"""

import base64
import json
from io import BytesIO
from typing import Optional

import pdfplumber
from PIL import Image

from app.config import settings


def _extract_figure_image(
    pdf_bytes: bytes, page_num: int, bbox: Optional[list[float]] = None
) -> Optional[bytes]:
    """
    Extract a figure image from a PDF page.

    Args:
        pdf_bytes: PDF file bytes
        page_num: Page number (1-indexed)
        bbox: Optional bounding box [x0, top, x1, bottom]

    Returns:
        PNG image bytes or None if extraction fails
    """
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return None

            page = pdf.pages[page_num - 1]

            # If bbox provided, crop to that region; otherwise use full page
            if bbox and len(bbox) == 4:
                x0, top, x1, bottom = bbox
                # Ensure coordinates are within page bounds
                x0 = max(0, min(x0, page.width))
                x1 = max(0, min(x1, page.width))
                top = max(0, min(top, page.height))
                bottom = max(0, min(bottom, page.height))

                # Only crop if we have a valid region
                if x1 > x0 and bottom > top:
                    cropped = page.crop((x0, top, x1, bottom))
                    img = cropped.to_image(resolution=200)
                else:
                    img = page.to_image(resolution=200)
            else:
                img = page.to_image(resolution=200)

            # Convert to PIL Image and save as PNG
            pil_image = img.original

            # Convert to RGB if necessary (some PDFs have CMYK or other color spaces)
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")

            # Save to bytes
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format="PNG")
            return img_bytes.getvalue()

    except Exception as e:
        print(f"Error extracting figure image: {e}")
        return None


def _encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _build_vision_prompt(figure_type: str = "auto") -> str:
    """
    Build prompt for vision model to analyze figure.

    Args:
        figure_type: Type hint ("chart", "graph", "diagram", "table", "auto")
    """
    _ = figure_type  # Reserved for future use
    return """Analyze this scientific figure and extract structured information.

Please provide:
1. **Figure Type**: Identify the type (bar chart, line graph, scatter plot, diagram, flowchart, table, microscopy image, etc.)
2. **Title/Caption**: Extract any visible title or caption
3. **Axis Labels**: For charts/graphs, identify X and Y axis labels and units
4. **Data Points**: Extract key data points, trends, or numerical values visible in the figure
5. **Legend**: Describe any legend or color coding
6. **Key Findings**: Summarize the main insight or message conveyed by this figure

Format your response as JSON with these keys:
- type: string (figure type)
- title: string (title/caption if visible, otherwise "")
- x_axis: string (X axis label with units, if applicable)
- y_axis: string (Y axis label with units, if applicable)
- data: array of objects with relevant data points (e.g., [{{"label": "...", "value": ...}}])
- legend: array of strings (legend items)
- summary: string (1-2 sentence summary of key findings)

If this is not a data figure (e.g., a photo, logo, or decorative element), set type to "non-data" and provide only type and summary fields.
"""


async def analyze_figure_with_vision(
    image_bytes: bytes,
    figure_metadata: Optional[dict] = None,
    provider: Optional[str] = None,
) -> dict:
    """
    Analyze a figure image using a vision model.

    Args:
        image_bytes: PNG image bytes
        figure_metadata: Optional metadata (page, bbox, etc.)
        provider: Optional LLM provider override

    Returns:
        Dictionary with extracted figure data
    """
    if not image_bytes:
        return {
            "error": "No image data provided",
            "raw_response": None,
        }

    # Use configured provider or default
    llm_provider = provider or settings.llm_provider

    # Build the vision prompt
    prompt = _build_vision_prompt()

    # Encode image
    base64_image = _encode_image_base64(image_bytes)

    # Different providers handle vision differently
    try:
        if llm_provider == "openai":
            # OpenAI vision format
            response = await _call_openai_vision(prompt, base64_image)
        elif llm_provider == "gemini":
            # Gemini vision format
            response = await _call_gemini_vision(prompt, image_bytes)
        elif llm_provider == "groq":
            # Groq uses meta-llama/llama-4-scout-17b-16e-instruct (replacement for deprecated llama-3.2-90b-vision-preview)
            response = await _call_groq_vision(prompt, base64_image)
        else:
            return {
                "error": f"Vision not supported for provider: {llm_provider}",
                "raw_response": None,
            }

        # Try to parse JSON response
        extracted_data = _parse_vision_response(response)

        # Add metadata
        if figure_metadata:
            extracted_data["metadata"] = figure_metadata

        extracted_data["raw_response"] = response

        return extracted_data

    except Exception as e:
        error_msg = str(e)
        # Provide helpful guidance for common errors
        if "model_permission_blocked" in error_msg and llm_provider == "groq":
            error_msg = (
                f"{error_msg}\n\n"
                "To fix: Enable 'meta-llama/llama-4-scout-17b-16e-instruct' in your Groq project settings at:\n"
                "https://console.groq.com/settings/project/limits\n\n"
                "Or use a different provider: --provider openai or --provider gemini"
            )
        return {
            "error": f"Vision analysis failed: {error_msg}",
            "raw_response": None,
        }


async def _call_openai_vision(prompt: str, base64_image: str) -> str:
    """Call OpenAI GPT-4 Vision."""
    import os

    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o",  # or gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 2000,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


async def _call_gemini_vision(prompt: str, image_bytes: bytes) -> str:
    """Call Google Gemini Vision."""
    import os

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise ValueError("google-generativeai package not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)  # type: ignore

    # Load model
    model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore

    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))

    # Generate content
    response = model.generate_content([prompt, image])  # type: ignore

    return response.text  # type: ignore


async def _call_groq_vision(prompt: str, base64_image: str) -> str:
    """Call Groq Vision (llama-3.2-90b-vision)."""
    import os

    from groq import AsyncGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = AsyncGroq(api_key=api_key)

    response = await client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    return content if content is not None else ""


def _parse_vision_response(response: str) -> dict[str, object]:
    """
    Parse vision model response into structured data.
    Attempts to extract JSON from the response.
    """
    try:
        # Try to find JSON in the response
        # Sometimes models wrap JSON in markdown code blocks
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Parse JSON
        data = json.loads(response)

        # Ensure required fields
        if "type" not in data:
            data["type"] = "unknown"
        if "summary" not in data:
            data["summary"] = ""

        return data

    except json.JSONDecodeError:
        # Fallback: return raw response in summary
        return {
            "type": "unknown",
            "summary": response[:500],  # Truncate to reasonable length
            "parse_error": "Failed to parse JSON response",
        }


async def extract_figures_with_vision(
    pdf_bytes: bytes,
    figures_metadata: list[dict[str, object]],
    max_figures: Optional[int] = None,
) -> list[dict]:
    """
    Extract and analyze figures from a PDF using vision models.

    Args:
        pdf_bytes: PDF file bytes
        figures_metadata: List of figure metadata from pdfplumber
            (must include 'page' and optionally 'bbox')
        max_figures: Optional limit on number of figures to process

    Returns:
        List of analyzed figures with extracted data
    """
    analyzed_figures = []

    # Limit number of figures if specified
    figures_to_process = (
        figures_metadata[:max_figures] if max_figures else figures_metadata
    )

    for fig_meta in figures_to_process:
        page = fig_meta.get("page")
        if not isinstance(page, int):
            continue

        bbox = fig_meta.get("bbox")
        if bbox is not None and not isinstance(bbox, list):
            bbox = None

        if not page:
            continue

        # Extract image
        image_bytes = _extract_figure_image(pdf_bytes, page, bbox)

        if not image_bytes:
            analyzed_figures.append(
                {
                    **fig_meta,
                    "error": "Failed to extract image",
                    "analysis": None,
                }
            )
            continue

        # Analyze with vision model
        analysis = await analyze_figure_with_vision(
            image_bytes,
            figure_metadata={
                "page": page,
                "bbox": bbox,
                "figure_index": fig_meta.get("figure_index"),
            },
        )

        analyzed_figures.append(
            {
                **fig_meta,
                "analysis": analysis,
                "image_extracted": True,
            }
        )

    return analyzed_figures


def build_figures_text(figures: list[dict[str, object]]) -> str:
    """
    Format analyzed figures as text for inclusion in document context.

    Args:
        figures: List of analyzed figures

    Returns:
        Formatted text representation
    """
    if not figures:
        return ""

    blocks = []
    for fig in figures:
        page = fig.get("page", "?")
        index = fig.get("figure_index", "?")
        analysis = fig.get("analysis")

        if not analysis or not isinstance(analysis, dict) or analysis.get("error"):
            continue

        # Skip non-data figures
        if analysis.get("type") == "non-data":
            continue

        fig_type = analysis.get("type", "unknown")
        title = analysis.get("title", "")
        summary = analysis.get("summary", "")

        lines = [f"Figure {index} (page {page}) - {fig_type}"]

        if title and isinstance(title, str):
            lines.append(f"Title: {title}")

        # Add axis labels for charts/graphs
        x_axis = analysis.get("x_axis")
        if x_axis and isinstance(x_axis, str):
            lines.append(f"X-axis: {x_axis}")

        y_axis = analysis.get("y_axis")
        if y_axis and isinstance(y_axis, str):
            lines.append(f"Y-axis: {y_axis}")

        # Add legend
        legend = analysis.get("legend")
        if legend and isinstance(legend, list):
            legend_str = ", ".join(str(item) for item in legend)
            lines.append(f"Legend: {legend_str}")

        # Add data points if available
        data = analysis.get("data")
        if data and isinstance(data, list):
            data_items = data[:5]  # Limit to first 5 data points
            data_str = "; ".join(
                [
                    f"{d.get('label', '')}: {d.get('value', '')}"
                    for d in data_items
                    if isinstance(d, dict)
                ]
            )
            if data_str:
                lines.append(f"Data: {data_str}")

        # Add summary
        if summary and isinstance(summary, str):
            lines.append(f"Summary: {summary}")

        blocks.append("\n".join(lines))

    if not blocks:
        return ""

    return "[FIGURES]\n" + "\n\n".join(blocks)
