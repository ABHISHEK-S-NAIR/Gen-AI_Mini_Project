#!/usr/bin/env python3
"""
Test script for vision-based figure extraction.
Demonstrates the figure extraction feature with sample PDFs.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.figure_extractor import (
    _extract_figure_image,
    analyze_figure_with_vision,
    build_figures_text,
    extract_figures_with_vision,
)
from app.services.text_extractor import extract_pdf_content


async def test_figure_extraction(pdf_path: str, provider: str = "groq"):
    """
    Test figure extraction on a PDF file.

    Args:
        pdf_path: Path to PDF file
        provider: Vision model provider (groq, openai, gemini)
    """
    print(f"\n{'=' * 80}")
    print(f"Testing Figure Extraction: {pdf_path}")
    print(f"Provider: {provider}")
    print(f"{'=' * 80}\n")

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return

    # Check API key
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    api_key_env = api_key_map.get(provider)
    if not api_key_env or not os.getenv(api_key_env):
        print(f"❌ Error: {api_key_env} not set in environment")
        print(f"   Please set it with: export {api_key_env}='your-key'")
        return

    # Set provider
    settings.llm_provider = provider

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("📄 Extracting PDF content...")

    # Extract basic content
    raw_text, tables, figures = extract_pdf_content(pdf_bytes)

    print(f"✓ Found {len(figures)} figure(s) in PDF")
    print(f"✓ Found {len(tables)} table(s) in PDF")
    print(f"✓ Extracted {len(raw_text)} characters of text\n")

    if not figures:
        print("⚠️  No figures detected in this PDF")
        return

    # Analyze figures with vision
    print(f"🔍 Analyzing figures with {provider}...")
    print(f"   (Processing {min(len(figures), 3)} figures for demo)\n")

    analyzed_figures = await extract_figures_with_vision(
        pdf_bytes,
        figures,
        max_figures=3,  # Limit to 3 for demo
    )

    # Display results
    for i, fig in enumerate(analyzed_figures, 1):
        print(f"\n{'─' * 80}")
        print(f"Figure {i}")
        print(f"{'─' * 80}")
        print(f"Page: {fig.get('page')}")
        print(f"Index: {fig.get('figure_index')}")

        analysis = fig.get("analysis")
        if not analysis:
            print("⚠️  No analysis available")
            continue

        if analysis.get("error"):
            print(f"❌ Error: {analysis['error']}")
            continue

        print(f"\n📊 Analysis:")
        print(f"  Type: {analysis.get('type', 'unknown')}")

        if analysis.get("title"):
            print(f"  Title: {analysis['title']}")

        if analysis.get("x_axis"):
            print(f"  X-axis: {analysis['x_axis']}")

        if analysis.get("y_axis"):
            print(f"  Y-axis: {analysis['y_axis']}")

        if analysis.get("legend"):
            print(f"  Legend: {', '.join(analysis['legend'])}")

        if analysis.get("data"):
            print(f"\n  📈 Data Points:")
            for dp in analysis["data"][:5]:  # Show first 5
                if isinstance(dp, dict):
                    label = dp.get("label", "")
                    value = dp.get("value", "")
                    print(f"    • {label}: {value}")

        if analysis.get("summary"):
            print(f"\n  💡 Summary:")
            print(f"    {analysis['summary']}")

    # Show formatted text output
    print(f"\n\n{'=' * 80}")
    print("Formatted Text Output (for RAG context)")
    print(f"{'=' * 80}\n")

    figures_text = build_figures_text(analyzed_figures)
    print(figures_text)

    print(f"\n\n{'=' * 80}")
    print("✅ Test Complete!")
    print(f"{'=' * 80}\n")


async def test_single_figure_image(
    pdf_path: str, page: int, bbox=None, provider: str = "groq"
):
    """
    Test extraction and analysis of a single figure image.

    Args:
        pdf_path: Path to PDF file
        page: Page number (1-indexed)
        bbox: Optional bounding box [x0, top, x1, bottom]
        provider: Vision model provider
    """
    print(f"\n{'=' * 80}")
    print(f"Testing Single Figure Extraction")
    print(f"PDF: {pdf_path}")
    print(f"Page: {page}")
    print(f"Provider: {provider}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return

    settings.llm_provider = provider

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"📷 Extracting image from page {page}...")
    image_bytes = _extract_figure_image(pdf_bytes, page, bbox)

    if not image_bytes:
        print("❌ Failed to extract image")
        return

    print(f"✓ Extracted {len(image_bytes)} bytes")

    print(f"\n🔍 Analyzing with vision model...")
    analysis = await analyze_figure_with_vision(image_bytes)

    if analysis.get("error"):
        print(f"❌ Error: {analysis['error']}")
        return

    print("\n📊 Analysis Result:")
    print(f"  Type: {analysis.get('type')}")
    print(f"  Summary: {analysis.get('summary')}")

    if analysis.get("data"):
        print(f"\n  Data: {len(analysis['data'])} points extracted")


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vision-based figure extraction")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "gemini"],
        default="groq",
        help="Vision model provider (default: groq)",
    )
    parser.add_argument(
        "--page", type=int, help="Extract only a specific page (for single figure test)"
    )

    args = parser.parse_args()

    if args.page:
        # Single figure test
        asyncio.run(
            test_single_figure_image(args.pdf_path, args.page, provider=args.provider)
        )
    else:
        # Full PDF test
        asyncio.run(test_figure_extraction(args.pdf_path, provider=args.provider))


if __name__ == "__main__":
    main()
