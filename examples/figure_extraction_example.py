#!/usr/bin/env python3
"""
Example: Using Vision-Based Figure Extraction

This example demonstrates how to use the figure extraction feature
both as a standalone tool and integrated with PaperMind's RAG pipeline.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.figure_extractor import (
    build_figures_text,
    extract_figures_with_vision,
)
from app.services.text_extractor import extract_pdf_content


async def example_1_basic_extraction():
    """Example 1: Basic figure extraction from a PDF."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Figure Extraction")
    print("=" * 80 + "\n")

    # Configure
    settings.llm_provider = "groq"  # or "openai" or "gemini"

    # Load PDF
    pdf_path = "sample_paper.pdf"  # Replace with your PDF
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please provide a PDF file at: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Extract figures using pdfplumber
    print("Step 1: Detecting figures...")
    _, _, figures = extract_pdf_content(pdf_bytes)
    print(f"✓ Found {len(figures)} figure(s)\n")

    if not figures:
        print("No figures detected in this PDF")
        return

    # Analyze with vision model
    print("Step 2: Analyzing with vision model...")
    analyzed_figures = await extract_figures_with_vision(
        pdf_bytes,
        figures,
        max_figures=3,  # Limit for demo
    )

    # Display results
    print("\nStep 3: Results\n")
    for i, fig in enumerate(analyzed_figures, 1):
        analysis = fig.get("analysis", {})

        print(f"Figure {i} (Page {fig.get('page')}):")
        print(f"  Type: {analysis.get('type', 'unknown')}")

        if analysis.get("title"):
            print(f"  Title: {analysis['title']}")

        if analysis.get("summary"):
            print(f"  Summary: {analysis['summary']}")

        print()


async def example_2_structured_data():
    """Example 2: Extracting structured data from charts."""
    print("\n" + "=" * 80)
    print("Example 2: Extracting Structured Data")
    print("=" * 80 + "\n")

    pdf_path = "sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please provide a PDF file at: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    _, _, figures = extract_pdf_content(pdf_bytes)

    if not figures:
        return

    analyzed = await extract_figures_with_vision(pdf_bytes, figures, max_figures=5)

    # Filter for data-containing figures
    data_figures = [
        f
        for f in analyzed
        if f.get("analysis", {}).get("type") not in ["non-data", "unknown"]
    ]

    print(f"Found {len(data_figures)} data figure(s)\n")

    for fig in data_figures:
        analysis = fig["analysis"]

        print(f"\n{analysis.get('title', 'Untitled Figure')}:")
        print("-" * 60)

        # Extract data points
        data_points = analysis.get("data", [])
        if data_points:
            print("\nData Points:")
            for dp in data_points[:10]:  # Show first 10
                if isinstance(dp, dict):
                    label = dp.get("label", "")
                    value = dp.get("value", "")
                    print(f"  • {label}: {value}")

        # Extract axis information
        x_axis = analysis.get("x_axis")
        y_axis = analysis.get("y_axis")
        if x_axis or y_axis:
            print("\nAxes:")
            if x_axis:
                print(f"  X: {x_axis}")
            if y_axis:
                print(f"  Y: {y_axis}")

        print()


async def example_3_rag_integration():
    """Example 3: Figure data formatted for RAG context."""
    print("\n" + "=" * 80)
    print("Example 3: RAG Integration Format")
    print("=" * 80 + "\n")

    pdf_path = "sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please provide a PDF file at: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    _, _, figures = extract_pdf_content(pdf_bytes)

    if not figures:
        return

    analyzed = await extract_figures_with_vision(pdf_bytes, figures)

    # Format for RAG
    figures_text = build_figures_text(analyzed)

    print("This text would be added to the document context:\n")
    print("=" * 80)
    print(figures_text)
    print("=" * 80)
    print("\nThis formatted text is automatically:")
    print("  1. Appended to the relevant section (Results, Methods, etc.)")
    print("  2. Included in the chunking process")
    print("  3. Embedded and indexed in ChromaDB")
    print("  4. Retrieved during Q&A queries")


async def example_4_provider_comparison():
    """Example 4: Compare different vision providers."""
    print("\n" + "=" * 80)
    print("Example 4: Provider Comparison")
    print("=" * 80 + "\n")

    pdf_path = "sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Please provide a PDF file at: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    _, _, figures = extract_pdf_content(pdf_bytes)

    if not figures:
        return

    # Test first figure with different providers
    figure = figures[0]
    providers = []

    # Check which providers are available
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")

    if not providers:
        print("⚠️  No API keys found. Please set at least one:")
        print("  - GROQ_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - GEMINI_API_KEY")
        return

    print(f"Comparing providers on Figure 1 (page {figure.get('page')})...\n")

    for provider in providers:
        print(f"\n{'─' * 60}")
        print(f"Testing: {provider.upper()}")
        print(f"{'─' * 60}")

        settings.llm_provider = provider

        import time

        start = time.time()

        result = await extract_figures_with_vision(pdf_bytes, [figure], max_figures=1)

        elapsed = time.time() - start

        if result and result[0].get("analysis"):
            analysis = result[0]["analysis"]

            print(f"⏱️  Time: {elapsed:.2f}s")
            print(f"📊 Type: {analysis.get('type', 'unknown')}")
            print(f"📝 Summary: {analysis.get('summary', 'N/A')[:100]}...")

            if analysis.get("data"):
                print(f"📈 Data points: {len(analysis['data'])}")
        else:
            print(f"❌ Failed")


async def main():
    """Run all examples."""
    print("\n" + "🔬 PaperMind Figure Extraction Examples " + "\n")
    print("Make sure you have:")
    print("  1. A sample PDF file named 'sample_paper.pdf'")
    print("  2. At least one API key set (GROQ_API_KEY recommended)")
    print()

    # Check for API keys
    has_key = any(
        [
            os.getenv("GROQ_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
        ]
    )

    if not has_key:
        print("⚠️  No API keys found!")
        print("\nPlease set one of:")
        print("  export GROQ_API_KEY='your_key'      # Recommended - free tier")
        print("  export OPENAI_API_KEY='your_key'    # Best accuracy")
        print("  export GEMINI_API_KEY='your_key'    # Good balance")
        return

    try:
        await example_1_basic_extraction()
        await example_2_structured_data()
        await example_3_rag_integration()
        await example_4_provider_comparison()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80 + "\n")

    except FileNotFoundError:
        print("\n⚠️  Please create a 'sample_paper.pdf' file to run these examples")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
