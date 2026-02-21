#!/bin/bash

echo "🚀 Vertex AI Setup Script for Video Analytics"
echo "=============================================="
echo ""

# Check if running in the correct directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Check for gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "⚠️  Warning: gcloud CLI not found"
    echo "   Install from: https://cloud.google.com/sdk/docs/install"
    echo ""
else
    echo "✅ gcloud CLI found"
fi

# Check for environment variables
echo "🔍 Checking environment variables..."
echo ""

if [ -z "$VERTEX_AI_PROJECT_ID" ]; then
    echo "⚠️  VERTEX_AI_PROJECT_ID not set"
    echo "   Set it with: export VERTEX_AI_PROJECT_ID='your-project-id'"
else
    echo "✅ VERTEX_AI_PROJECT_ID: $VERTEX_AI_PROJECT_ID"
fi

if [ -z "$VERTEX_AI_LOCATION" ]; then
    echo "⚠️  VERTEX_AI_LOCATION not set (will default to us-central1)"
    echo "   Set it with: export VERTEX_AI_LOCATION='us-central1'"
else
    echo "✅ VERTEX_AI_LOCATION: $VERTEX_AI_LOCATION"
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "   Set it with: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'"
else
    echo "✅ GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"
fi

echo ""
echo "📋 Next Steps:"
echo "=============================================="
echo "1. Set environment variables if not already set:"
echo "   export VERTEX_AI_PROJECT_ID='your-project-id'"
echo "   export VERTEX_AI_LOCATION='us-central1'"
echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'"
echo ""
echo "2. Enable Vertex AI API:"
echo "   gcloud services enable aiplatform.googleapis.com"
echo ""
echo "3. Test the setup:"
echo "   python test_vertex_setup.py"
echo ""
echo "4. Run the main pipeline:"
echo "   python main.py videos/sumka_512x512_2fps \"Find when the bag is picked up\""
echo ""
