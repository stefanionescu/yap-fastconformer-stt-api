#!/usr/bin/env python3
"""Test script to verify Sherpa-ONNX online punctuation functionality."""

from pathlib import Path
import sys
import logging

# Add server module to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from server.settings import Settings
from server.ws_server import VoskServer

def test_punctuation():
    """Test the punctuation functionality through VoskServer wrapper."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Create settings - you may need to adjust paths based on your container setup
    settings = Settings.from_env()
    
    print(f"Testing punctuation with model dir: {settings.punct_dir}")
    print(f"Expected model file: {settings.punct_dir / 'model.onnx'}")
    print(f"Expected vocab file: {settings.punct_dir / 'bpe.vocab'}")
    
    # Test if files exist
    model_file = settings.punct_dir / "model.onnx"
    vocab_file = settings.punct_dir / "bpe.vocab"
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
        
    if not vocab_file.exists():
        print(f"❌ Vocab file not found: {vocab_file}")
        return False
        
    print("✅ Required files found")
    
    # Build punctuator
    try:
        punctuator = VoskServer._build_punctuator(settings)
    except RuntimeError as e:
        print(f"❌ RuntimeError during punctuator building: {e}")
        return False
    
    if punctuator is None:
        print("❌ Failed to build punctuator")
        return False
        
    print("✅ Punctuator built successfully")
    
    # Test punctuation
    test_text = "how are you i am fine thank you"
    expected_result = "How are you? I am fine. Thank you."
    
    try:
        result = punctuator(test_text)
        print(f"Input:  '{test_text}'")
        print(f"Output: '{result}'")
        print(f"Expected: '{expected_result}'")
        
        # Check if result has capitalization and punctuation
        has_capitalization = result[0].isupper() if result else False
        has_punctuation = any(char in result for char in '.?!')
        
        print(f"✅ Has capitalization: {has_capitalization}")
        print(f"✅ Has punctuation: {has_punctuation}")
        
        if has_capitalization and has_punctuation:
            print("🎉 VoskServer punctuation test PASSED!")
            return True
        else:
            print("❌ VoskServer punctuation test FAILED - missing capitalization or punctuation")
            return False
            
    except Exception as e:
        print(f"❌ Punctuation failed with error: {e}")
        return False

def test_direct_sherpa():
    """Test Sherpa-ONNX directly without our wrapper (smoke test from the user)."""
    try:
        import sherpa_onnx as so
        print("✅ sherpa_onnx imported successfully")
        
        # Test with exact paths from user's smoke test
        d = Path("/models/punct/sherpa-onnx-online-punct-en-2024-08-06")
        
        if not d.exists():
            print(f"❌ Direct test: Model directory not found at {d}")
            return False
            
        if not (d / "model.onnx").exists():
            print(f"❌ Direct test: Model file not found at {d / 'model.onnx'}")
            return False
            
        if not (d / "bpe.vocab").exists():
            print(f"❌ Direct test: Vocab file not found at {d / 'bpe.vocab'}")
            return False
        
        cfg = so.OnlinePunctuationConfig(
            model_config=so.OnlinePunctuationModelConfig(
                cnn_bilstm=str(d / "model.onnx"),
                bpe_vocab=str(d / "bpe.vocab"),
                provider="cpu",
                num_threads=1,
                debug=False,
            )
        )
        p = so.OnlinePunctuation(cfg)
        
        # Test the CORRECT method name
        if not hasattr(p, "add_punctuation_with_case"):
            print("❌ add_punctuation_with_case method not found!")
            return False
            
        test_text = "how are you i am fine thank you"
        result = p.add_punctuation_with_case(test_text)
        expected = "How are you? I am fine. Thank you."
        
        print(f"Direct sherpa input:  '{test_text}'")
        print(f"Direct sherpa output: '{result}'")
        print(f"Expected:            '{expected}'")
        
        # Basic validation
        if result.startswith("How") and ("?" in result or "." in result):
            print("✅ Direct sherpa test PASSED!")
            return True
        else:
            print("❌ Direct sherpa test FAILED - unexpected output")
            return False
        
    except ImportError:
        print("❌ sherpa_onnx not available for direct test")
        return False
    except Exception as e:
        print(f"❌ Direct sherpa test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Sherpa-ONNX Online Punctuation (add_punctuation_with_case) ===\n")
    
    print("1. Testing direct sherpa_onnx usage:")
    direct_success = test_direct_sherpa()
    
    print("\n2. Testing through VoskServer wrapper:")
    wrapper_success = test_punctuation()
    
    print("\n=== Summary ===")
    if direct_success and wrapper_success:
        print("🎉 All tests PASSED!")
        sys.exit(0)
    elif direct_success:
        print("✅ Direct sherpa test passed, ❌ VoskServer wrapper failed")
        sys.exit(1)
    elif wrapper_success:
        print("❌ Direct sherpa test failed, ✅ VoskServer wrapper passed")
        sys.exit(1)
    else:
        print("❌ All tests FAILED!")
        sys.exit(1)
