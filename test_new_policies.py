#!/usr/bin/env python3
"""
Test script to verify that the new Colorado and GDPR policies are properly recognized.
"""

import sys
import os

# Add the current directory to the path so we can import ai_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_pipeline import load_relevancy_map

def test_policy_loading():
    """Test that the new policies can be loaded correctly."""
    
    print("Testing policy loading for new policies...")
    
    # Test Colorado policy
    print("\n1. Testing Colorado policy...")
    try:
        colorado_map = load_relevancy_map("Colorado.txt")
        print(f"   ✓ Colorado policy loaded successfully")
        print(f"   ✓ Found {len(colorado_map)} sections")
        print(f"   ✓ Sample sections: {list(colorado_map.keys())[:3]}")
    except Exception as e:
        print(f"   ✗ Error loading Colorado policy: {e}")
        return False
    
    # Test GDPR policy
    print("\n2. Testing GDPR policy...")
    try:
        gdpr_map = load_relevancy_map("GDPR.txt")
        print(f"   ✓ GDPR policy loaded successfully")
        print(f"   ✓ Found {len(gdpr_map)} sections")
        print(f"   ✓ Sample sections: {list(gdpr_map.keys())[:3]}")
    except Exception as e:
        print(f"   ✗ Error loading GDPR policy: {e}")
        return False
    
    # Test existing policies still work
    print("\n3. Testing existing policies still work...")
    try:
        eu_map = load_relevancy_map("EU.txt")
        ccpa_map = load_relevancy_map("CCPA.txt")
        aida_map = load_relevancy_map("AIDA.txt")
        print(f"   ✓ EU policy loaded successfully")
        print(f"   ✓ CCPA policy loaded successfully")
        print(f"   ✓ AIDA policy loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading existing policies: {e}")
        return False
    
    print("\n🎉 All policy loading tests passed!")
    return True

def test_sample_responses():
    """Test that sample response files exist for the new policies."""
    
    print("\nTesting sample response files...")
    
    # Check Colorado sample responses
    colorado_dir = "sample_responses/Colorado"
    if os.path.exists(colorado_dir):
        colorado_files = os.listdir(colorado_dir)
        print(f"   ✓ Colorado sample responses directory exists with {len(colorado_files)} files")
        print(f"   ✓ Sample files: {colorado_files[:3]}")
    else:
        print(f"   ✗ Colorado sample responses directory not found")
        return False
    
    # Check GDPR sample responses
    gdpr_dir = "sample_responses/GDPR"
    if os.path.exists(gdpr_dir):
        gdpr_files = os.listdir(gdpr_dir)
        print(f"   ✓ GDPR sample responses directory exists with {len(gdpr_files)} files")
        print(f"   ✓ Sample files: {gdpr_files[:3]}")
    else:
        print(f"   ✗ GDPR sample responses directory not found")
        return False
    
    print("\n🎉 All sample response tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Policy Integration: Colorado and GDPR")
    print("=" * 60)
    
    success = True
    
    # Test policy loading
    if not test_policy_loading():
        success = False
    
    # Test sample responses
    if not test_sample_responses():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! New policies are properly integrated.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
