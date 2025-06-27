#!/usr/bin/env python3
"""Test script to find Tesseract installation."""

import os
import subprocess

def test_tesseract_paths():
    """Test different Tesseract paths."""
    paths_to_test = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\doupa\AppData\Local\Tesseract-OCR\tesseract.exe",
        "tesseract",  # System PATH
    ]
    
    print("=== Testing Tesseract Paths ===")
    
    for path in paths_to_test:
        print(f"Testing: {path}")
        
        if path == "tesseract":
            # Test system PATH
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ FOUND in system PATH")
                    print(f"Version: {result.stdout.split()[1] if len(result.stdout.split()) > 1 else 'Unknown'}")
                    return path
                else:
                    print(f"✗ Not in system PATH")
            except FileNotFoundError:
                print(f"✗ Not in system PATH")
        else:
            # Test specific path
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✓ FOUND at {path}")
                        print(f"Version: {result.stdout.split()[1] if len(result.stdout.split()) > 1 else 'Unknown'}")
                        return path
                    else:
                        print(f"✗ Found file but execution failed")
                except Exception as e:
                    print(f"✗ Found file but error: {e}")
            else:
                print(f"✗ File does not exist")
    
    print("\n=== Searching in common directories ===")
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"C:\Users\doupa\AppData\Local"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if "tesseract.exe" in files:
                    full_path = os.path.join(root, "tesseract.exe")
                    print(f"Found Tesseract at: {full_path}")
                    try:
                        result = subprocess.run([full_path, "--version"], capture_output=True, text=True)
                        if result.returncode == 0:
                            print(f"✓ WORKING: {full_path}")
                            return full_path
                    except:
                        pass
    
    return None

if __name__ == "__main__":
    found_path = test_tesseract_paths()
    if found_path:
        print(f"\n🎉 Use this path: {found_path}")
    else:
        print(f"\n❌ Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")