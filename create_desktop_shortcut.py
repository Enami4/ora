#!/usr/bin/env python3
"""
Create desktop shortcut for JABE Regulatory Processor
"""

import os
import sys
from pathlib import Path


def create_windows_shortcut():
    """Create Windows desktop shortcut"""
    try:
        import win32com.client
        
        # Get paths
        exe_path = Path(__file__).parent / "dist" / "JABE_Regulatory_Processor.exe"
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "JABE Regulatory Processor.lnk"
        
        if not exe_path.exists():
            print(f"❌ Executable not found: {exe_path}")
            print("   Build the executable first using: python build_executable.py")
            return False
        
        # Create shortcut
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.TargetPath = str(exe_path)
        shortcut.WorkingDirectory = str(exe_path.parent)
        shortcut.Description = "JABE Regulatory Document Processor - AI-powered validation and article extraction"
        
        # Set icon if available
        icon_path = Path(__file__).parent / "static" / "img" / "JABE_LOGO_02.ico"
        if icon_path.exists():
            shortcut.IconLocation = str(icon_path)
        
        shortcut.save()
        
        print(f"✅ Desktop shortcut created: {shortcut_path}")
        return True
        
    except ImportError:
        print("❌ pywin32 not installed. Install with: pip install pywin32")
        return False
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        return False


def create_linux_desktop_file():
    """Create Linux .desktop file"""
    try:
        # Get paths
        exe_path = Path(__file__).parent / "dist" / "JABE_Regulatory_Processor"
        desktop = Path.home() / "Desktop"
        desktop_file = desktop / "JABE_Regulatory_Processor.desktop"
        
        if not exe_path.exists():
            print(f"❌ Executable not found: {exe_path}")
            print("   Build the executable first using: python build_executable.py")
            return False
        
        # Desktop file content
        content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=JABE Regulatory Processor
Comment=AI-powered regulatory document validation and article extraction
Exec={exe_path.absolute()}
Icon={Path(__file__).parent / "static" / "img" / "JABE_LOGO_02.png"}
Terminal=false
Categories=Office;
"""
        
        # Write desktop file
        with open(desktop_file, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(desktop_file, 0o755)
        
        print(f"✅ Desktop file created: {desktop_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating desktop file: {e}")
        return False


def main():
    """Main function"""
    print("🖥️ Creating desktop shortcut for JABE Regulatory Processor...")
    
    if sys.platform.startswith('win'):
        success = create_windows_shortcut()
    elif sys.platform.startswith('linux'):
        success = create_linux_desktop_file()
    else:
        print(f"❌ Unsupported platform: {sys.platform}")
        success = False
    
    if success:
        print("\n🎉 Shortcut created successfully!")
        print("   You can now launch the app from your desktop.")
    else:
        print("\n❌ Failed to create shortcut.")
        print("   You can still run the executable directly from the dist/ folder.")


if __name__ == "__main__":
    main()