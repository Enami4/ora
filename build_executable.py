#!/usr/bin/env python3
"""
Build script for creating JABE Regulatory Processor executable
"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


class JABEExecutableBuilder:
    """Builder class for creating JABE executable"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.dist_dir = self.base_dir / "dist"
        self.build_dir = self.base_dir / "build"
        self.output_dir = self.base_dir / "releases"
        
    def clean_previous_builds(self):
        """Clean up previous build artifacts"""
        print("🧹 Cleaning previous builds...")
        
        # Remove dist and build directories
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
            print(f"   ✓ Removed {self.dist_dir}")
            
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            print(f"   ✓ Removed {self.build_dir}")
    
    def check_requirements(self):
        """Check if all requirements are installed"""
        print("📋 Checking requirements...")
        
        required_packages = [
            'pyinstaller',
            'PyQt5',
            'pandas',
            'openpyxl',
            'anthropic',
            'PyPDF2',
            'pdfplumber',
            'nltk'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"   ✗ {package} (missing)")
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("   Run: pip install -r requirements.txt")
            return False
        
        return True
    
    def check_resources(self):
        """Check if required resources exist"""
        print("🖼️ Checking resources...")
        
        logo_path = self.base_dir / "static" / "img" / "JABE_LOGO_02.png"
        if logo_path.exists():
            print(f"   ✓ Logo found: {logo_path}")
        else:
            print(f"   ⚠️  Logo not found: {logo_path}")
            print("      The app will work but without logo")
        
        # Check main script
        main_script = self.base_dir / "regulatory_gui_branded.py"
        if main_script.exists():
            print(f"   ✓ Main script: {main_script}")
            return True
        else:
            print(f"   ❌ Main script not found: {main_script}")
            return False
    
    def create_icon_file(self):
        """Convert PNG logo to ICO format for Windows executable"""
        print("🎨 Creating icon file...")
        
        logo_path = self.base_dir / "static" / "img" / "JABE_LOGO_02.png"
        ico_path = self.base_dir / "static" / "img" / "JABE_LOGO_02.ico"
        
        if logo_path.exists():
            try:
                from PIL import Image
                
                # Open PNG and convert to ICO
                img = Image.open(logo_path)
                # Resize to common icon sizes
                icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                img.save(ico_path, format='ICO', sizes=icon_sizes)
                print(f"   ✓ Icon created: {ico_path}")
                return str(ico_path)
                
            except ImportError:
                print("   ⚠️  PIL/Pillow not installed, using PNG as icon")
                return str(logo_path)
            except Exception as e:
                print(f"   ⚠️  Could not create ICO: {e}")
                return str(logo_path)
        else:
            print("   ⚠️  No logo found, using default icon")
            return None
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        print("📚 Downloading NLTK data...")
        
        try:
            import nltk
            
            # Download required NLTK data
            nltk_data = ['punkt', 'punkt_tab']
            for data in nltk_data:
                try:
                    nltk.download(data, quiet=True)
                    print(f"   ✓ Downloaded {data}")
                except:
                    print(f"   ⚠️  Could not download {data}")
                    
        except ImportError:
            print("   ⚠️  NLTK not available")
    
    def build_executable(self):
        """Build the executable using PyInstaller"""
        print("🔨 Building executable...")
        
        # Create icon
        icon_path = self.create_icon_file()
        
        # PyInstaller command
        cmd = [
            'pyinstaller',
            '--clean',
            '--onefile',
            '--windowed',  # No console window
            '--name', 'JABE_Regulatory_Processor',
            '--distpath', str(self.dist_dir),
            '--workpath', str(self.build_dir),
        ]
        
        # Add icon if available
        if icon_path:
            cmd.extend(['--icon', icon_path])
        
        # Add data files
        cmd.extend([
            '--add-data', f'regulatory_processor{os.pathsep}regulatory_processor',
            '--add-data', f'static{os.pathsep}static',
        ])
        
        # Add hidden imports
        hidden_imports = [
            'regulatory_processor',
            'PyQt5.QtCore',
            'PyQt5.QtGui',
            'PyQt5.QtWidgets',
            'pandas',
            'openpyxl',
            'anthropic',
            'PyPDF2',
            'pdfplumber',
            'nltk',
            'nltk.tokenize',
        ]
        
        for imp in hidden_imports:
            cmd.extend(['--hidden-import', imp])
        
        # Add main script
        cmd.append('regulatory_gui_branded.py')
        
        print(f"   Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ Build successful!")
                return True
            else:
                print("   ❌ Build failed!")
                print(f"   Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Build error: {e}")
            return False
    
    def create_distribution(self):
        """Create distribution package"""
        print("📦 Creating distribution package...")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Find the executable
        exe_path = self.dist_dir / "JABE_Regulatory_Processor.exe"
        if not exe_path.exists():
            print(f"   ❌ Executable not found: {exe_path}")
            return False
        
        # Create distribution folder
        dist_name = f"JABE_Regulatory_Processor_v2.0_{timestamp}"
        dist_folder = self.output_dir / dist_name
        dist_folder.mkdir(exist_ok=True)
        
        # Copy executable
        shutil.copy2(exe_path, dist_folder / "JABE_Regulatory_Processor.exe")
        
        # Create README for distribution
        readme_content = f"""# JABE Regulatory Document Processor v2.0

## Installation
1. Extract this folder to your desired location
2. Double-click JABE_Regulatory_Processor.exe to run

## Features
- AI-powered document validation
- Regulatory article extraction
- Materiality assessment
- Excel export with user information
- French interface with JABE branding

## System Requirements
- Windows 10/11 (64-bit)
- 4GB RAM minimum
- 500MB free disk space

## Usage
1. Launch the application
2. Enter your name and surname
3. Select PDF documents or folders
4. Configure processing settings
5. Start processing and download results

## Support
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: 2.0.0
"""
        
        with open(dist_folder / "README.txt", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create ZIP archive
        zip_path = self.output_dir / f"{dist_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dist_folder.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(dist_folder))
        
        print(f"   ✅ Distribution created:")
        print(f"      📁 Folder: {dist_folder}")
        print(f"      📦 Archive: {zip_path}")
        
        # Print file sizes
        exe_size = exe_path.stat().st_size / (1024 * 1024)
        zip_size = zip_path.stat().st_size / (1024 * 1024)
        
        print(f"   📊 Executable size: {exe_size:.1f} MB")
        print(f"   📊 Archive size: {zip_size:.1f} MB")
        
        return True
    
    def run_tests(self):
        """Test the built executable"""
        print("🧪 Testing executable...")
        
        exe_path = self.dist_dir / "JABE_Regulatory_Processor.exe"
        if exe_path.exists():
            print(f"   ✅ Executable exists: {exe_path}")
            print(f"   📊 Size: {exe_path.stat().st_size / (1024 * 1024):.1f} MB")
            
            # Try to run a quick test (optional)
            print("   ℹ️  Manual testing recommended:")
            print("      - Launch the executable")
            print("      - Check GUI loads properly") 
            print("      - Test basic functionality")
            
            return True
        else:
            print("   ❌ Executable not found")
            return False
    
    def build_all(self):
        """Run the complete build process"""
        print("🚀 JABE Regulatory Processor - Build Process")
        print("=" * 60)
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Checking resources", self.check_resources),
            ("Cleaning previous builds", self.clean_previous_builds),
            ("Downloading NLTK data", self.download_nltk_data),
            ("Building executable", self.build_executable),
            ("Testing executable", self.run_tests),
            ("Creating distribution", self.create_distribution),
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Build failed at step: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Executable location: {self.dist_dir}")
        print(f"📦 Distribution packages: {self.output_dir}")
        print("\n💡 To run the app, double-click: JABE_Regulatory_Processor.exe")
        
        return True


def main():
    """Main function"""
    builder = JABEExecutableBuilder()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            builder.clean_previous_builds()
        elif sys.argv[1] == "--test":
            builder.run_tests()
        elif sys.argv[1] == "--dist":
            builder.create_distribution()
        else:
            print("Usage:")
            print("  python build_executable.py          # Full build")
            print("  python build_executable.py --clean  # Clean only")
            print("  python build_executable.py --test   # Test only")
            print("  python build_executable.py --dist   # Create distribution only")
    else:
        builder.build_all()


if __name__ == "__main__":
    main()