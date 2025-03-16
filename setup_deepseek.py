#!/usr/bin/env python
"""
Setup script for the DeepSeek integration with Volcengine.

This script helps users set up the necessary environment for using the DeepSeek model
via the Volcengine API in the asymmetric learning experiments.
"""

import os
import sys
import argparse
import subprocess
import getpass


def check_openai_package():
    """Check if the OpenAI package is installed."""
    try:
        import openai
        print(f"✅ OpenAI package is installed (version: {openai.__version__})")
        if int(openai.__version__.split('.')[0]) < 1:
            print("⚠️ Warning: OpenAI package version should be 1.0.0 or higher.")
            print("   Please upgrade: pip install --upgrade 'openai>=1.0.0'")
        return True
    except ImportError:
        print("❌ OpenAI package is not installed.")
        print("   Please install it: pip install 'openai>=1.0.0'")
        return False


def setup_api_key(api_key=None):
    """
    Set up the API key for Volcengine.
    
    Args:
        api_key: API key for Volcengine. If None, user will be prompted to enter it.
    
    Returns:
        bool: Whether the API key was set up successfully
    """
    if not api_key:
        print("\n=== Volcengine API Key Setup ===")
        print("To use the DeepSeek model, you need a Volcengine API key.")
        print("You can find your API key in the Volcengine console: https://console.volcengine.com/")
        print("Navigate to: 火山方舟 > 系统管理 > API Key\n")
        
        api_key = getpass.getpass("Enter your Volcengine API key (will not be displayed): ")
        
    if not api_key:
        print("❌ No API key provided. Setup aborted.")
        return False
    
    # Store the API key in environment variables
    os.environ["ARK_API_KEY"] = api_key
    
    # Ask if the user wants to save the API key permanently
    save_key = input("\nDo you want to save the API key to your environment? (y/n): ").lower() == 'y'
    
    if save_key:
        shell = os.environ.get('SHELL', '')
        if 'bash' in shell:
            config_file = os.path.expanduser('~/.bashrc')
        elif 'zsh' in shell:
            config_file = os.path.expanduser('~/.zshrc')
        else:
            print("⚠️ Couldn't determine your shell. Please add the API key to your environment manually:")
            print(f"   export ARK_API_KEY='{api_key}'")
            return True
        
        # Check if the key is already in the config file
        try:
            with open(config_file, 'r') as f:
                if f'ARK_API_KEY' in f.read():
                    update = input(f"API key already exists in {config_file}. Update it? (y/n): ").lower() == 'y'
                    if not update:
                        print("✅ Using existing API key.")
                        return True
        except FileNotFoundError:
            pass
        
        # Add the key to the config file
        try:
            with open(config_file, 'a') as f:
                f.write(f'\n# Volcengine API key for DeepSeek\nexport ARK_API_KEY="{api_key}"\n')
            print(f"✅ API key saved to {config_file}")
            print(f"   Please restart your terminal or run: source {config_file}")
        except Exception as e:
            print(f"❌ Error saving API key: {str(e)}")
            print(f"   Please add the API key to your environment manually:")
            print(f"   export ARK_API_KEY='{api_key}'")
    
    print("\n✅ API key set for this session. You can now run the experiments.")
    return True


def test_volcengine_api():
    """Test the Volcengine API connection."""
    print("\n=== Testing Volcengine API Connection ===")
    api_key = os.environ.get("ARK_API_KEY")
    
    if not api_key:
        print("❌ API key not found in environment variables.")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        # Make a simple completion request
        completion = client.chat.completions.create(
            model="deepseek-r1-250120",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API connection successful' if you can see this message."}
            ],
            max_tokens=20
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"Response from API: '{response}'")
        
        if "successful" in response.lower():
            print("✅ Successfully connected to Volcengine API!")
            return True
        else:
            print("⚠️ Connected to API but received unexpected response.")
            return True
    
    except Exception as e:
        print(f"❌ Error connecting to Volcengine API: {str(e)}")
        return False


def install_requirements():
    """Install the required packages."""
    print("\n=== Installing Required Packages ===")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed requirements.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up the DeepSeek integration with Volcengine")
    parser.add_argument("--api-key", help="Volcengine API key")
    parser.add_argument("--skip-test", action="store_true", help="Skip API connection test")
    parser.add_argument("--skip-install", action="store_true", help="Skip requirements installation")
    args = parser.parse_args()
    
    print("=== DeepSeek Setup for Asymmetric Learning ===\n")
    
    if not args.skip_install:
        install_requirements()
    
    check_openai_package()
    
    if setup_api_key(args.api_key) and not args.skip_test:
        test_volcengine_api()
    
    print("\n=== Setup Complete ===")
    print("You can now run the experiments with the DeepSeek model:")
    print("python src/main.py --task=1 --use-deepseek")


if __name__ == "__main__":
    main() 