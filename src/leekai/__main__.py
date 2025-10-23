import sys
import traceback # Import traceback

def main():
    """
    Main entrypoint for the package.
    Detects if the user wants 'gui' mode or CLI mode.
    """
    # Removed DEBUG print for sys.argv

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'gui': # Use .lower() for robustness
        print("Starting GUI mode...") # This initial print might still go to console
        # Removed DEBUG print before import
        try:
            from .GUI import main_gui
            # Removed DEBUG print after import
            main_gui()
            # Removed DEBUG print after main_gui returns
        except ImportError as e:
            # (Error handling remains the same)
            # Check specifically for tkinter import error first
            if "tkinter" in str(e).lower() or "'_tkinter'" in str(e): # More robust check
                # Print directly to original stderr since GUI might not be working
                print("---", file=sys.__stderr__)
                print("Error: Could not import tkinter.", file=sys.__stderr__)
                print("tkinter is required to run the GUI.", file=sys.__stderr__)
                print("Please ensure it's installed for your Python environment.", file=sys.__stderr__)
                print("(e.g., on Debian/Ubuntu: `sudo apt-get install python3-tk`)", file=sys.__stderr__)
                print("(e.g., on Fedora: `sudo dnf install python3-tkinter`)", file=sys.__stderr__)
                print("(On Windows/macOS, it's usually included with Python)", file=sys.__stderr__)
                print("---", file=sys.__stderr__)
            # Check for the specific GUI module import error
            elif "GUI" in str(e) or "gui" in str(e).lower(): # Check both cases
                 print("---", file=sys.__stderr__)
                 print(f"Error: Could not find/import the GUI module.", file=sys.__stderr__)
                 print(f"Please ensure 'GUI.py' exists in the 'src/leekai/' directory.", file=sys.__stderr__)
                 print(f"(Details: {e})", file=sys.__stderr__)
                 print("---", file=sys.__stderr__)
            else:
                # General import error
                print(f"Error: A required module could not be imported.", file=sys.__stderr__)
                print(f"(Details: {e})", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__) # Print traceback for other import errors
            sys.exit(1)
        except Exception as e:
            # Print GUI loading errors directly to original stderr
            print(f"An unexpected error occurred while loading or running the GUI: {e}", file=sys.__stderr__)
            traceback.print_exc(file=sys.__stderr__)
            sys.exit(1)
    else:
        # (CLI part remains the same)
        # Default to CLI mode
        try:
            # Removed DEBUG prints for CLI
            from .cli import main as cli_main
            cli_main()
            # Removed DEBUG prints for CLI
        except ImportError as e:
            print(f"Error starting CLI: {e}", file=sys.stderr)
            print("Please ensure numpy, numba, pillow, and scipy are installed.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred in CLI mode: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()