import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import threading
import queue
import numpy as np
import time
import json # For loading arch defs
import traceback # Import traceback

# --- Imports from your project ---
# Removed DEBUG prints for imports
try:
    from . import layers as layers_numpy
    from .model import CNNModel # Now dynamic
    from .train import train
    from .data_loader import load_data_from_directory, read_image
    from .architecture_builder import ArchitectureBuilder # Import the new builder GUI
except ImportError as e:
    # This error should ideally be caught by __main__.py, but good to have safety
    print(f"CRITICAL GUI Import Error: {e}", file=sys.__stderr__)
    traceback.print_exc(file=sys.__stderr__)
    # Attempt to show messagebox even if Tk isn't fully working
    try:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Startup Error", f"Failed to import core modules:\n{e}")
    except: pass
    sys.exit(1)


# --- Thread-safe stdout redirector ---
# (QueueIO class remains the same)
class QueueIO(queue.Queue):
    """A file-like object that writes to a queue."""
    def write(self, msg):
        self.put(msg)

    def flush(self):
        # The train function might call flush, needs to exist
        pass

class App(tk.Tk):
    def __init__(self):
        # Removed DEBUG print
        super().__init__()
        self.title("Dynamic CNN Trainer")
        self.geometry("850x750")
        self.minsize(600, 500)

        # --- Style ---
        # (Style setup remains the same)
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TLabel", font=("Inter", 10))
        self.style.configure("TButton", font=("Inter", 10))
        self.style.configure("TEntry", font=("Inter", 10))
        self.style.configure("TCombobox", font=("Inter", 10))
        self.style.configure("TCheckbutton", font=("Inter", 10))
        self.style.configure("TLabelframe.Label", font=("Inter", 11, "bold"))


        # --- Class variables ---
        # (Variables remain the same)
        self.train_thread = None
        self.model = None
        self.current_arch_def = None
        self.current_arch_name = tk.StringVar(value='SimpleNet')
        self.X_train, self.y_train, self.class_names = None, None, []
        self.data_loaded = False


        # --- PanedWindow layout ---
        # (Layout remains the same)
        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = ttk.Frame(paned_window, padding=10, width=350)
        left_frame.pack_propagate(False)
        paned_window.add(left_frame, weight=0)
        right_frame = ttk.Frame(paned_window, padding=10, width=500)
        paned_window.add(right_frame, weight=1)


        # --- Create Controls & Log ---
        # Removed DEBUG prints
        self.log_queue = QueueIO() # Create the queue FIRST
        self.create_controls(left_frame)
        self.create_log(right_frame)

        # --- Redirect stdout AND stderr ---
        print("DEBUG: App.__init__ - Redirecting stdout/stderr...", file=sys.__stderr__) # Keep one debug msg
        # Store original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Redirect
        sys.stdout = self.log_queue
        sys.stderr = self.log_queue # Redirect stderr too

        # Removed DEBUG prints
        self.after(100, self.check_log_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.set_ui_state(False)
        # Removed DEBUG print

    def on_closing(self):
        # Restore stdout/stderr reliably
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        print("DEBUG: Restored stdout/stderr.", file=sys.__stderr__) # Keep for confirmation
        self.destroy()

    def create_controls(self, parent):
        """Creates all the buttons, labels, and entry fields."""
        parent.columnconfigure(0, weight=1)

        # --- 1. Data & Model Paths ---
        data_group = ttk.LabelFrame(parent, text="1. Data & Model", padding=10)
        data_group.grid(row=0, column=0, sticky="ew", pady=5)
        data_group.columnconfigure(1, weight=1)

        self.data_dir = tk.StringVar()
        self.model_save_path = tk.StringVar() # Saves PARAMS (.npz)
        self.model_load_path = tk.StringVar() # Loads PARAMS (.npz)
        self.arch_load_path = tk.StringVar() # Loads ARCH DEF (.json)

        self.create_path_entry(data_group, "Data Dir:", self.data_dir, "data_dir", 0)
        self.load_data_button = ttk.Button(data_group, text="Load Data", command=self.load_data_thread)
        self.load_data_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)

        self.create_path_entry(data_group, "Load Arch:", self.arch_load_path, "load_arch", 2)
        self.load_arch_button = ttk.Button(data_group, text="Load Arch (.json)", command=self.load_arch_def_from_path)
        self.load_arch_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0,5), padx=5)

        self.create_path_entry(data_group, "Load Params:", self.model_load_path, "load_model", 4)
        self.load_params_button = ttk.Button(data_group, text="Load Params (.npz)", command=self.load_model_params_from_path)
        self.load_params_button.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0,5), padx=5)

        self.create_path_entry(data_group, "Save Params:", self.model_save_path, "save_model", 6)


        # --- 2. Architecture & Training ---
        train_group = ttk.LabelFrame(parent, text="2. Training Config", padding=10)
        train_group.grid(row=1, column=0, sticky="ew", pady=5)
        train_group.columnconfigure(1, weight=1)

        # --- Training Hyperparameters ---
        self.jit_var = tk.BooleanVar(value=True)
        self.epochs_var = tk.StringVar(value="10")
        self.lr_var = tk.StringVar(value="0.01")
        self.batch_var = tk.StringVar(value="32")
        self.clip_var = tk.StringVar(value="1.0")
        self.decay_var = tk.StringVar(value="0.95")
        self.create_param_entry(train_group, "Epochs:", self.epochs_var, 0)
        self.create_param_entry(train_group, "Learn Rate:", self.lr_var, 1)
        self.create_param_entry(train_group, "Batch Size:", self.batch_var, 2)
        self.create_param_entry(train_group, "Grad Clip:", self.clip_var, 3)
        self.create_param_entry(train_group, "LR Decay:", self.decay_var, 4)


        # --- Architecture Selection ---
        ttk.Label(train_group, text="Architecture:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.arch_combobox = ttk.Combobox(
            train_group,
            textvariable=self.current_arch_name,
            values=['MicroNet', 'SimpleNet', 'LeNet-5', 'Custom'], # Add Custom
            state="readonly",
        )
        self.arch_combobox.grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.arch_combobox.bind("<<ComboboxSelected>>", self.on_arch_select)

        # --- Button to open builder ---
        self.build_arch_button = ttk.Button(train_group, text="Define Custom Arch...", command=self.open_arch_builder)
        self.build_arch_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5, padx=5)

        # JIT Checkbox
        jit_check = ttk.Checkbutton(train_group, text="Enable JIT Compilation (Fast)", variable=self.jit_var)
        jit_check.grid(row=7, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        # Train Button
        self.train_button = ttk.Button(train_group, text="Start Training", command=self.start_training_thread)
        self.train_button.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 5), padx=5)


        # --- 3. Prediction ---
        pred_group = ttk.LabelFrame(parent, text="3. Prediction", padding=10)
        pred_group.grid(row=2, column=0, sticky="ew", pady=5)
        pred_group.columnconfigure(1, weight=1)

        self.image_path = tk.StringVar()
        self.create_path_entry(pred_group, "Image:", self.image_path, "image", 0)

        self.predict_button = ttk.Button(pred_group, text="Run Prediction", command=self.run_prediction)
        self.predict_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Status: Ready. Load data or define architecture.")
        status_label = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_label.grid(row=3, column=0, sticky="ew", pady=(10,0))


    def create_log(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Menlo", 9) if sys.platform == "darwin" else ("Monospace", 9)
        self.log_text = tk.Text(parent, wrap=tk.WORD, height=10, font=log_font, state=tk.DISABLED, bg="#2b2b2b", fg="#a9b7c6", insertbackground="white")
        scrollbar_y = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

    # --- Helper UI Functions ---
    def create_path_entry(self, parent, label_text, string_var, mode, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=string_var)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        browse_cmd = lambda: self.browse_path(string_var, mode)
        button = ttk.Button(parent, text="...", width=3, command=browse_cmd)
        button.grid(row=row, column=2, sticky="e", padx=5, pady=5)

    def create_param_entry(self, parent, label_text, string_var, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=string_var, width=10)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)


    def browse_path(self, string_var, mode):
        # (Function remains the same)
        path = ""
        if mode == "data_dir":
            path = filedialog.askdirectory(title="Select Data Directory")
        elif mode == "save_model": # Save PARAMS
            path = filedialog.asksaveasfilename(title="Save Model Parameters", defaultextension=".npz", filetypes=[("NumPy Parameters", "*.npz")])
        elif mode == "load_model": # Load PARAMS
            path = filedialog.askopenfilename(title="Load Model Parameters", filetypes=[("NumPy Parameters", "*.npz")])
        elif mode == "load_arch": # Load ARCH DEF
            path = filedialog.askopenfilename(title="Load Architecture Definition", filetypes=[("JSON Arch Def", "*.json")])
        elif mode == "image":
            path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

        if path:
            string_var.set(path)
            if mode == "load_arch":
                self.load_arch_def_from_path(filepath=path)


    def check_log_queue(self):
        # (Function remains the same)
        messages_processed = 0
        max_messages_per_cycle = 50

        while not self.log_queue.empty() and messages_processed < max_messages_per_cycle:
            try:
                msg = self.log_queue.get(block=False)
                messages_processed += 1
                self.log_text.config(state=tk.NORMAL)
                if "\r" in msg:
                    last_line_start = self.log_text.index("end-1c linestart")
                    current_end = self.log_text.index("end-1c")
                    if self.log_text.get(current_end) != '\n':
                         delete_start = last_line_start
                    else:
                         delete_start = self.log_text.index(f"{last_line_start}+1c")
                         if delete_start == "1.1" and last_line_start == "1.0": delete_start = "1.0"

                    self.log_text.delete(delete_start, "end")
                    processed_msg = msg.split('\r')[-1]
                    if processed_msg: self.log_text.insert(tk.END, processed_msg + "\n")
                else:
                    self.log_text.insert(tk.END, msg)

                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
            except queue.Empty: break
            except tk.TclError as e:
                print(f"Tkinter error processing log queue: {e}", file=sys.__stderr__)
                break
            except Exception as e:
                print(f"Error processing log queue: {e}", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
        self.after(100, self.check_log_queue)


    def set_ui_state(self, is_running):
        # (Function remains the same)
        train_ready = self.data_loaded and (self.current_arch_name.get() != 'Custom' or self.current_arch_def is not None)
        predict_ready = self.model is not None
        train_state = tk.DISABLED if is_running else (tk.NORMAL if train_ready else tk.DISABLED)
        predict_state = tk.DISABLED if is_running else (tk.NORMAL if predict_ready else tk.DISABLED)
        general_state = tk.DISABLED if is_running else tk.NORMAL

        if is_running:
            self.train_button.config(text="Training...", state=tk.DISABLED)
            self.status_var.set("Status: Training...")
        else:
            self.train_button.config(text="Start Training", state=train_state)
            if not self.data_loaded: status_msg = "Status: Ready. Load data to begin."
            elif self.current_arch_name.get() == 'Custom' and not self.current_arch_def: status_msg = "Status: Data loaded. Define or load a Custom architecture."
            elif predict_ready: model_name = getattr(self.model, 'architecture_name', 'Unknown'); status_msg = f"Status: Ready. Model '{model_name}' loaded/trained."
            elif train_ready: status_msg = "Status: Data & Arch ready. Ready to train."
            else: status_msg = "Status: Ready."
            self.status_var.set(status_msg)

        self.predict_button.config(state=predict_state)

        widgets_to_control = [
            self.load_data_button, self.load_arch_button, self.load_params_button,
            self.build_arch_button, self.arch_combobox
        ]
        try:
            if len(self.winfo_children()) > 0 and len(self.winfo_children()[0].winfo_children()) > 0:
                 data_group = self.winfo_children()[0].winfo_children()[0].winfo_children()[0]
                 for w in data_group.winfo_children():
                      if isinstance(w, (ttk.Entry, ttk.Button)) and w not in [self.load_data_button, self.load_arch_button, self.load_params_button]: widgets_to_control.append(w)
            if len(self.winfo_children()) > 0 and len(self.winfo_children()[0].winfo_children()) > 0:
                 train_group = self.winfo_children()[0].winfo_children()[0].winfo_children()[1]
                 for w in train_group.winfo_children():
                      if isinstance(w, (ttk.Entry, ttk.Checkbutton, ttk.Button)) and w not in [self.train_button, self.build_arch_button, self.arch_combobox]: widgets_to_control.append(w)
            if len(self.winfo_children()) > 0 and len(self.winfo_children()[0].winfo_children()) > 0:
                 predict_group = self.winfo_children()[0].winfo_children()[0].winfo_children()[2]
                 for w in predict_group.winfo_children():
                      if isinstance(w, (ttk.Entry, ttk.Button)) and w != self.predict_button: widgets_to_control.append(w)
        except Exception as e: print(f"Warning: Error finding widgets: {e}", file=sys.__stderr__)

        for widget in widgets_to_control:
             if widget and hasattr(widget, 'configure') and 'state' in widget.configure():
                try:
                    current_state = widget.cget('state')
                    if current_state != general_state: widget.config(state=general_state)
                except tk.TclError: pass
                except Exception as e: print(f"Warning: Failed state set {widget}: {e}", file=sys.__stderr__)


    # --- Architecture Handling ---

    def open_arch_builder(self):
        # Removed DEBUG print
        builder = ArchitectureBuilder(self, self.current_arch_def)

    def set_custom_architecture(self, arch_def):
        # Removed DEBUG print
        self.current_arch_def = arch_def
        self.current_arch_name.set("Custom")
        self.arch_load_path.set("")
        self.model = None
        self.set_ui_state(False)

    def load_arch_def_from_path(self, filepath=None):
        # Removed DEBUG print
        if filepath is None: path = self.arch_load_path.get()
        else: path = filepath
        if not path or not os.path.exists(path):
            if filepath is None: messagebox.showerror("Error", "Arch def file not found.")
            return
        try:
            with open(path, 'r') as f: loaded_def = json.load(f)
            if isinstance(loaded_def, list) and all(isinstance(item, dict) for item in loaded_def):
                # Use print() which now goes to queue
                print(f"Architecture definition loaded from: {path}")
                self.current_arch_def = loaded_def
                self.current_arch_name.set("Custom")
                self.arch_load_path.set(path)
                self.model = None
                self.set_ui_state(False)
            else:
                messagebox.showerror("Load Error", "Invalid JSON format.")
                self.arch_load_path.set("")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load/parse arch file:\n{e}")
            # Use print() which now goes to queue (if redirection works)
            print(f"ERROR loading arch file: {e}")
            traceback.print_exc() # Goes to queue
            self.arch_load_path.set("")
            self.current_arch_def = None
            self.set_ui_state(False)

    def on_arch_select(self, event=None):
        # Removed DEBUG print
        selected_name = self.current_arch_name.get()
        if selected_name != 'Custom':
            self.current_arch_def = None
            self.arch_load_path.set("")
            print(f"Switched to predefined arch: {selected_name}") # Goes to queue
        elif not self.current_arch_def:
             print("Selected 'Custom', but none defined/loaded.") # Goes to queue
        else:
             print("Keeping current custom arch definition.") # Goes to queue

        self.model = None
        self.set_ui_state(False)


    # --- Core Logic Functions ---

    def load_data_thread(self):
        # (Remains the same - uses print() which should now go to queue)
        data_path = self.data_dir.get()
        if not data_path or not os.path.exists(data_path):
            messagebox.showerror("Error", "Data directory not found.")
            return
        self.status_var.set("Status: Loading data...")
        self.set_ui_state(is_running=True)
        print("Starting data loading thread...") # Goes to queue
        threading.Thread(target=self._load_data_task, args=(data_path,), daemon=True).start()


    def _load_data_task(self, data_path):
        # (Uses print() which should now go to queue)
        try:
            print(f"Loading data from {data_path} in background thread...")
            images, labels, class_names = load_data_from_directory(data_path, quiet=True)

            if images is None:
                self.after(0, lambda: messagebox.showerror("Error", "No images found."))
                self.data_loaded = False
                print("Data loading failed: No images found.")
                return

            self.X_train, self.y_train, self.class_names = images, labels, list(class_names)
            self.data_loaded = True

            msg = f"Successfully loaded {images.shape[0]} images. Classes: {self.class_names}"
            print(msg)
            self.after(0, self.status_var.set, f"Status: Loaded {images.shape[0]} images.")

        except Exception as e:
            print(f"--- DATA LOADING FAILED ---")
            print(f"Error: {e}")
            traceback.print_exc()
            self.after(0, lambda exc=e: messagebox.showerror("Data Load Failed", f"An error occurred:\n{exc}"))
            self.data_loaded = False
        finally:
            self.after(0, self.set_ui_state, False)
            print("Data loading task finished.")


    def load_model_params_from_path(self):
        # (Uses print() which should now go to queue)
        model_path = self.model_load_path.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Model parameters file (.npz) not found.")
            return

        try:
            print(f"Attempting CNNModel.load('{model_path}')...")
            print(f"Loading model parameters from {model_path}...")
            self.model = CNNModel.load(model_path)

            print(f"CNNModel.load successful. Arch='{self.model.architecture_name}', Classes={self.model.num_classes}")

            self.current_arch_def = self.model.architecture_def
            loaded_arch_display_name = self.model.architecture_name
            if loaded_arch_display_name == "CustomLoaded": loaded_arch_display_name = "Custom"
            self.current_arch_name.set(loaded_arch_display_name)
            self.arch_load_path.set("")

            self.class_names = list(self.model.class_names)
            self.status_var.set(f"Status: Loaded model '{self.model.architecture_name}'. Ready.")
            print("Model parameters loaded successfully.")
            self.set_ui_state(False)

        except Exception as e:
            print(f"ERROR loading model parameters: {e}")
            traceback.print_exc()
            messagebox.showerror("Load Error", f"Failed to load model parameters:\n{e}")
            self.model = None
            self.set_ui_state(False)


    def start_training_thread(self):
        # (Remains the same - uses print() which should now go to queue)
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load data before training.")
            return
        arch_choice = self.current_arch_name.get()
        if arch_choice == 'Custom' and not self.current_arch_def:
             messagebox.showerror("Error", "Custom architecture selected, but none is defined or loaded.")
             return

        self.set_ui_state(is_running=True)
        print("Starting training thread...")
        # Pass the log queue to the thread target
        threading.Thread(target=self.run_training, args=(self.log_queue,), daemon=True).start()

    def run_training(self, output_queue): # Accept the queue as argument
        # (Corrected indentation and lambda error capture)
        try:
            # Send initial messages directly to queue
            save_path = self.model_save_path.get()
            jit_enabled = self.jit_var.get()
            layers_numpy.JIT_ENABLED = jit_enabled
            output_queue.write("---------------------------------\n")
            output_queue.write(f"JIT Compilation: {'ENABLED' if jit_enabled else 'DISABLED'}\n")
            output_queue.write("---------------------------------\n")
            # time.sleep(0.1) # Less critical

            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            batch = int(self.batch_var.get())
            clip_str = self.clip_var.get()
            clip = float(clip_str) if clip_str else None
            decay = float(self.decay_var.get())
            num_classes = len(self.class_names)

            arch_choice = self.current_arch_name.get()
            arch_def_to_use = None
            if arch_choice == 'Custom':
                if not self.current_arch_def: raise ValueError("Custom arch selected but none defined.")
                arch_def_to_use = self.current_arch_def
                output_queue.write("Using Custom architecture definition...\n")
            else:
                 arch_def_to_use = arch_choice
                 output_queue.write(f"Using Predefined architecture: {arch_choice}\n")

            reuse_model = False
            if self.model is not None:
                arch_match = False
                model_arch_def = getattr(self.model, 'architecture_def', None)
                model_arch_name = getattr(self.model, 'architecture_name', None)
                if isinstance(arch_def_to_use, str) and model_arch_name == arch_def_to_use: arch_match = True
                elif isinstance(arch_def_to_use, list) and model_arch_def == arch_def_to_use: arch_match = True

                if arch_match and self.model.num_classes == num_classes:
                    output_queue.write("Reusing existing model instance.\n")
                    reuse_model = True
                else:
                    output_queue.write("Current model instance does not match target. Reinitializing.\n")
                    self.model = None

            if not reuse_model:
                output_queue.write("Initializing new model...\n")
                self.model = CNNModel(architecture_def=arch_def_to_use, num_classes=num_classes)
                self.model.class_names = np.array(self.class_names, dtype=object)

            # --- Pass output_queue to train function ---
            train(
                self.model, self.X_train, self.y_train,
                epochs=epochs, learning_rate=lr, batch_size=batch,
                clip_grad=clip, lr_decay=decay,
                output_stream=output_queue # Pass the queue here
            )

            if save_path:
                # model.save prints to stdout (now the queue)
                self.model.save(save_path)
            else:
                output_queue.write("Note: Model was not saved (no save path specified).\n")

        except Exception as e:
            # Log error details to stderr (now the queue) AND schedule messagebox
            print(f"--- TRAINING FAILED ---") # Goes to queue
            print(f"Error: {e}") # Goes to queue
            traceback.print_exc() # Goes to queue
            # Schedule messagebox using corrected lambda
            self.after(0, lambda exc=e: messagebox.showerror("Training Failed", f"An error occurred:\n{exc}"))
        finally:
            output_queue.write("\nTraining process finished.\n") # Goes to queue
            self.after(0, self.set_ui_state, False) # Schedule UI update

    def run_prediction(self):
        # --- MODIFIED: Write output to queue ---
        print("\n--- Running Prediction ---") # Goes to queue
        img_path = self.image_path.get()
        layers_numpy.JIT_ENABLED = self.jit_var.get()

        if self.model is None:
            print("Error: No model loaded.") # Goes to queue
            messagebox.showerror("Error", "No model is loaded or trained.")
            return

        if not img_path or not os.path.exists(img_path):
            print(f"Error: Image not found at '{img_path}'") # Goes to queue
            if img_path: messagebox.showerror("Error", f"Image not found:\n{img_path}")
            return

        try:
            img = read_image(img_path)
            if img is None: raise ValueError("Could not read image.")
            img_batch = np.expand_dims(img, axis=0)

            print("Running forward pass...") # Goes to queue
            logits, _ = self.model.forward(img_batch)
            print("Forward pass complete.") # Goes to queue

            exp_z = np.exp(logits - np.max(logits))
            probs = (exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)).flatten()

            class_labels = self.model.class_names
            valid_labels = False
            if class_labels is not None:
                try:
                    num_labels = len(class_labels)
                    if num_labels == len(probs): valid_labels = True
                except TypeError: pass

            if not valid_labels:
                 print("Warning: Class names missing/mismatch. Using defaults.") # Goes to queue
                 class_labels = [f"class_{i}" for i in range(len(probs))]

            results = list(zip(class_labels, probs))
            results.sort(key=lambda x: x[1], reverse=True)

            # --- Print results to queue ---
            print("Probabilities:")
            for cname, p in results:
                print(f"  {cname}: {p:.4f}")
            print(f"\nPredicted class: {results[0][0]}")
            # --- END Print results to queue ---

            # Show results in messagebox (remains the same)
            messagebox.showinfo(
                "Prediction Result",
                f"Predicted class: {results[0][0]}\n\n" +
                "\n".join(f"{c}: {p:.2%}" for c, p in results[:3])
            )

        except Exception as e:
            # Print error to queue
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            messagebox.showerror("Prediction Error", f"Failed to predict: {e}")
        # --- END MODIFIED ---


# --- Main Execution ---
def main_gui():
    # Removed DEBUG prints
    try:
        app = App()
        app.mainloop()
        # Removed DEBUG prints
    except Exception as e:
        # Print critical errors to original stderr
        print(f"CRITICAL ERROR during GUI startup or mainloop: {e}", file=sys.__stderr__)
        traceback.print_exc(file=sys.__stderr__)
        try:
            root = tk.Tk(); root.withdraw(); messagebox.showerror("Fatal GUI Error", f"Failed to start GUI:\n{e}")
        except Exception: pass
        sys.exit(1)


if __name__ == "__main__":
    # Ensure stdout/stderr are normal if run directly
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Starting GUI directly... (Better run as module: `python -m leekai gui`)", file=sys.__stderr__)
    main_gui()