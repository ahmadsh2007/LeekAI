import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os

# Define available layer types and their default parameters
LAYER_DEFAULTS = {
    "conv": {"type": "conv", "filters": 8, "size": 3, "stride": 1, "padding": 0},
    "relu": {"type": "relu"},
    "pool": {"type": "pool", "size": 2, "stride": 2},
    "flatten": {"type": "flatten"},
    "dense": {"type": "dense", "units": 64},
}

class ArchitectureBuilder(tk.Toplevel):
    def __init__(self, parent, current_arch_def=None):
        super().__init__(parent)
        self.parent = parent # Reference to the main App
        self.title("CNN Architecture Builder")
        self.geometry("600x600")
        self.minsize(450, 400)

        # The architecture definition (list of layer dicts)
        self.arch_def = list(current_arch_def) if current_arch_def else []

        # --- Main Layout ---
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        list_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(self, padding=(0, 0, 10, 10), width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        controls_frame.pack_propagate(False) # Prevent shrinking

        bottom_frame = ttk.Frame(self, padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Top: File Operations ---
        ttk.Button(top_frame, text="Load Arch (.json)", command=self.load_arch_def).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Save Arch (.json)", command=self.save_arch_def).pack(side=tk.LEFT, padx=5)

        # --- Left: Layer List ---
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.layer_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, exportselection=False, font=("Monospace", 10))
        self.layer_listbox.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.layer_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.layer_listbox.config(yscrollcommand=list_scroll.set)
        self.layer_listbox.bind("<<ListboxSelect>>", self.on_layer_select)
        self.populate_listbox() # Initial population

        # --- Right: Controls ---
        # Layer Type Buttons
        ttk.Label(controls_frame, text="Add Layer:", font=("Inter", 10, "bold")).pack(pady=(0, 5), anchor="w")
        ttk.Button(controls_frame, text="Convolution", command=lambda: self.add_layer("conv")).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="ReLU Activation", command=lambda: self.add_layer("relu")).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Max Pooling", command=lambda: self.add_layer("pool")).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Flatten", command=lambda: self.add_layer("flatten")).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Dense (Fully Connected)", command=lambda: self.add_layer("dense")).pack(fill=tk.X, pady=2)

        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Movement/Deletion Buttons
        ttk.Button(controls_frame, text="Move Up", command=self.move_layer_up).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Move Down", command=self.move_layer_down).pack(fill=tk.X, pady=2)
        ttk.Button(controls_frame, text="Delete Selected", command=self.delete_layer).pack(fill=tk.X, pady=2)

        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Parameter Editor Frame (populated dynamically)
        self.param_frame = ttk.Frame(controls_frame)
        self.param_frame.pack(fill=tk.X, pady=10)
        self.param_widgets = {} # To store dynamically created entry widgets

        # --- Bottom: Use/Cancel ---
        ttk.Button(bottom_frame, text="Use This Architecture", command=self.use_architecture).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        # Center window on parent
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)


    def populate_listbox(self):
        """Updates the listbox based on self.arch_def."""
        self.layer_listbox.delete(0, tk.END)
        for i, layer in enumerate(self.arch_def):
            self.layer_listbox.insert(tk.END, f"{i}: {self.get_layer_str(layer)}")

    def get_layer_str(self, layer_dict):
        """Generates a descriptive string for a layer dictionary."""
        l_type = layer_dict.get("type", "unknown")
        if l_type == "conv":
            return f"Conv(filters={layer_dict.get('filters', '?')}, size={layer_dict.get('size', '?')}x{layer_dict.get('size', '?')}, stride={layer_dict.get('stride', '?')}, pad={layer_dict.get('padding', '?')})"
        elif l_type == "relu":
            return "ReLU"
        elif l_type == "pool":
            return f"MaxPool(size={layer_dict.get('size', '?')}x{layer_dict.get('size', '?')}, stride={layer_dict.get('stride', '?')})"
        elif l_type == "flatten":
            return "Flatten"
        elif l_type == "dense":
            return f"Dense(units={layer_dict.get('units', '?')})"
        else:
            return f"Unknown Layer ({l_type})"

    def add_layer(self, layer_type):
        """Adds a new layer of the specified type to the end."""
        new_layer = LAYER_DEFAULTS.get(layer_type, {}).copy()
        if not new_layer:
            messagebox.showerror("Error", f"Unknown layer type: {layer_type}")
            return
        self.arch_def.append(new_layer)
        self.populate_listbox()
        # Select the newly added layer
        self.layer_listbox.selection_clear(0, tk.END)
        self.layer_listbox.selection_set(tk.END)
        self.layer_listbox.activate(tk.END)
        self.on_layer_select(None) # Trigger parameter editor update

    def get_selected_index(self):
        """Returns the index of the selected layer or None."""
        selection = self.layer_listbox.curselection()
        return selection[0] if selection else None

    def move_layer_up(self):
        idx = self.get_selected_index()
        if idx is not None and idx > 0:
            self.arch_def[idx], self.arch_def[idx-1] = self.arch_def[idx-1], self.arch_def[idx]
            self.populate_listbox()
            self.layer_listbox.selection_set(idx - 1)
            self.layer_listbox.activate(idx - 1)
            self.on_layer_select(None)

    def move_layer_down(self):
        idx = self.get_selected_index()
        if idx is not None and idx < len(self.arch_def) - 1:
            self.arch_def[idx], self.arch_def[idx+1] = self.arch_def[idx+1], self.arch_def[idx]
            self.populate_listbox()
            self.layer_listbox.selection_set(idx + 1)
            self.layer_listbox.activate(idx + 1)
            self.on_layer_select(None)

    def delete_layer(self):
        idx = self.get_selected_index()
        if idx is not None:
            del self.arch_def[idx]
            self.populate_listbox()
            self.clear_param_editor()
            # Try to select the previous item if possible
            if idx > 0:
                 self.layer_listbox.selection_set(idx - 1)
                 self.layer_listbox.activate(idx - 1)
                 self.on_layer_select(None)
            elif self.arch_def: # Select first item if list not empty
                 self.layer_listbox.selection_set(0)
                 self.layer_listbox.activate(0)
                 self.on_layer_select(None)


    def clear_param_editor(self):
        """Removes all widgets from the parameter editor frame."""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_widgets = {}

    def on_layer_select(self, event):
        """Populates the parameter editor based on the selected layer."""
        self.clear_param_editor()
        idx = self.get_selected_index()
        if idx is None:
            return

        layer = self.arch_def[idx]
        l_type = layer.get("type")

        ttk.Label(self.param_frame, text=f"Edit Layer {idx}: {l_type.upper()}", font=("Inter", 10, "bold")).pack(anchor="w")

        params_to_edit = {}
        if l_type == "conv":
            params_to_edit = {"filters": "Filters:", "size": "Kernel Size:", "stride": "Stride:", "padding": "Padding:"}
        elif l_type == "pool":
            params_to_edit = {"size": "Pool Size:", "stride": "Stride:"}
        elif l_type == "dense":
            params_to_edit = {"units": "Units:"}
        # ReLU and Flatten have no editable params

        for key, label_text in params_to_edit.items():
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(layer.get(key, '')))
            entry = ttk.Entry(frame, textvariable=var, width=8)
            entry.pack(side=tk.LEFT, padx=5)
            # Store var and entry for updating the arch_def
            self.param_widgets[key] = {"var": var, "entry": entry}
            # Add trace to update arch_def when value changes
            var.trace_add("write", lambda name, index, mode, k=key: self.update_layer_param(k))

    def update_layer_param(self, key):
        """Updates the arch_def when a parameter entry changes."""
        idx = self.get_selected_index()
        if idx is None or key not in self.param_widgets:
            return

        var = self.param_widgets[key]["var"]
        new_value_str = var.get()
        try:
            # Try converting to int, handle potential errors
            new_value = int(new_value_str)
            if new_value >= 0: # Basic validation
                 self.arch_def[idx][key] = new_value
                 # Update listbox display
                 self.layer_listbox.delete(idx)
                 self.layer_listbox.insert(idx, f"{idx}: {self.get_layer_str(self.arch_def[idx])}")
                 self.layer_listbox.selection_set(idx) # Keep selection
            else:
                print(f"Ignoring negative value for {key}: {new_value}")

        except ValueError:
            # Handle cases where input is not a valid integer (e.g., empty string)
             print(f"Invalid input for {key}: {new_value_str}")
             # Optionally, revert the entry or show an error indicator

    def validate_architecture(self):
        """Basic checks for common architecture errors."""
        has_flatten = False
        last_type = None
        for i, layer in enumerate(self.arch_def):
            l_type = layer.get("type")
            if l_type == "flatten":
                if has_flatten:
                     messagebox.showerror("Validation Error", f"Layer {i}: Multiple Flatten layers are not allowed.")
                     return False
                if last_type not in ["conv", "pool", "relu"]:
                    messagebox.showerror("Validation Error", f"Layer {i}: Flatten must follow a Conv, Pool, or ReLU layer.")
                    return False
                has_flatten = True
            elif l_type == "dense":
                if not has_flatten:
                    messagebox.showerror("Validation Error", f"Layer {i}: Dense layer must follow a Flatten layer.")
                    return False
            elif l_type == "conv" or l_type == "pool":
                 if has_flatten:
                     messagebox.showerror("Validation Error", f"Layer {i}: {l_type.capitalize()} cannot come after Flatten.")
                     return False
            last_type = l_type

        if not self.arch_def:
             messagebox.showerror("Validation Error", "Architecture cannot be empty.")
             return False

        if self.arch_def[-1].get("type") != "dense":
             messagebox.showerror("Validation Error", "The last layer must be a Dense layer (for classification output).")
             return False

        return True


    def save_arch_def(self):
        """Saves the current architecture definition to a JSON file."""
        if not self.arch_def:
            messagebox.showwarning("Save Arch", "Architecture is empty. Nothing to save.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Architecture Definition",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                json.dump(self.arch_def, f, indent=4)
            messagebox.showinfo("Save Arch", f"Architecture saved successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save architecture:\n{e}")

    def load_arch_def(self):
        """Loads an architecture definition from a JSON file."""
        filepath = filedialog.askopenfilename(
            title="Load Architecture Definition",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                loaded_def = json.load(f)
            # Basic validation: check if it's a list of dicts
            if isinstance(loaded_def, list) and all(isinstance(item, dict) for item in loaded_def):
                self.arch_def = loaded_def
                self.populate_listbox()
                self.clear_param_editor()
                messagebox.showinfo("Load Arch", f"Architecture loaded successfully from:\n{filepath}")
            else:
                messagebox.showerror("Load Error", "Invalid JSON format. Expected a list of layer dictionaries.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load or parse architecture file:\n{e}")


    def use_architecture(self):
        """Validates and passes the architecture back to the main GUI."""
        if not self.validate_architecture():
            return

        # Pass the definition back to the parent (main App)
        self.parent.set_custom_architecture(self.arch_def)
        self.destroy()