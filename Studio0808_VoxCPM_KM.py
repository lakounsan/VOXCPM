# -*- coding: utf-8 -*-
"""
Studio0808 VoxCPM - កម្មវិធីសំយោគសំឡេង AI (Khmer Launcher)
Directly launches the application with Khmer (ភាសាខ្មែរ) interface.
"""
import sys
import os

if __name__ == "__main__":
    from Studio0808_VoxCPM import App
    app = App(default_lang="km")
    app.mainloop()
