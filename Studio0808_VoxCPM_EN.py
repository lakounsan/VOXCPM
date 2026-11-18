# -*- coding: utf-8 -*-
"""
Studio0808 VoxCPM - English Launcher
Directly launches the application with English interface.
"""
import sys
import os

if __name__ == "__main__":
    from Studio0808_VoxCPM import App
    app = App(default_lang="en")
    app.mainloop()
