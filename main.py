import sys
import random
import time
import json
import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, 
                             QRadioButton, QButtonGroup, QLineEdit, QTabWidget, 
                             QGridLayout, QGroupBox, QSlider, QFileDialog, QListWidget,
                             QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pygame

# Define Morse code dictionary
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', 
    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
    '8': '---..', '9': '----.', '.': '.-.-.-', ',': '--..--', '?': '..--..', 
    '/': '-..-.', '@': '.--.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
    '(': '-.--.', ')': '-.--.-', '"': '.-..-.', '\'': '.----.', ':': '---...',
    ';': '-.-.-.', '!': '-.-.--'
}

class MorseCodePlayer:
    def __init__(self, freq=600, char_wpm=20, word_wpm=5):
        pygame.mixer.init()
        self.freq = freq
        self.set_speeds(char_wpm, word_wpm)
        
    def set_speeds(self, char_wpm, word_wpm):
        # Calculate timing parameters based on WPM using Farnsworth method
        self.char_wpm = char_wpm
        self.word_wpm = word_wpm
        
        # Paris standard: "PARIS" is 50 dot units
        # At N WPM, we need N * 50 dot units per minute
        self.dit_length = 60.0 / (char_wpm * 50)  # Duration of a dit in seconds
        self.dah_length = 3 * self.dit_length     # A dah is 3 times longer than a dit
        
        # For Farnsworth, we adjust the pauses between characters and words
        self.element_gap = self.dit_length  # Gap between elements within a character
        
        # Calculate character and word gaps based on the word WPM
        word_dit_length = 60.0 / (word_wpm * 50)
        self.char_gap = word_dit_length * 3  # Gap between characters 
        self.word_gap = word_dit_length * 7  # Gap between words
    
    def set_frequency(self, freq):
        self.freq = freq
    
    def generate_tone(self, duration):
        # Generate a sine wave of the specified frequency and duration
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * self.freq * t)
        # Apply fade in/out to avoid clicks
        fade_duration = min(0.01, duration / 10)
        fade_samples = int(fade_duration * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
        # Convert to 16-bit data
        audio = np.asarray(tone * 32767, dtype=np.int16)
        return pygame.sndarray.make_sound(audio)
    
    def play_character(self, char):
        if char not in MORSE_CODE:
            return  # Skip unsupported characters
        
        code = MORSE_CODE[char]
        for i, symbol in enumerate(code):
            if symbol == '.':
                sound = self.generate_tone(self.dit_length)
                sound.play()
                pygame.time.wait(int(self.dit_length * 1000))  # Wait for the dit to finish
            elif symbol == '-':
                sound = self.generate_tone(self.dah_length)
                sound.play()
                pygame.time.wait(int(self.dah_length * 1000))  # Wait for the dah to finish
            
            # Add gap between elements (if not the last element)
            if i < len(code) - 1:
                pygame.time.wait(int(self.element_gap * 1000))
        
        # Add gap between characters
        pygame.time.wait(int(self.char_gap * 1000))
    
    def play_text(self, text):
        for i, char in enumerate(text.upper()):
            if char == ' ':
                # Add word gap (subtract the already added character gap)
                pygame.time.wait(int((self.word_gap - self.char_gap) * 1000))
            else:
                self.play_character(char)

class MorseCodeStudySession:
    def __init__(self, characters, rounds=float('inf'), round_size=10):
        self.characters = characters
        self.rounds = rounds
        self.round_size = round_size
        self.current_round = 0
        self.stats = {}
        self.initialize_stats()
        
    def initialize_stats(self):
        for char in self.characters:
            self.stats[char] = {
                'attempts': 0,
                'correct': 0,
                'response_times': []
            }
    
    def generate_round_sequence(self):
        # If we have statistics, calculate difficulty scores
        if any(self.stats[char]['attempts'] > 0 for char in self.characters):
            # Calculate difficulty scores based on error rate and response time
            difficulty_scores = {}
            for char in self.characters:
                stats = self.stats[char]
                attempts = stats['attempts']
                correct = stats['correct']
                response_times = stats['response_times']
                
                # Calculate error rate (default to 0.5 if no attempts)
                error_rate = 1.0 - (correct / attempts) if attempts > 0 else 0.5
                
                # Calculate average response time (default to 1.0 second if no data)
                avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
                
                # Combine error rate and response time into a single difficulty score
                difficulty_scores[char] = error_rate + avg_response_time
            
            # Normalize difficulty scores
            max_score = max(difficulty_scores.values())
            normalized_scores = {char: score / max_score for char, score in difficulty_scores.items()}
            
            # Add baseline probability to ensure all characters are included
            baseline_probability = 0.1
            total_weight = sum(normalized_scores.values()) + baseline_probability * len(self.characters)
            probabilities = {
                char: (normalized_scores[char] + baseline_probability) / total_weight
                for char in self.characters
            }
            
            # Generate weighted random selection
            sequence = random.choices(
                population=self.characters,
                weights=[probabilities[char] for char in self.characters],
                k=self.round_size
            )
        else:
            # For the first round, just randomly select characters
            sequence = [random.choice(self.characters) for _ in range(self.round_size)]
        
        return sequence
    
    def record_result(self, char, correct, response_time):
        self.stats[char]['attempts'] += 1
        if correct:
            self.stats[char]['correct'] += 1
        self.stats[char]['response_times'].append(response_time)
    
    def get_char_stats(self, char):
        stats = self.stats.get(char, {'attempts': 0, 'correct': 0, 'response_times': []})
        accuracy = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0
        avg_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0
        return accuracy, avg_time
    
    def get_overall_stats(self):
        total_attempts = sum(self.stats[char]['attempts'] for char in self.characters)
        total_correct = sum(self.stats[char]['correct'] for char in self.characters)
        all_times = []
        for char in self.characters:
            all_times.extend(self.stats[char]['response_times'])
        
        accuracy = total_correct / total_attempts if total_attempts > 0 else 0
        avg_time = sum(all_times) / len(all_times) if all_times else 0
        return accuracy, avg_time

class MorseCodeTrainer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Morse Code Trainer")
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.morse_player = MorseCodePlayer()
        self.session = None
        self.current_round_sequence = []
        self.current_char_index = 0
        self.current_round = 0
        self.mode = "single_keyboard"  # Default mode
        self.auto_recognize_enabled = True # Default input recognition mode
        self.is_session_active = False
        self.user_input = ""
        self.start_time = 0
        self.dark_mode_enabled = False
        
        # Session history
        self.session_history = []
        self.load_session_history()
        
        # Setup UI
        self.setup_ui()
        self.apply_style()
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different sections
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Create tabs
        self.setup_practice_tab(tabs)
        self.setup_stats_tab(tabs)
        self.setup_settings_tab(tabs)
        
    def setup_practice_tab(self, tabs):
        practice_widget = QWidget()
        practice_layout = QVBoxLayout(practice_widget)
        
        # Character selection section
        char_selection_group = QGroupBox("Character Selection")
        char_selection_layout = QVBoxLayout()
        
        # Create checkboxes for character groups
        self.create_character_selection(char_selection_layout)
        
        char_selection_group.setLayout(char_selection_layout)
        practice_layout.addWidget(char_selection_group)
        
        # Session configuration
        session_config_group = QGroupBox("Session Configuration")
        session_config_layout = QGridLayout()
        
        # Character speed (WPM)
        session_config_layout.addWidget(QLabel("Character Speed (WPM):"), 0, 0)
        self.char_wpm_spinner = QSpinBox()
        self.char_wpm_spinner.setRange(5, 50)
        self.char_wpm_spinner.setValue(20)
        self.char_wpm_spinner.setSingleStep(1)
        session_config_layout.addWidget(self.char_wpm_spinner, 0, 1)
        
        # Word speed (WPM)
        session_config_layout.addWidget(QLabel("Word Speed (WPM):"), 1, 0)
        self.word_wpm_spinner = QSpinBox()
        self.word_wpm_spinner.setRange(2, 30)
        self.word_wpm_spinner.setValue(5)
        self.word_wpm_spinner.setSingleStep(1)
        session_config_layout.addWidget(self.word_wpm_spinner, 1, 1)
        
        # Tone frequency
        session_config_layout.addWidget(QLabel("Tone Frequency (Hz):"), 2, 0)
        self.freq_spinner = QSpinBox()
        self.freq_spinner.setRange(200, 1200)
        self.freq_spinner.setValue(600)
        self.freq_spinner.setSingleStep(10)
        session_config_layout.addWidget(self.freq_spinner, 2, 1)
        
        # Number of rounds
        session_config_layout.addWidget(QLabel("Number of Rounds:"), 0, 2)
        self.rounds_spinner = QSpinBox()
        self.rounds_spinner.setRange(1, 100)
        self.rounds_spinner.setValue(5)
        self.rounds_spinner.setSingleStep(1)
        self.rounds_spinner.setSpecialValueText("Unlimited")
        session_config_layout.addWidget(self.rounds_spinner, 0, 3)
        
        # Round size
        session_config_layout.addWidget(QLabel("Characters per Round:"), 1, 2)
        self.round_size_spinner = QSpinBox()
        self.round_size_spinner.setRange(5, 100)
        self.round_size_spinner.setValue(10)
        self.round_size_spinner.setSingleStep(5)
        session_config_layout.addWidget(self.round_size_spinner, 1, 3)
        
        session_config_group.setLayout(session_config_layout)
        practice_layout.addWidget(session_config_group)
        
        # Mode selection
        mode_group = QGroupBox("Practice Mode")
        mode_layout = QHBoxLayout()
        
        self.hand_copy_radio = QRadioButton("Hand Copy")
        self.hand_copy_radio.setToolTip("Copy on paper, report accuracy at the end")
        self.single_keyboard_radio = QRadioButton("Single Keyboard")
        self.single_keyboard_radio.setToolTip("Type each character, wait for next")
        self.continuous_keyboard_radio = QRadioButton("Continuous Keyboard")
        self.continuous_keyboard_radio.setToolTip("Characters play continuously, type as you go")
        
        self.single_keyboard_radio.setChecked(True)  # Default
        
        mode_layout.addWidget(self.hand_copy_radio)
        mode_layout.addWidget(self.single_keyboard_radio)
        mode_layout.addWidget(self.continuous_keyboard_radio)
        
        mode_group.setLayout(mode_layout)
        practice_layout.addWidget(mode_group)
        
        # Connect mode radio buttons
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.hand_copy_radio, 0)
        self.mode_group.addButton(self.single_keyboard_radio, 1)
        self.mode_group.addButton(self.continuous_keyboard_radio, 2)
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        
        # Disable statistics for hand copy checkbox
        self.disable_stats_checkbox = QCheckBox("Disable statistics tracking for hand copy mode")
        self.disable_stats_checkbox.setVisible(False)
        practice_layout.addWidget(self.disable_stats_checkbox)
        
        # Practice area
        practice_area_group = QGroupBox("Practice Area")
        practice_area_layout = QVBoxLayout()
        
        # Current character display
        self.current_char_label = QLabel("Ready to start")
        self.current_char_label.setAlignment(Qt.AlignCenter)
        self.current_char_label.setFont(QFont("Arial", 48, QFont.Bold))
        practice_area_layout.addWidget(self.current_char_label)
        
        # Input area
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Your Input:"))
        self.input_field = QLineEdit()
        self.input_field.setEnabled(False)
        self.input_field.returnPressed.connect(self.on_input_submitted)
        input_layout.addWidget(self.input_field)
        practice_area_layout.addLayout(input_layout)
        
        # Progress display
        self.progress_label = QLabel("Not started")
        self.progress_label.setAlignment(Qt.AlignCenter)
        practice_area_layout.addWidget(self.progress_label)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self.start_session)
        self.stop_button = QPushButton("Stop Session")
        self.stop_button.clicked.connect(self.stop_session)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        practice_area_layout.addLayout(button_layout)
        
        practice_area_group.setLayout(practice_area_layout)
        practice_layout.addWidget(practice_area_group)
        
        tabs.addTab(practice_widget, "Practice")
    
    def create_character_selection(self, layout):
        # Create a grid for character checkboxes
        char_grid = QGridLayout()
        
        # Letters group
        letters_group = QGroupBox("Letters")
        letters_layout = QGridLayout()
        
        # Create checkboxes for letters
        self.letter_checkboxes = {}
        for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            checkbox = QCheckBox(letter)
            self.letter_checkboxes[letter] = checkbox
            letters_layout.addWidget(checkbox, i // 9, i % 9)
        
        # Select all letters button
        select_all_letters = QPushButton("Select All Letters")
        select_all_letters.clicked.connect(lambda: self.toggle_group_selection(self.letter_checkboxes, True))
        letters_layout.addWidget(select_all_letters, 3, 0, 1, 4)
        
        # Clear all letters button
        clear_all_letters = QPushButton("Clear All Letters")
        clear_all_letters.clicked.connect(lambda: self.toggle_group_selection(self.letter_checkboxes, False))
        letters_layout.addWidget(clear_all_letters, 3, 5, 1, 4)
        
        letters_group.setLayout(letters_layout)
        layout.addWidget(letters_group)
        
        # Numbers group
        numbers_group = QGroupBox("Numbers")
        numbers_layout = QGridLayout()
        
        # Create checkboxes for numbers
        self.number_checkboxes = {}
        for i, number in enumerate('0123456789'):
            checkbox = QCheckBox(number)
            self.number_checkboxes[number] = checkbox
            numbers_layout.addWidget(checkbox, 0, i)
        
        # Select all numbers button
        select_all_numbers = QPushButton("Select All Numbers")
        select_all_numbers.clicked.connect(lambda: self.toggle_group_selection(self.number_checkboxes, True))
        numbers_layout.addWidget(select_all_numbers, 1, 0, 1, 5)
        
        # Clear all numbers button
        clear_all_numbers = QPushButton("Clear All Numbers")
        clear_all_numbers.clicked.connect(lambda: self.toggle_group_selection(self.number_checkboxes, False))
        numbers_layout.addWidget(clear_all_numbers, 1, 5, 1, 5)
        
        numbers_group.setLayout(numbers_layout)
        layout.addWidget(numbers_group)
        
        # Symbols group
        symbols_group = QGroupBox("Symbols")
        symbols_layout = QGridLayout()
        
        # Create checkboxes for symbols
        symbols = '.,:;?/=+-()@\'"!_'
        self.symbol_checkboxes = {}
        for i, symbol in enumerate(symbols):
            checkbox = QCheckBox(symbol)
            self.symbol_checkboxes[symbol] = checkbox
            symbols_layout.addWidget(checkbox, i // 8, i % 8)
        
        # Select all symbols button
        select_all_symbols = QPushButton("Select All Symbols")
        select_all_symbols.clicked.connect(lambda: self.toggle_group_selection(self.symbol_checkboxes, True))
        symbols_layout.addWidget(select_all_symbols, 2, 0, 1, 4)
        
        # Clear all symbols button
        clear_all_symbols = QPushButton("Clear All Symbols")
        clear_all_symbols.clicked.connect(lambda: self.toggle_group_selection(self.symbol_checkboxes, False))
        symbols_layout.addWidget(clear_all_symbols, 2, 4, 1, 4)
        
        symbols_group.setLayout(symbols_layout)
        layout.addWidget(symbols_group)
        
        # Preset selections
        presets_group = QGroupBox("Presets")
        presets_layout = QHBoxLayout()
        
        beginner_button = QPushButton("Beginner (ETANIM)")
        beginner_button.clicked.connect(lambda: self.apply_preset("ETANIM"))
        presets_layout.addWidget(beginner_button)
        
        basic_button = QPushButton("Basic (ETANIMSORUH)")
        basic_button.clicked.connect(lambda: self.apply_preset("ETANIMSORUH"))
        presets_layout.addWidget(basic_button)
        
        all_letters_button = QPushButton("All Letters")
        all_letters_button.clicked.connect(lambda: self.apply_preset("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        presets_layout.addWidget(all_letters_button)
        
        all_numbers_button = QPushButton("All Numbers")
        all_numbers_button.clicked.connect(lambda: self.apply_preset("0123456789"))
        presets_layout.addWidget(all_numbers_button)
        
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)
    
    def toggle_group_selection(self, checkbox_dict, state):
        for checkbox in checkbox_dict.values():
            checkbox.setChecked(state)
    
    def apply_preset(self, chars):
        # Clear all existing selections
        for checkbox_dict in [self.letter_checkboxes, self.number_checkboxes, self.symbol_checkboxes]:
            self.toggle_group_selection(checkbox_dict, False)
        
        # Apply the preset
        for char in chars:
            if char in self.letter_checkboxes:
                self.letter_checkboxes[char].setChecked(True)
            elif char in self.number_checkboxes:
                self.number_checkboxes[char].setChecked(True)
            elif char in self.symbol_checkboxes:
                self.symbol_checkboxes[char].setChecked(True)
    
    def setup_stats_tab(self, tabs):
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # Add different statistics views
        stats_tabs = QTabWidget()
        
        # Current session stats
        current_session_widget = QWidget()
        current_session_layout = QVBoxLayout(current_session_widget)
        
        self.current_stats_label = QLabel("No active session.")
        current_session_layout.addWidget(self.current_stats_label)
        
        # Current session chart
        self.current_session_figure = plt.figure(figsize=(8, 6))
        self.current_session_canvas = FigureCanvas(self.current_session_figure)
        current_session_layout.addWidget(self.current_session_canvas)
        
        stats_tabs.addTab(current_session_widget, "Current Session")
        
        # Historical stats
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_selected)
        history_layout.addWidget(self.history_list)
        
        # Historical session chart
        self.history_figure = plt.figure(figsize=(8, 6))
        self.history_canvas = FigureCanvas(self.history_figure)
        history_layout.addWidget(self.history_canvas)
        
        # Export data button
        export_button = QPushButton("Export Session History")
        export_button.clicked.connect(self.export_session_history)
        history_layout.addWidget(export_button)
        
        stats_tabs.addTab(history_widget, "Session History")
        
        # Progress over time
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        
        self.progress_figure = plt.figure(figsize=(8, 6))
        self.progress_canvas = FigureCanvas(self.progress_figure)
        progress_layout.addWidget(self.progress_canvas)
        
        stats_tabs.addTab(progress_widget, "Progress Over Time")
        
        stats_layout.addWidget(stats_tabs)
        tabs.addTab(stats_widget, "Statistics")
        
        # Populate history list and update charts
        self.update_history_list()
        self.update_progress_chart()
    
    def setup_settings_tab(self, tabs):
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # UI theme settings
        theme_group = QGroupBox("UI Theme")
        theme_layout = QVBoxLayout()
        
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)
        theme_layout.addWidget(self.dark_mode_checkbox)
        
        theme_group.setLayout(theme_layout)
        settings_layout.addWidget(theme_group)

        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QGridLayout()
        
        audio_layout.addWidget(QLabel("Default Tone Frequency (Hz):"), 0, 0)
        default_freq_spinner = QSpinBox()
        default_freq_spinner.setRange(200, 1200)
        default_freq_spinner.setValue(600)
        default_freq_spinner.setSingleStep(10)
        default_freq_spinner.valueChanged.connect(self.on_default_freq_changed)
        audio_layout.addWidget(default_freq_spinner, 0, 1)
        
        # Test tone button
        test_tone_button = QPushButton("Test Tone")
        test_tone_button.clicked.connect(lambda: self.play_test_tone(default_freq_spinner.value()))
        audio_layout.addWidget(test_tone_button, 0, 2)
        
        audio_group.setLayout(audio_layout)
        settings_layout.addWidget(audio_group)
        
        # Default speed settings
        speed_group = QGroupBox("Default Speed Settings")
        speed_layout = QGridLayout()
        
        speed_layout.addWidget(QLabel("Default Character Speed (WPM):"), 0, 0)
        default_char_wpm = QSpinBox()
        default_char_wpm.setRange(5, 50)
        default_char_wpm.setValue(20)
        default_char_wpm.setSingleStep(1)
        default_char_wpm.valueChanged.connect(self.on_default_char_wpm_changed)
        speed_layout.addWidget(default_char_wpm, 0, 1)
        
        speed_layout.addWidget(QLabel("Default Word Speed (WPM):"), 1, 0)
        default_word_wpm = QSpinBox()
        default_word_wpm.setRange(2, 30)
        default_word_wpm.setValue(5)
        default_word_wpm.setSingleStep(1)
        default_word_wpm.valueChanged.connect(self.on_default_word_wpm_changed)
        speed_layout.addWidget(default_word_wpm, 1, 1)
        
        speed_group.setLayout(speed_layout)
        settings_layout.addWidget(speed_group)
        
        # Add keyboard input settings
        keyboard_group = QGroupBox("Keyboard Input Settings")
        keyboard_layout = QVBoxLayout()
        
        self.auto_recognize_checkbox = QCheckBox("Auto-recognize characters (without pressing Enter)")
        self.auto_recognize_checkbox.setChecked(True)  # Enabled by default
        self.auto_recognize_checkbox.stateChanged.connect(self.on_auto_recognize_changed)
        keyboard_layout.addWidget(self.auto_recognize_checkbox)
        
        keyboard_group.setLayout(keyboard_layout)
        settings_layout.addWidget(keyboard_group, 3)
        
        # Data management
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        
        import_button = QPushButton("Import Session History")
        import_button.clicked.connect(self.import_session_history)
        data_layout.addWidget(import_button)
        
        export_button = QPushButton("Export Session History")
        export_button.clicked.connect(self.export_session_history)
        data_layout.addWidget(export_button)
        
        clear_button = QPushButton("Clear All Session History")
        clear_button.clicked.connect(self.clear_session_history)
        data_layout.addWidget(clear_button)
        
        data_group.setLayout(data_layout)
        settings_layout.addWidget(data_group)
        
        # About
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout()
        
        about_text = QLabel("Morse Code Trainer\nVersion 1.0\n\nA tool for learning Morse code using the Farnsworth method.")
        about_text.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_text)
        
        about_group.setLayout(about_layout)
        settings_layout.addWidget(about_group)
        
        settings_layout.addStretch()
        tabs.addTab(settings_widget, "Settings")

    def setup_input_field_behavior(self):
        # Disconnect any existing connections
        try:
            self.input_field.textChanged.disconnect()
        except:
            pass
            
        try:
            self.input_field.returnPressed.disconnect()
        except:
            pass
            
        # Connect appropriate signals based on settings
        if self.auto_recognize_enabled:
            self.input_field.textChanged.connect(self.on_text_changed)
        else:
            self.input_field.returnPressed.connect(self.on_input_submitted) 
        
    def apply_style(self):
        # Apply default style or dark mode based on setting
        if self.dark_mode_enabled:
            self.apply_dark_mode()
        else:
            self.apply_light_mode()
    
    def apply_dark_mode(self):
        # Set dark mode palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        self.setPalette(dark_palette)
        
        # Set style sheet for additional customization
        self.setStyleSheet("""
            QWidget {
                background-color: #353535;
                color: #ffffff;
            }
            QToolTip { 
                color: #ffffff; 
                background-color: #2a82da; 
                border: 1px solid white; 
            }
            QPushButton { 
                background-color: #2a82da;
                border: none;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #3a92ea;
            }
            QPushButton:pressed {
                background-color: #1a72ca;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #ffffff;
            }
        """)
        
        # Update matplotlib figures for dark mode
        plt.style.use('dark_background')
        for fig in [self.current_session_figure, self.history_figure, self.progress_figure]:
            fig.set_facecolor('#353535')
            fig.canvas.draw()
    
    def apply_light_mode(self):
        # Reset to default palette
        self.setPalette(self.style().standardPalette())
        
        # Set style sheet for light mode
        self.setStyleSheet("""
            QPushButton { 
                background-color: #0078d7;
                border: none;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #1a88e7;
            }
            QPushButton:pressed {
                background-color: #006bc7;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)
        
        # Update matplotlib figures for light mode
        plt.style.use('default')
        for fig in [self.current_session_figure, self.history_figure, self.progress_figure]:
            fig.set_facecolor('#f0f0f0')
            fig.canvas.draw()
    
    def toggle_dark_mode(self, state):
        self.dark_mode_enabled = bool(state)
        self.apply_style()
    
    def on_mode_changed(self, button):
        # Handle mode change
        if button == self.hand_copy_radio:
            self.mode = "hand_copy"
            self.disable_stats_checkbox.setVisible(True)
            self.input_field.setEnabled(False)
        elif button == self.single_keyboard_radio:
            self.mode = "single_keyboard"
            self.disable_stats_checkbox.setVisible(False)
            self.input_field.setEnabled(True)
        elif button == self.continuous_keyboard_radio:
            self.mode = "continuous_keyboard"
            self.disable_stats_checkbox.setVisible(False)
            self.input_field.setEnabled(True)
    
    def on_auto_recognize_changed(self, state):
        self.auto_recognize_enabled = bool(state)
        if self.is_session_active and self.mode in ["single_keyboard", "continuous_keyboard"]:
            self.setup_input_field_behavior()

    def on_text_changed(self, text):
        if not self.is_session_active:
            return
        
        # If we have exactly one character, process it
        if len(text) == 1:
            self.process_input(text)
    
    def on_default_freq_changed(self, value):
        self.freq_spinner.setValue(value)
    
    def on_default_char_wpm_changed(self, value):
        self.char_wpm_spinner.setValue(value)
    
    def on_default_word_wpm_changed(self, value):
        self.word_wpm_spinner.setValue(value)
    
    def start_session(self):
        # Get selected characters
        selected_chars = []
        for char, checkbox in self.letter_checkboxes.items():
            if checkbox.isChecked():
                selected_chars.append(char)
        for char, checkbox in self.number_checkboxes.items():
            if checkbox.isChecked():
                selected_chars.append(char)
        for char, checkbox in self.symbol_checkboxes.items():
            if checkbox.isChecked():
                selected_chars.append(char)
        
        if not selected_chars:
            QMessageBox.warning(self, "No Characters Selected", "Please select at least one character to practice.")
            return
        
        # Get session parameters
        rounds = float('inf') if self.rounds_spinner.value() == 1 else self.rounds_spinner.value()
        round_size = self.round_size_spinner.value()
        
        # Configure morse player
        self.morse_player.set_frequency(self.freq_spinner.value())
        self.morse_player.set_speeds(self.char_wpm_spinner.value(), self.word_wpm_spinner.value())
        
        # Create session
        self.session = MorseCodeStudySession(selected_chars, rounds, round_size)
        self.current_round = 0
        self.is_session_active = True
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.input_field.setEnabled(self.mode != "hand_copy")
        self.input_field.clear()

        # Set up input field behavior
        if self.mode in ["single_keyboard", "continuous_keyboard"]:
            self.setup_input_field_behavior()
        
        self.input_field.setFocus()
        
        # Start first round
        self.start_next_round()
    
    def stop_session(self):
        # Handle session completion
        self.is_session_active = False
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.input_field.setEnabled(False)
        self.current_char_label.setText("Session stopped")
        self.progress_label.setText("Session stopped")
        
        # If hand copy mode, prompt for results
        if self.mode == "hand_copy" and not self.disable_stats_checkbox.isChecked():
            self.prompt_hand_copy_results()
        
        # Save session to history
        if self.session:
            session_data = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'characters': self.session.characters,
                'mode': self.mode,
                'char_wpm': self.char_wpm_spinner.value(),
                'word_wpm': self.word_wpm_spinner.value(),
                'frequency': self.freq_spinner.value(),
                'rounds_completed': self.current_round,
                'stats': self.session.stats
            }
            self.session_history.append(session_data)
            self.save_session_history()
            self.update_history_list()
            self.update_progress_chart()
    
    # TODO: add feature
    def prompt_hand_copy_results(self):
        # To be implemented: dialog to enter hand copy results
        QMessageBox.information(self, "Hand Copy Results", 
                               "This would normally open a dialog to enter your hand copy results.\n"
                               "This feature is not fully implemented in this demo.")
    
    def start_next_round(self):
        if not self.is_session_active:
            return
            
        self.current_round += 1
        
        if self.current_round > self.session.rounds:
            self.stop_session()
            QMessageBox.information(self, "Session Complete", "You have completed all rounds!")
            return
        
        # Generate new sequence for the round
        self.current_round_sequence = self.session.generate_round_sequence()
        self.current_char_index = 0
        
        # Update progress display
        if self.session.rounds == float('inf'):
            self.progress_label.setText(f"Round {self.current_round} - Character 1/{len(self.current_round_sequence)}")
        else:
            self.progress_label.setText(f"Round {self.current_round}/{int(self.session.rounds)} - Character 1/{len(self.current_round_sequence)}")
        
        # Start playing characters
        self.play_next_character()
    
    def play_next_character(self):
        if not self.is_session_active or self.current_char_index >= len(self.current_round_sequence):
            # End of round
            self.start_next_round()
            return
        
        # Get the next character to play
        current_char = self.current_round_sequence[self.current_char_index]
        
        # Clear input field and update display
        self.input_field.clear()
        
        # If in continuous mode, we don't show the character immediately
        if self.mode != "continuous_keyboard":
            self.current_char_label.setText("?")
        
        # Play the character
        self.morse_player.play_character(current_char)
        
        # Record start time for response
        self.start_time = time.time()
        
        # In single keyboard mode, we wait for user input
        # In continuous mode or hand copy, we proceed to the next character automatically
        if self.mode == "single_keyboard":
            # Keep focus on input field
            self.input_field.setFocus()
        elif self.mode == "continuous_keyboard":
            # Show the character after playing it
            self.current_char_label.setText(current_char)
            # Move to next character after a delay based on word speed
            char_gap_ms = int(self.morse_player.char_gap * 1000)
            QTimer.singleShot(char_gap_ms, self.advance_to_next_character)
        elif self.mode == "hand_copy":
            # In hand copy mode, just show what's being played and continue
            self.current_char_label.setText(current_char)
            # Move to next character after a delay based on word speed
            char_gap_ms = int(self.morse_player.char_gap * 1000)
            QTimer.singleShot(char_gap_ms, self.advance_to_next_character)
    
    def advance_to_next_character(self):
        # Move to next character in the sequence
        self.current_char_index += 1
        
        # Update progress display
        if self.current_char_index < len(self.current_round_sequence):
            if self.session.rounds == float('inf'):
                self.progress_label.setText(f"Round {self.current_round} - Character {self.current_char_index + 1}/{len(self.current_round_sequence)}")
            else:
                self.progress_label.setText(f"Round {self.current_round}/{int(self.session.rounds)} - Character {self.current_char_index + 1}/{len(self.current_round_sequence)}")
        
        # Play next character if still in round
        if self.current_char_index < len(self.current_round_sequence):
            self.play_next_character()
        else:
            # End of round
            if self.mode == "hand_copy" and not self.disable_stats_checkbox.isChecked():
                self.prompt_hand_copy_results()
            self.start_next_round()
    
    # Add enhanced visual feedback for incorrect input
    def provide_incorrect_feedback(self, correct_char, user_input):
        # Save original label properties
        original_font = self.current_char_label.font()
        original_style = self.current_char_label.styleSheet()
        
        # Create feedback message
        feedback_text = f"{user_input} ➝ {correct_char}"
        self.current_char_label.setText(feedback_text)
        
        # Apply highlighted style
        self.current_char_label.setStyleSheet("""
            color: #721c24; 
            background-color: #f8d7da; 
            border: 2px solid #f5c6cb; 
            border-radius: 5px;
            padding: 5px;
        """)
        
        # Use a larger, bold font
        font = QFont(original_font)
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        self.current_char_label.setFont(font)
        
        # Play error tone (optional)
        if pygame.mixer.get_init():
            error_tone = self.generate_error_tone()
            error_tone.play()
        
        # Reset to original style after a delay
        QTimer.singleShot(800, lambda: self.current_char_label.setFont(original_font))
        QTimer.singleShot(800, lambda: self.current_char_label.setStyleSheet(original_style))
        QTimer.singleShot(800, lambda: self.current_char_label.setText(correct_char))

    # Add method to generate error tone
    def generate_error_tone(self):
        sample_rate = 44100
        duration = 0.2  # Short duration
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a descending tone
        freq1 = 880  # A5
        freq2 = 440  # A4
        tone = np.sin(2 * np.pi * np.linspace(freq1, freq2, len(t)) * t)
        
        # Apply fade in/out
        fade_samples = int(0.05 * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
        # Convert to 16-bit data
        audio = np.asarray(tone * 32767, dtype=np.int16)
        return pygame.sndarray.make_sound(audio)
    
    def process_input(self, input_text):
        if not self.is_session_active:
            return
            
        # Get user input and the current character
        user_input = input_text.strip().upper()
        
        if not user_input:
            return  # Ignore empty input
            
        if self.current_char_index >= len(self.current_round_sequence):
            return  # Ignore input after round end
            
        current_char = self.current_round_sequence[self.current_char_index]
        
        # Calculate response time
        response_time = time.time() - self.start_time
        
        # Check if correct
        is_correct = (user_input == current_char)
        
        # Record result in session stats
        self.session.record_result(current_char, is_correct, response_time)
        
        # Update UI based on correctness
        if is_correct:
            self.input_field.setStyleSheet("background-color: #d4edda;")
            self.current_char_label.setText(current_char)
        else:
            self.input_field.setStyleSheet("background-color: #f8d7da;")
            # Enhanced visual feedback for incorrect input
            self.provide_incorrect_feedback(current_char, user_input)
        
        # Reset input field style after a delay
        QTimer.singleShot(500, lambda: self.input_field.setStyleSheet(""))
        
        # Clear the input field for next character
        self.input_field.clear()
        
        # In single keyboard mode, move to next character
        if self.mode == "single_keyboard":
            # Move to next character after a short delay to show result
            QTimer.singleShot(800, self.advance_to_next_character)
        
        # Update current session statistics
        self.update_current_session_stats()
    
    def on_input_submitted(self):
        if not self.is_session_active:
            return
        
        user_input = self.input_field.text()
        self.process_input(user_input)
            
    def play_test_tone(self, frequency):
        # Play a short test tone to check audio
        test_player = MorseCodePlayer(frequency)
        test_player.play_character('E')  # Just a single dit
    
    def update_current_session_stats(self):
        if not self.session:
            self.current_stats_label.setText("No active session.")
            return
            
        # Calculate overall stats
        accuracy, avg_time = self.session.get_overall_stats()
        
        # Display stats
        stats_text = f"Current Session Statistics\n\n"
        stats_text += f"Round: {self.current_round}\n"
        stats_text += f"Characters: {', '.join(self.session.characters)}\n"
        stats_text += f"Accuracy: {accuracy * 100:.1f}%\n"
        stats_text += f"Average Response Time: {avg_time:.2f} seconds\n"
        
        self.current_stats_label.setText(stats_text)
        
        # Update chart
        self.update_current_session_chart()
    
    def update_current_session_chart(self):
        if not self.session:
            return
            
        # Clear the figure
        self.current_session_figure.clear()
        
        # Create accuracy subplot
        ax1 = self.current_session_figure.add_subplot(211)
        char_list = sorted(self.session.characters)
        accuracies = []
        
        for char in char_list:
            stats = self.session.stats.get(char, {'attempts': 0, 'correct': 0})
            acc = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0
            accuracies.append(acc * 100)
        
        bars = ax1.bar(char_list, accuracies)
        
        # Color bars based on accuracy
        for i, bar in enumerate(bars):
            if accuracies[i] >= 90:
                bar.set_color('green')
            elif accuracies[i] >= 70:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        ax1.set_title("Character Accuracy (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Create response time subplot
        ax2 = self.current_session_figure.add_subplot(212)
        response_times = []
        
        for char in char_list:
            stats = self.session.stats.get(char, {'response_times': []})
            avg_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0
            response_times.append(avg_time)
        
        ax2.bar(char_list, response_times)
        ax2.set_title("Average Response Time (seconds)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.current_session_figure.tight_layout()
        self.current_session_canvas.draw()
    
    def load_session_history(self):
        try:
            if os.path.exists('morse_session_history.json'):
                with open('morse_session_history.json', 'r') as f:
                    self.session_history = json.load(f)
        except Exception as e:
            print(f"Error loading session history: {e}")
            self.session_history = []
    
    def save_session_history(self):
        try:
            with open('morse_session_history.json', 'w') as f:
                json.dump(self.session_history, f)
        except Exception as e:
            print(f"Error saving session history: {e}")
    
    def update_history_list(self):
        self.history_list.clear()
        for i, session in enumerate(self.session_history):
            date = session.get('date', 'Unknown date')
            chars = ''.join(session.get('characters', []))
            if len(chars) > 10:
                chars = chars[:10] + "..."
            mode = session.get('mode', 'unknown')
            item_text = f"{date} - {chars} ({mode})"
            self.history_list.addItem(item_text)
    
    def on_history_item_selected(self, item):
        index = self.history_list.row(item)
        if 0 <= index < len(self.session_history):
            session = self.session_history[index]
            self.display_history_session(session)
    
    def display_history_session(self, session):
        # Display session details and chart
        date = session.get('date', 'Unknown date')
        chars = session.get('characters', [])
        mode = session.get('mode', 'unknown')
        char_wpm = session.get('char_wpm', 0)
        word_wpm = session.get('word_wpm', 0)
        
        # Calculate overall stats from stored session data
        stats = session.get('stats', {})
        total_attempts = sum(stats.get(char, {}).get('attempts', 0) for char in chars)
        total_correct = sum(stats.get(char, {}).get('correct', 0) for char in chars)
        
        all_times = []
        for char in chars:
            all_times.extend(stats.get(char, {}).get('response_times', []))
        
        accuracy = total_correct / total_attempts if total_attempts > 0 else 0
        avg_time = sum(all_times) / len(all_times) if all_times else 0
        
        # Update the history figure
        self.history_figure.clear()
        
        # Create accuracy subplot
        ax1 = self.history_figure.add_subplot(211)
        char_list = sorted(chars)
        accuracies = []
        
        for char in char_list:
            char_stats = stats.get(char, {'attempts': 0, 'correct': 0})
            acc = char_stats.get('correct', 0) / char_stats.get('attempts', 1) if char_stats.get('attempts', 0) > 0 else 0
            accuracies.append(acc * 100)
        
        bars = ax1.bar(char_list, accuracies)
        
        # Color bars based on accuracy
        for i, bar in enumerate(bars):
            if accuracies[i] >= 90:
                bar.set_color('green')
            elif accuracies[i] >= 70:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        ax1.set_title(f"Session {date} - Character Accuracy (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Create response time subplot
        ax2 = self.history_figure.add_subplot(212)
        response_times = []
        
        for char in char_list:
            char_stats = stats.get(char, {'response_times': []})
            times = char_stats.get('response_times', [])
            avg_time = sum(times) / len(times) if times else 0
            response_times.append(avg_time)
        
        ax2.bar(char_list, response_times)
        ax2.set_title("Average Response Time (seconds)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.history_figure.tight_layout()
        self.history_canvas.draw()
    
    def update_progress_chart(self):
        if not self.session_history:
            return
            
        # Clear the figure
        self.progress_figure.clear()
        
        # Extract dates and overall accuracies from session history
        dates = []
        accuracies = []
        response_times = []
        
        for session in self.session_history:
            date = session.get('date', '')
            chars = session.get('characters', [])
            stats = session.get('stats', {})
            
            total_attempts = 0
            total_correct = 0
            all_times = []
            
            for char in chars:
                char_stats = stats.get(char, {})
                total_attempts += char_stats.get('attempts', 0)
                total_correct += char_stats.get('correct', 0)
                all_times.extend(char_stats.get('response_times', []))
            
            if total_attempts > 0:
                dates.append(date)
                accuracies.append(total_correct / total_attempts * 100)
                if all_times:
                    response_times.append(sum(all_times) / len(all_times))
                else:
                    response_times.append(0)
        
        # Create accuracy over time subplot
        ax1 = self.progress_figure.add_subplot(211)
        ax1.plot(dates, accuracies, 'o-', color='blue')
        ax1.set_title("Accuracy Over Time")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Create response time over time subplot
        ax2 = self.progress_figure.add_subplot(212)
        ax2.plot(dates, response_times, 'o-', color='green')
        ax2.set_title("Response Time Over Time")
        ax2.set_ylabel("Response Time (s)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        self.progress_figure.tight_layout()
        self.progress_canvas.draw()
    
    def export_session_history(self):
        if not self.session_history:
            QMessageBox.information(self, "No Data", "There is no session history to export.")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Session History", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    json.dump(self.session_history, f)
                QMessageBox.information(self, "Export Successful", f"Session history exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export session history: {str(e)}")
    
    def import_session_history(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Session History", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    imported_history = json.load(f)
                
                # Validate the imported data
                if not isinstance(imported_history, list):
                    raise ValueError("Invalid session history format")
                
                # Merge with existing history or replace it
                merge = QMessageBox.question(self, "Import Options", 
                                             "Do you want to merge with existing history? Click 'No' to replace.",
                                             QMessageBox.Yes | QMessageBox.No)
                
                if merge == QMessageBox.Yes:
                    self.session_history.extend(imported_history)
                else:
                    self.session_history = imported_history
                
                self.save_session_history()
                self.update_history_list()
                self.update_progress_chart()
                
                QMessageBox.information(self, "Import Successful", f"Imported {len(imported_history)} session records")
            except Exception as e:
                QMessageBox.critical(self, "Import Failed", f"Failed to import session history: {str(e)}")
    
    def clear_session_history(self):
        confirm = QMessageBox.question(self, "Clear History", 
                                      "Are you sure you want to clear all session history? This cannot be undone.",
                                      QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            self.session_history = []
            self.save_session_history()
            self.update_history_list()
            self.update_progress_chart()
            QMessageBox.information(self, "History Cleared", "Session history has been cleared.")

def main():
    # Initialize pygame mixer
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    
    # Create and run the application
    app = QApplication(sys.argv)
    window = MorseCodeTrainer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
