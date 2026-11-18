
class TextComposer:
    def __init__(self):
        self.text = ""

    def append_letter(self, letter):
        if letter:
            self.text += letter

    # Alias for flexibility
    add_letter = append_letter

    def add_space(self):
        self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

