# multiline_scanner.py: multiline scanner with indent/dedent support

class MultiLineScanner():
    def __init__(self, lines, allow_negative_numbers=True, 
        quotes=None, allow_extended_ids=None, echo_lines=False):
        if not quotes:
            quotes = ["'", '"']

        self.text = ""
        self.len = 0
        self.indent = 0         # index of first non-space character
        self.lines = lines
        self.line_count = len(lines)
        self.line_index = 0
        self.quotes = quotes
        self.index = 0
        self.token_type = None
        self.token = None
        self.prev_index = 0
        self.allow_negative_numbers = allow_negative_numbers
        self.allow_extended_ids = allow_extended_ids
        #console.print("Scanner created, text=", text)
        self.echo_lines = echo_lines
        self.last_full_comment = None
        self.stay_on_current_line = False    

    def scan(self, allow_extended_ids=None):
        if allow_extended_ids is None:
            allow_extended_ids = self.allow_extended_ids

        self.prev_index = self.index
        
        # skip spaces
        found_backslash = False

        while True:     # self.index < self.len:
            if self.index >= self.len:
                # not more output on this line
                if self.stay_on_current_line and not found_backslash:
                    self.token_type = "eol"
                    self.token = None
                    return self.token
                
                # try next line
                found_backslash = False 

                if self.line_index < self.line_count:
                    self.text = self.lines[self.line_index]

                    self.text = self.text.rstrip()
                    if self.echo_lines:
                        print(self.text)
                    
                    if "//" in self.text:
                        self.last_full_comment = self.text.strip()
                        self.text = self.text.split("//")[0]
                    elif "#" in self.text:
                        self.last_full_comment = self.text.strip()
                        self.text = self.text.split("#")[0]

                    self.indent = len(self.text) - len(self.text.lstrip())
                    self.text = self.text.strip()
                    self.len = len(self.text)

                    self.index = 0
                    self.line_index += 1
                    continue
                else:
                    # out of lines 
                    break

            if self.text[self.index] == ' ':
                self.index += 1
                found_backslash = False 

            elif self.text[self.index] == '\\':
                self.index += 1
                found_backslash = True 
            else:
                # found a non-space character
                break

        if self.index >= self.len:
            self.token_type =  "eof"
            self.token = None
        else:
            text = self.text

            ch = text[self.index]
            start = self.index
            self.index += 1

            if self.index < self.len:
                ch_next = text[self.index]
            else:
                ch_next = None

            if allow_extended_ids and ch.lower() in "~/_abcdefghijklmnopqrstuvwxyz*?$-'`" or (ch == "." and ch_next in [".", "/", "\\"]):
                # scan an ID or FILENAME or a WILDCARD or box-addr
                while self.index < self.len and text[self.index].lower() in "/@._-abcdefghijklmnopqrstuvwxyz0123456789$?*:/\\'`":
                    self.index += 1
                self.token_type = "id"
                self.token = text[start:self.index]

            elif not allow_extended_ids and ch.lower() in '_abcdefghijklmnopqrstuvwxyz': 
                # scan a simple ID
                while self.index < self.len and text[self.index].lower() in '_abcdefghijklmnopqrstuvwxyz0123456789':
                    self.index += 1
                self.token_type = "id"
                self.token = text[start:self.index]

            elif (ch in '.0123456789') or (self.allow_negative_numbers and ch == '-' and ch_next in '.0123456789'):
                # scan a NUMBER
                while self.index < self.len and text[self.index] in '.0123456789':
                    self.index += 1

                self.token_type = "number"
                self.token = text[start:self.index]

            elif ch in self.quotes:
                # scan a STRING
                quote = ch
                last_ch = ""
                while self.index < self.len:
                    if text[self.index] == quote and last_ch != "\\":
                        break
                    last_ch = text[self.index]
                    self.index += 1

                if self.index >= len(text) or text[self.index] != quote:
                    raise Exception("Unterminated string at offset=" + str(start) + " in cmd: " + text)

                self.token_type = "string"
                self.index += 1        # skip over the ending quote
                self.token = text[start+1:self.index-1]
                # un-embed contained quotes
                self.token = self.token.replace("\\" + quote, quote)
                
            else:
                # scan a special char
                self.token_type = "special"
                self.token = ch
                if self.index < self.len:
                    ch2 = ch + self.text[self.index]
                    #console.print("ch2=", ch2)
                    if ch2 in ["--", "<=", ">=", "!=", "<>", "==", "**"]:
                        self.index += 1
                        self.token = ch2

        #console.print("scanner.scan returning=", self.token, ", type=", self.token_type)
        return self.token

    def save_state(self):
        state = MultiLineScanner(self.lines)
        state.len = self.len
        state.index = self.index
        state.token_type = self.token_type
        state.token = self.token
        state.line_index = self.line_index
        state.indent = self.indent
        return state

    def restore_state(self, state):
        self.text = state.text
        self.len = state.len
        self.index = state.index
        self.token_type = state.token_type
        self.line_index = state.line_index
        self.token = state.token
        self.indent = state.indent

    def peek(self):
        # peek ahead 1 token
        state = self.save_state()
        tok = self.scan()
        state = self.restore_state(state)
        return tok

    def get_rest_of_text(self, include_current_token=False):
        if include_current_token:
            text = self.text[self.prev_index:]
        else:
            text = self.text[self.index:]

        # show input all processed
        self.token = text
        self.index = len(self.text)
        self.token_type = "text"

        self.text = text
        self.len = len(text)

        return text
                