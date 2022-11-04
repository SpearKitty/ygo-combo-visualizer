import json
import time
import timeit
import ast
import re
import os
import ast
import cv2
import logging
import numpy as np
import tkinter as tk
import threading as thread
from urllib.request import Request, urlopen
from tkinter.scrolledtext import ScrolledText
from tkinter import Frame, Entry, Button, Listbox, Label, Toplevel, messagebox, Scrollbar
from ygoprodeck import fuzzy_search, get_db_version

MANUAL_RATE_LIMIT = 0.2
LOCAL_PATH        = os.path.join(os.getcwd(), 'local')
OUT_PATH          = os.path.join(os.getcwd(), 'out')
HARD_CARD_RESIZE  = (110, 153) #card sizes in final image
START_COORD       = (50, 50)   #anchor for placing images
ARROW_COLOR       = (240, 240, 240)  #color for the arrow
BG_COLOR          = (75, 75, 75)#color for the background
CARD_GAP_X        = 10 #               horiz card spacing
CARD_GAP_Y        = 15 #            vertical card spacing
#HAND_CODE         = '<END_HAND>'#unused
#PASS_LIST         = [HAND_CODE] #unused

ADD_STARTER_HOTKEY = 'q'
ADD_ENDER_HOTKEY   = 'e'

logger = logging.Logger(__name__, logging.INFO)

class ComboWrapper:
    def __init__(self):
        self.combo_json = json.loads('{}')
        self.sub_units  = json.loads('{}')
          #TODO: implement subunits SEPARATELY FROM KEY TUPLES
          #(add empty subunits if not used, only render if at least 1 isn't None)

    def add_key_tuple(self, names:list[str]):
        try:
            self.combo_json[str(names)]
        except:
            self.combo_json[str(names)] = []
        return self

    def add_end_tuple(self, names:list[str], result:list[str], hand:list[str]=None):
        #TODO: xyz materials eventually (1/8 of card shown?)
        self.combo_json[str(names)].append(result)
        if hand is not None:
            self.combo_json[str(names)][-1].extend(hand)
            print(f'handed_value={self.combo_json[str(names)][-1]}')
        return self

    def add_combo(self, names:list[str], result:list[str], hand:list[str]=None):
        self.add_key_tuple(names).add_end_tuple(names, result, hand)

class ScrollableListbox(Listbox):
    def __init__(self, parent):
        self.host_subframe = Frame(parent)
        self.scrollbar = Scrollbar(self.host_subframe, orient='vertical')
        self.scrollbar.config(command=self.yview)
        Listbox.__init__(self, self.host_subframe)
        self.config(yscrollcommand=self.scrollbar.set)

    def pack(self, **args):
        self.host_subframe.pack(**args)
        self.scrollbar.pack(side='right', fill='y')
        Listbox.pack(self, side='left', expand=1, fill='both')

class DisplayWindow():
    def __init__(self, card_data:list[dict[str, str]], combos:ComboWrapper):
        self.root = tk.Tk()
        self.root.title('Combo Builder')
        self.root.geometry('500x600')

        self.cards = card_data
        self.combos = combos
        self.temp_combos = []

        #search/add frame
        self.search_frame = Frame(self.root)
        self.bar_frame    = Frame(self.search_frame)

        self.search_input = Entry(self.bar_frame)
        self.search_input.bind('<KeyRelease>', lambda key:(
            self.do_search(self.search_input.get())
                if len(self.search_input.get()) >= 2 or key.keysym == 'Return' else 
            None)
            #tldr edopro search
        )

        self.search_button= Button(self.bar_frame, text='Search',
                command=lambda:self.do_search(self.search_input.get())
        )

        self.search_box   = ScrollableListbox(self.search_frame)

        self.search_box.bind(f'<{ADD_STARTER_HOTKEY}>', lambda key:
            self.starter_box.insert(self.starter_box.size(), self.search_box.get(self.search_box.curselection()[0]))
        )
        self.search_box.bind(f'<{ADD_ENDER_HOTKEY}>', lambda key:
            self.ender_box.insert(self.ender_box.size(), self.search_box.get(self.search_box.curselection()[0]))
        )
        self.search_box.bind('<Double-Button>', lambda key:
            self.preview_card()
        )

        self.add_frame = Frame(self.search_frame)

        self.starter_button = Button(self.add_frame, text=f'Add Starter ({ADD_STARTER_HOTKEY})', command=lambda:
            self.starter_box.insert(self.starter_box.size(), self.search_box.get(self.search_box.curselection()[0]))
        )

        self.ender_button = Button(self.add_frame, text=f'Add Ender ({ADD_ENDER_HOTKEY})', command=lambda:
            self.ender_box.insert(self.ender_box.size(), self.search_box.get(self.search_box.curselection()[0]))
        )

        #combo edit frame
        self.other_frame = Frame(self.root)
        self.starter_frame = Frame(self.other_frame)
        self.ender_frame   = Frame(self.other_frame)

        self.starter_box = ScrollableListbox(self.starter_frame)
        self.remove_starter = Button(self.starter_frame, text='Remove Selected', command=lambda:
            (self.starter_box.delete(self.starter_box.curselection()[0])
            if self.starter_box.size() > 0 else None)
        )
        self.ender_box   = ScrollableListbox(self.ender_frame)
        self.remove_ender = Button(self.ender_frame, text='Remove Selected', command=lambda:
            (self.ender_box.delete(self.ender_box.curselection()[0])
            if self.ender_box.size() > 0 else None)
        )

        #other edit frame
        self.lower_left_frame = Frame(self.other_frame, borderwidth=2, relief=tk.RAISED)

        self.finalize_combo = Button(self.lower_left_frame, text='Add Pair', command=lambda:
            self.add_combo()
        )

        self.render_combo_btn = Button(self.lower_left_frame, text='Render Combo Image', command=
            #>simply put a thread between the threads so the thread doesn't block the thread
            #lambda:thread.Thread(target=self.render_image, daemon=True).start()
            self.do_render
        )

        self.active_combo_list = ScrollableListbox(self.lower_left_frame)
        self.active_button_frame = Frame(self.lower_left_frame)
        self.edit_combo_button = Button(self.active_button_frame, text='Edit Pair', command=self.edit_pair)
        self.remove_combo_button = Button(self.active_button_frame, text='Remove Pair', command=self.remove_pair)
        self.duplicate_combo = Button(self.active_button_frame, text='Duplicate Pair', command=self.duplicate_pair)

        #pack all
        self.other_frame.pack(side='left', expand=1, fill='both')
        self.search_frame.pack(side='right', fill='y')
        self.bar_frame.pack(side='top', fill='x', padx=2, pady=2)
        self.search_input.pack(side='left', fill='x', expand=1, padx=2, pady=2)
        self.search_button.pack(side='right', fill='x', padx=2, pady=2)
        self.search_box.pack(side='top', fill='both', expand=1, padx=2, pady=2)
        self.add_frame.pack(side='bottom', fill='x')
        self.starter_button.pack(side='left', expand=1, fill='x', padx=2, pady=2)
        self.ender_button.pack(side='right', expand=1, fill='x', padx=2, pady=2)

        self.lower_left_frame.pack(side='bottom', fill='both', expand=1)
        self.render_combo_btn.pack(padx=2, pady=2, side='bottom', fill='x')
        self.finalize_combo.pack(padx=2, pady=2, side='top', fill='x')

        self.starter_frame.pack(side='left', fill='both', expand=1, padx=2, pady=2)
        self.remove_starter.pack(side='bottom', fill='x', padx=2, pady=2)
        Label(self.starter_frame, text='Combo Starters', borderwidth=2, relief=tk.RAISED).pack(side='top', fill='x')
        self.ender_frame.pack(side='right', fill='both', expand=1, padx=2, pady=2)
        self.remove_ender.pack(side='bottom', fill='x', padx=2, pady=2)
        Label(self.ender_frame, text='Combo End Board', borderwidth=2, relief=tk.RAISED).pack(side='top', fill='x')
        self.starter_box.pack(side='left', fill='both', expand=1)
        self.ender_box.pack(side='right', fill='both', expand=1)

        self.active_combo_list.pack(expand=1, fill='both', padx=1, pady=1)
        self.active_button_frame.pack(fill='x')
        self.edit_combo_button.pack(side='left', fill='x', expand=1, padx=2, pady=2)
        self.remove_combo_button.pack(side='left', fill='x', expand=1, padx=2, pady=2)
        self.duplicate_combo.pack(side='left', fill='x', expand=1, padx=2, pady=2)

    def preview_card(self):
        self.temp_card_name = self.search_box.get(self.search_box.curselection()[0])
        self.temp_card = image_from_name(self.temp_card_name)
        cv2.imshow(f'Preview ({self.temp_card_name})', self.temp_card)
        cv2.waitKey(0)

    def calculate_output_size(self):
        #TODO: add support for subunits (hand, gy, etc.), calculate space for line(s) and place text accordingly
        vert_lines = sum([len(o) for o in self.combos.combo_json.values()])
        horiz_lines = max([len(set(ast.literal_eval(key))) + max([len(set(sub)) for sub in (self.combos.combo_json[key])]) for key in self.combos.combo_json.keys()])
        self.output_hgt = vert_lines * (HARD_CARD_RESIZE[1] + CARD_GAP_Y) + 2 * START_COORD[1]
        self.output_wid = horiz_lines * (HARD_CARD_RESIZE[0] + CARD_GAP_X) + 2 * START_COORD[0] + 80
        return self

    def add_combo(self):
        if self.starter_box.size() > 0 and self.ender_box.size() > 0:
            self.temp_combos.append([self.starter_box.get(0, 'end'), self.ender_box.get(0, 'end')])
            self.active_combo_list.insert('end', self.temp_combos[-1])
            self.starter_box.delete(0, 'end')
            self.ender_box.delete(0, 'end')
        return self

    def edit_pair(self, swap_active:bool=True):
        if swap_active:
            self.add_combo()
            self.selected = self.active_combo_list.curselection()[0]
        else:
            self.selected = self.active_combo_list.size()-1
            
        self.move = self.temp_combos.pop(self.selected)
        
        self.active_combo_list.delete(self.selected, self.selected)
        self.starter_box.delete(0, 'end')
        self.ender_box.delete(0, 'end')
        
        for starter in self.move[0]:
            self.starter_box.insert('end', starter)
        for ender in self.move[1]:
            self.ender_box.insert('end', ender)
        return self
    
    def remove_pair(self):
        self.selected = self.active_combo_list.curselection()[0]
        self.temp_combos.pop(self.selected)
        self.active_combo_list.delete(self.selected, self.selected)
        return self

    def duplicate_pair(self):
        self.selected = self.active_combo_list.curselection()[0]
        self.temp_combos.append(self.temp_combos[self.selected])
        self.active_combo_list.insert('end', self.temp_combos[self.selected])
        self.edit_pair(False)
        return self

    def do_search(self, value:str) -> None:
        self.search_results = search_cards(value, self.cards)
        self.search_names = [card['name'] for card in self.search_results]
        self.search_box.delete(0, 'end')
        for name in self.search_names:
            self.search_box.insert('end', name)
        return self

    def do_render(self):
        self.log_win = LoggingWindow(self.root)
        self.scr_logger = ScrollLogger(self.log_win.output)
        logger.addHandler(self.scr_logger)
        thread.Thread(target=self.render_image, daemon=True).start()

    def render_image(self):
        try:
            for combo in self.temp_combos:
                self.combos.add_combo(combo[0], combo[1], None)
            self.calculate_output_size()
            self.render_combo_btn.configure(state='disabled')
            render_thread = thread.Thread(target=render_combo, args=(self.output_wid, self.output_hgt, self.combos, True))
            render_thread.start()
            render_thread.join()
            self.render_combo_btn.configure(state='normal')
            self.combos.combo_json = json.loads('{}')
            logger.removeHandler(self.scr_logger)
            #self.log_win.root.destroy()
        except:
            self.combos.combo_json = json.loads('{}')
            logger.removeHandler(self.scr_logger)
        return self

    def show(self):
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror('Critical Error', f'An error has caused the application to crash:{e}')

class ScrollLogger(logging.Handler):
    def __init__(self, widget):
        logging.Handler.__init__(self)
        self.widget = widget
        self.widget.configure(state='disabled')

    def emit(self, record):
        self.widget.configure(state='normal')
        self.widget.insert('end', self.format(record) + '\n')
        self.widget.see('end')
        self.widget.configure(state='disabled')
        self.widget.update()

class LoggingWindow:
    def __init__(self, parent:tk.Tk):
        self.root = Toplevel(parent)
        self.root.title('Process Log')
        self.root.geometry('500x220')
        self.root.resizable(False,False)

        self.output = ScrolledText(self.root)
        self.output.pack()

class AdvancedWindow(Toplevel):
    def __init__(self, parent:tk.Tk, selected_combo:list[list[str]]):
        Toplevel.__init__(parent)
        self.root = Toplevel(parent)
        self.root.title('Advanced Edit')
        self.root.geometry('400x250')

        self.info_frame = Frame(self.root)
        self.starter_box = Entry(self.info_frame)
        self.starter_box.insert(selected_combo[0])
        self.starter_box.config(state='disabled')
        self.ender_box = Entry(self.info_frame)
        self.ender_box.insert(selected_combo[1])
        self.ender_box.config(state='disabled')

        self.add_frame = Frame(self.root)
        self.add_box = ScrollableListbox(self.add_frame)

        for card_name in selected_combo[2]:
            self.add_box.insert('end', card_name)

        self.add_frame_btn = Frame(self.add_frame)
        self.add_button = Button(self.add_frame_btn, text='Add Card')
        self.rem_button = Button(self.add_frame_btn, text='Remove Card')

        self.info_frame.pack(side='left', expand=1, fill='y')
        Label(self.info_frame, text='Starter Set').pack()
        self.starter_box.pack()
        Label(self.info_frame, text='Ender Set').pack()
        self.ender_box.pack()

        self.add_frame.pack(side='right', expand=1, fill='both')
        Label(self.add_frame, text='Additional Cards').pack()
        self.add_box.pack(expand=1, fill='both')
        self.add_frame_btn.pack(side='bottom', fill='x')
        self.add_button.pack(side='left', expand=1, fill='x')
        self.rem_button.pack(side='right', expand=1, fill='x')

    def show(self):
        self.root.mainloop()

def unique_cards(wrap:ComboWrapper, card_db:list[dict[str, str]]=None, **u_cards_list) -> dict:
    for key, value in wrap.combo_json.items():
        for item in ast.literal_eval(key):
            if item not in u_cards_list:
                start = timeit.default_timer()
                u_cards_list[item] = sanit_fuzzy(item, fuzzy_search(item))
                end = timeit.default_timer()
                if end - start < MANUAL_RATE_LIMIT:
                    logger.warning(f'forcing rate limit (dl_time={end-start})')
                    time.sleep(MANUAL_RATE_LIMIT - (end-start))

        for item in value:
            for subitem in item:
                if subitem not in u_cards_list:
                    start = timeit.default_timer()
                    u_cards_list[subitem] = sanit_fuzzy(subitem, fuzzy_search(subitem))
                    end = timeit.default_timer()
                    if end - start < MANUAL_RATE_LIMIT:
                        logger.warning(f'forcing rate limit (dl_time={end-start})')
                        time.sleep(MANUAL_RATE_LIMIT - (end-start))
    
    return u_cards_list

def sanit_fuzzy(name:str, fuzzy_result:list[dict[str, str]]) -> dict[str:str]:
    card_json = None
    for term in fuzzy_result:
        if term['name'].lower() == name.lower():
            card_json = term
            break
    return card_json

def load_card_db() -> list[dict[str, str]]:
    current_db_version = get_db_version()
    local_db_version = 0
    if not os.path.exists(LOCAL_PATH):
        os.mkdir(LOCAL_PATH)
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    try:
        with open('cdb_ver.txt', 'r') as file:
            local_db_version = file.readline()
    except:
        with open('cdb_ver.txt', 'w') as file:
            logger.warning('no local db version')
            local_db_version = 0
            file.write(str(local_db_version))

    logger.info(f'local db:{local_db_version}')

    if current_db_version != local_db_version:
        logger.info(f'newer db is available ({current_db_version}), updating...')
        os.remove('cdb.txt')
        with open('cdb.txt', 'w', encoding='utf8') as file:
            file.write(str(fuzzy_search('')))
        os.remove('cdb_ver.txt')
        with open('cdb_ver.txt', 'w') as file:
            file.write(current_db_version)
        logger.info('updated local db')

    with open('cdb.txt', 'r') as file:
        return ast.literal_eval(file.readline())
    
def search_cards(search_term:str, card_list:list[dict[str, str]]) -> list[dict[str, str]]:
    try:
        #heard ya like list comprehensions
        search_results = [card for card in card_list if re.compile('^.*{}.*'.format(''.join([c if c not in '.+*?[^]$(){}=!<>|:\\' else '' for c in search_term])), re.IGNORECASE).match(card['name'])]
        
        for term in search_results:
            if term['name'].lower() == search_term.lower():
                search_results.remove(term)
                search_results.insert(0, term)

        return search_results
    except:
        return []

def get_image_list(wrap:ComboWrapper) -> dict[str:np.array]:
    u_cards = unique_cards(wrap)
    stored_img_dict = {}
    for key in u_cards:
        rec_image = get_image(u_cards[key]['image'])
        if rec_image is None:
            logger.warning(f'image is None for name {u_cards[key]}')
            return []
        download_image = cv2.resize(rec_image, HARD_CARD_RESIZE)
        stored_img_dict[key] = download_image

    return stored_img_dict

def bw_text(image:np.array, loc_x:int, loc_y:int, text:str) -> np.array:
    cv2.putText(image, text, 
                (loc_x + HARD_CARD_RESIZE[0] - 50, loc_y + HARD_CARD_RESIZE[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,0),
                3,
                cv2.LINE_AA)
    cv2.putText(image, text, 
                (loc_x + HARD_CARD_RESIZE[0] - 50, loc_y + HARD_CARD_RESIZE[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                1,
                cv2.LINE_AA)
    return image

def render_combo(width:int, height:int, combos:ComboWrapper, write_image:bool=False) -> np.array:
    image = np.zeros((height, width, 3), np.uint8)

    cv2.rectangle(image, (0, 0), (width, height), BG_COLOR, -1)

    card_images = get_image_list(combos)

    offset_y = START_COORD[1]

    for key in combos.combo_json:
        offset_x = START_COORD[0]
        offset_return = offset_x

        starter_images = {name:card_images[name] for name in ast.literal_eval(key)}
        
        for img_name in starter_images:
            logger.info(f'rendering ip: {img_name}')
            try:
                image[  offset_y : HARD_CARD_RESIZE[1] + offset_y,
                        offset_x : HARD_CARD_RESIZE[0] + offset_x
                    ] = starter_images[img_name]
                if ast.literal_eval(key).count(img_name) > 1:
                    logger.info(f'{img_name} has multiple occurences, showing text...')
                    image = bw_text(image, offset_x, offset_y, f'x{ast.literal_eval(key).count(img_name)}')
            except Exception as e:
                logger.warning(f'whoops: {e}')
                logger.warning(f'failed to place image to [{offset_y, offset_x}]')
                logger.warning(f'im_dims=[{starter_images[img_name].shape}]')
                cv2.imshow('name', starter_images[img_name])
                cv2.waitKey(0)

            offset_x += HARD_CARD_RESIZE[0] + CARD_GAP_X
            offset_return = offset_x

        for n in range(len(combos.combo_json[key])):

            ender_images   = {name:card_images[name] for name in combos.combo_json[key][n]}
            
            if n == 0:
                #primary line from starter to end board
                image = cv2.arrowedLine(image, 
                                (offset_x, offset_y + int(HARD_CARD_RESIZE[1] / 2)),
                                (offset_x + 70, offset_y + int(HARD_CARD_RESIZE[1] / 2)),
                                ARROW_COLOR,
                                5, tipLength=0.4)
            else:
                #secondary line breaking off the first 
                image = cv2.line(image,
                                (offset_x + 20, offset_y - int(HARD_CARD_RESIZE[1] / 2) - CARD_GAP_Y),
                                (offset_x + 20, offset_y + int(HARD_CARD_RESIZE[1] / 2)),
                                ARROW_COLOR,
                                5)

                image = cv2.arrowedLine(image, 
                                (offset_x + 20, offset_y + int(HARD_CARD_RESIZE[1] / 2)),
                                (offset_x + 70, offset_y + int(HARD_CARD_RESIZE[1] / 2)),
                                ARROW_COLOR,
                                5, tipLength=0.4)

            offset_x += 80

            for img_name in ender_images:
                logger.info(f'rendering op: {img_name}')
                image[  offset_y : HARD_CARD_RESIZE[1] + offset_y,
                        offset_x : HARD_CARD_RESIZE[0] + offset_x
                    ] = ender_images[img_name]
                
                if combos.combo_json[key][n].count(img_name) > 1:
                    logger.info(f'{img_name} has multiple occurences, showing text...')
                    image = bw_text(image, offset_x, offset_y, f'x{combos.combo_json[key][n].count(img_name)}')
                offset_x += HARD_CARD_RESIZE[0] + CARD_GAP_X

            offset_y += CARD_GAP_Y + HARD_CARD_RESIZE[1]
            offset_x = offset_return
    if write_image:
        logger.info('writing image to out folder')
        name = str(time.time()).replace('.', '') + '.png'
        cv2.imwrite(os.path.join(OUT_PATH, name), image)
        logger.info(f'wrote image to {name}')
    
    logger.info('done render')
    #cv2.imshow('img', image)
    #cv2.waitKey(0)

def image_from_name(name:str) -> np.array:
    card_json = sanit_fuzzy(name, fuzzy_search(name))

    if card_json is None:
        return None

    return get_image(card_json['image'])

def get_image(img_url:str) -> np.array:
    try:
        img_name = img_url.split('/')[-1]
        logger.info(f'checking {img_name}...')
        if os.path.exists(os.path.join(LOCAL_PATH, img_name)):
            logger.info(f'loading {img_name} from local')
            return cv2.imread(os.path.join(LOCAL_PATH, img_name))
        else:
            logger.info(f'downloading new {img_url}')
            req = Request(
                url=img_url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            resp = urlopen(req)
            raw_out = np.asarray(bytearray(resp.read()))
            img = cv2.imdecode(raw_out, -1)
            cv2.imwrite(os.path.join(LOCAL_PATH, img_name), img)
            return img
    except Exception as e:
        logger.warning(e)
        return None

if __name__ == '__main__':
    combo_set = ComboWrapper()
    cdb = load_card_db()

    window = DisplayWindow(cdb, combo_set)
    window.show()